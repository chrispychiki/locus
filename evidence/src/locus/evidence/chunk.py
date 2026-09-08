"""One stored chunk → (visitor, canonical event) pairs — the decode side of the recorder's wire format.

chunk_pairs is the pure core, transport-free: one decoded chunk payload (the recorder's wire format — visitorId, sliceId, recorderVersion, envelope, events, errors) to hydration-ready pairs; decode_or_skip wraps it in the gunzip and the resilience the open-write store forces. The chunk's recorderVersion becomes _envelope.script_version — the version of the recorder that *captured* the events, which is not always the one that shipped them — and its sliceId becomes _envelope.recorder_slice, the upstream slice identity every event in the chunk belongs to, which slice materialization later keys local slices by. The device facts land whole: device/os/browser derived from the userAgent (with the envelope's touch points and screen correcting the one UA that lies about form factor — parse_user_agent), plus the visitor's language, time zone, and screen — events carry none of it, and only the recording browser could ever have said it. The snippet id is the one piece of identity the chunk does not carry: it lives only in the object key, so the read path parses it out (parse_chunk_key) and passes it in to become _envelope.snippet.

Double-delivery and chunk arrival order are explicitly not this layer's problem: hydration's content-hash dedup and (timestamp, counter) ordering make the hydrated stream invariant to how events were cut into chunks.
"""

import gzip
import json
import zlib
from collections import Counter
from datetime import datetime, timezone

import ua_parser

from .hydrate import event_skip_reason
from .slices import SLICE_ID_SHAPE, slice_open_ms
from .speak import say

# The screen aspect ratio at which a touch device behind a Macintosh UA is a phone rather than a tablet. The device class is a form factor — the shape of the screen the visitor used — so a device used in a tablet shape is a tablet for that page context. Tablet panels run 4:3 to 10:7 (at most ~1.44), phone panels 16:9 to 19.5:9 (at least ~1.78); the cut sits at the middle of the gap. A window-sized "screen" (iPadOS Stage Manager and Split View report the window, not the panel) keeps a tablet's verdict for as long as the window stays tablet-shaped.
PHONE_ASPECT_FLOOR = 1.6


def parse_user_agent(
    ua: str | None,
    touch_points: int | None = None,
    screen_width: int | None = None,
    screen_height: int | None = None,
) -> dict:
    """UA facts in uap-core's vocabulary: os and browser are the families ua-parser assigns ("Chrome Mobile", "Samsung Internet", "Mac OS X", "Mobile Safari UI/WKWebView", ...), so the maintained regexes — not this module — own which token wins when a UA carries several (every Chromium skin also says Chrome/) and what an engine-only webview is called. uap-core names families, never form factors, so the three-bucket device fact reads the UA's own form-factor markers: iPad or Android-without-Mobile is a tablet, iPhone or Mobile is mobile, anything else desktop. Unknowns stay absent rather than guessed — no match, and the family "Other" uap-core assigns when it can only say "not that", both leave the fact out. iPadOS desktop-mode UAs read as Mac OS X on a desktop — a lie in the UA itself (https://webkit.org/blog/9674/new-webkit-features-in-safari-13/: "With the exception of iPad mini, Safari on iPad will now send a user-agent string that is identical to Safari on macOS") — and an iPhone with Request Desktop Website on sends the same UA. touch_points, the envelope's navigator.maxTouchPoints, separates both from a real Mac, which reports 0; only that shape is overridden — a touch-screen Windows or Linux machine is honestly a desktop. The envelope's screen separates the two touch devices by aspect ratio (PHONE_ASPECT_FLOOR), the one envelope fact that differs between them; no usable screen leaves the verdict at tablet. Chunks from recorders that predate the field carry no touch_points and keep the lie, as recorded."""
    if not ua:
        return {}
    facts = {}
    tablet = "iPad" in ua or ("Android" in ua and "Mobile" not in ua)
    mobile = not tablet and ("iPhone" in ua or "Mobile" in ua)
    if (
        not tablet
        and not mobile
        and "Macintosh" in ua
        and type(touch_points) is int
        and touch_points > 0
    ):
        if (
            type(screen_width) is int
            and type(screen_height) is int
            and min(screen_width, screen_height) > 0
            and max(screen_width, screen_height) / min(screen_width, screen_height)
            >= PHONE_ASPECT_FLOOR
        ):
            mobile = True
        else:
            tablet = True
    facts["device"] = "tablet" if tablet else "mobile" if mobile else "desktop"
    parsed = ua_parser.parse(ua)
    if parsed.os is not None and parsed.os.family != "Other":
        facts["os"] = parsed.os.family
    if parsed.user_agent is not None and parsed.user_agent.family != "Other":
        facts["browser"] = parsed.user_agent.family
    return facts


def parse_chunk_key(key: str) -> dict | None:
    """The read side of the store's object-key layout, whose one declaration is the write side that mints it — store/src/keys.js (chunkTarget). None for a key that is not a chunk: an open-write bucket holds whatever anyone put in it, and an inventory that silently dropped what it cannot parse would lie about the bucket.

    This is the only place in Locus that takes a key apart; moving the layout is a change on both sides of the store."""
    parts = key.split("/")
    if len(parts) != 5:
        return None
    snippet, date, visitor_id, slice_id, chunk = parts
    if not SLICE_ID_SHAPE.match(slice_id):
        return None
    return {
        "snippet": snippet,
        "date": date,
        "visitor_id": visitor_id,
        "slice_id": slice_id,
        "chunk": chunk,
    }


# The characters that spell a pattern. The store matches a prefix literally, and the bucket is open-write — any byte can sit in a listable key — so these three are refused and nothing else is: a prefix carrying one asks for a match the store cannot perform, and would otherwise read as an empty store.
_PATTERN_CHARS = frozenset("*?[")


def prefix_skip_reason(prefix: str) -> str | None:
    """Why a key prefix asks the store for something it cannot do — None when it doesn't."""
    found = sorted({c for c in prefix if c in _PATTERN_CHARS})
    if not found:
        return None
    shown = " ".join(repr(c) for c in found)
    return (
        f"{shown} spells a pattern, and the store matches a prefix literally — a prefix is a "
        f"leading substring of a key (snippet/YYYY-MM-DD/visitor/slice/chunk); there are no wildcards"
    )


def chunk_pairs(payload: dict, snippet: str | None = None) -> list[tuple[str, dict]]:
    wire = payload.get("envelope") or {}
    screen = wire.get("screen") if isinstance(wire.get("screen"), dict) else {}
    envelope = {
        **parse_user_agent(
            wire.get("userAgent"),
            wire.get("maxTouchPoints"),
            screen.get("width"),
            screen.get("height"),
        ),
        "language": wire.get("language"),
        "time_zone": wire.get("timeZone"),
        "screen_width": screen.get("width"),
        "screen_height": screen.get("height"),
        "snippet": snippet,
        "script_version": payload.get("recorderVersion"),
        "recorder_slice": payload.get("sliceId"),
    }
    return [
        (payload["visitorId"], {**event, "_envelope": envelope})
        for event in payload["events"]
    ]


def chunk_skip_reason(payload) -> str | None:
    """Why a decoded payload is not a recorder chunk — None when it is. The write gate is shape-only (the worker never parses bodies), so bytes that decode fine may still not be a chunk; this is the payload-level half of the decode gate, per-event shape being hydrate.event_skip_reason's.

    The recorder stamps a slice id and its own version on every chunk it builds, so a payload without them is not one. Admitting it would carry an event with no slice id into slice materialization, which cannot recover a slicing and would stop the whole load; rejecting it here names the loss by its key instead."""
    if not isinstance(payload, dict):
        return "payload is not an object"
    visitor = payload.get("visitorId")
    if not (isinstance(visitor, str) and visitor):
        return "visitorId is not a string"
    slice_id = payload.get("sliceId")
    if not isinstance(slice_id, str) or not SLICE_ID_SHAPE.match(slice_id):
        return "sliceId is not a recorder slice id"
    version = payload.get("recorderVersion")
    if not (isinstance(version, str) and version):
        return "recorderVersion is not a string"
    envelope = payload.get("envelope")
    if envelope is not None and not isinstance(envelope, dict):
        return "envelope is not an object"
    if isinstance(envelope, dict):
        ua = envelope.get("userAgent")
        if ua is not None and not isinstance(ua, str):
            return "userAgent is not a string"
    if not isinstance(payload.get("events"), list):
        return "events is not a list"
    errors = payload.get("errors")
    if errors is not None and not isinstance(errors, list):
        return "errors is not a list"
    return None


def _wire_payload(wire: bytes):
    """The JSON payload out of what the wire delivered: parsed directly, or inflated first when the bytes arrived deflate-compressed — zlib-wrapped and raw are both real emitter outputs, so both framings are tried. Raises like json.loads/zlib on bytes that are neither."""
    try:
        return json.loads(wire)
    except (json.JSONDecodeError, UnicodeDecodeError):
        try:
            return json.loads(zlib.decompress(wire))
        except zlib.error:
            return json.loads(zlib.decompress(wire, -zlib.MAX_WBITS))


def decode_or_skip(
    blob: bytes, key: str, snippet: str | None = None, on_errors=None, on_clean=None
) -> list[tuple[str, dict]]:
    """Load-loop resilience around the strict decode. The store is open-write by shape (the worker checks key grammar and size, never contents), so whatever the write gate could not see fails here, one loss at a time, each named by its chunk: bytes that don't decode skip the chunk; a decodable object that isn't a chunk skips the chunk; a chunk whose events[] carries malformed entries skips those entries and lands the rest.

    Each is said on the run's stream, one whole line, from whichever thread decoded the chunk.

    on_clean(key) fires only for a real chunk that lost no events. Its caller records those in the manifest and stops re-fetching them; an object with any loss is never recorded, so a later load fetches it again and re-reports it — the reported line is the loss's one report, re-derived from the bucket rather than a stored copy that could drift.

    Every stored object is gzip — the worker canonicalizes non-gzip arrivals byte-exact under the wrapper (store/src/keys.js states why the bytes are not a gate) — so one gunzip yields exactly what the wire delivered. Usually that is the JSON payload; a visitor's environment can hand it over deflate-compressed instead (the classic broken-CompressionStream output), so a parse failure retries through both deflate framings before the skip is named.

    A decodable chunk whose payload carries `errors` — the recorder's own error log riding home with its data — reports them through on_errors(key, payload) to land in `chunk_errors` (db.py)."""
    try:
        payload = _wire_payload(gzip.decompress(blob))
    except (
        OSError,
        EOFError,
        zlib.error,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as e:
        say(f"locus: skipping corrupt chunk {key}: {type(e).__name__}: {e}")
        return []
    reason = chunk_skip_reason(payload)
    if reason is not None:
        say(f"locus: skipping corrupt chunk {key}: {reason}")
        return []
    kept = []
    skipped = Counter()
    for event in payload["events"]:
        event_reason = event_skip_reason(event)
        if event_reason is None:
            kept.append(event)
        else:
            skipped[event_reason] += 1
    if skipped:
        detail = "; ".join(f"{n} x {r}" for r, n in skipped.items())
        say(
            f"locus: skipping {sum(skipped.values())} malformed events in chunk {key}: {detail}"
        )
    elif on_clean is not None:
        on_clean(key)
    if on_errors is not None and payload.get("errors"):
        on_errors(key, payload)
    return chunk_pairs({**payload, "events": kept}, snippet)


def slice_date(slice_id: str) -> str:
    return datetime.fromtimestamp(
        slice_open_ms(slice_id) / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d")
