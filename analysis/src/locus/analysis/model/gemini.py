"""Gemini-direct backend — the stateful Interactions API, images via the Files API.

Images are never inlined: every image uploads through the Files API and is referenced by URI, so request size never bounds visual context. Uploads are cached in a local SQLite table keyed by API key and content hash — the same image bytes hit the cache from any path, so analyses that hardlink screenshots into their own directories never re-upload what an earlier analysis already sent, while a remote file is readable only through the key that uploaded it, so a rotated key's rows simply miss and re-upload rather than serving URIs the new key cannot read; Gemini holds files for 48h (https://ai.google.dev/gemini-api/docs/files: "Files are stored for 48 hours"), the cache trusts them for 47h. Remote storage is bounded where it grows: an upload that would push live bytes past MAX_STORAGE_BYTES first deletes remote files older than PROTECTED_AGE_S, oldest first, forgetting a cache row only when the remote delete actually succeeded.

The conversation is stateful at the server (store=true, each turn chaining previous_interaction_id): the wire carries only what this turn adds, the server holds the history exactly as the model produced it — thought signatures included, so reasoning continuity costs no client bookkeeping.
What the model sees does not change with the transport shape: every turn re-reads (and re-bills) the whole conversation, and the record persisted at the protocol layer is the full conversation as this client holds it.
Anything a successful call carried is history by construction: the server's copy of the conversation holds it (previous_interaction_id), and the record keeps it for the same reason.

Image cost is per-part media resolution, flat within a tier whatever the pixels (GEMINI_IMAGE_TOKENS, the Gemini 3 token table from https://ai.google.dev/gemini-api/docs/media-resolution). The card's media_resolution is a table, {kind: tier} — `activity_screenshots` for images added with the pushed flag, `pulled_screenshots` for everything else. A table with no `activity_screenshots` rides its `pulled_screenshots` tier for every image; a table with no `pulled_screenshots` rides the API's default tier for pulls. Appending parts never perturbs an earlier turn's bytes, so implicit prefix caching is unaffected by anything a later turn adds; cached-token counts arrive in the usage record and the discount is dollars, never context.
"""

import hashlib
import logging
import sqlite3
import sys
import time
import typing
from pathlib import Path
from typing import Any

from .cards import model_id, pricing, thinking_levels
from .protocol import Conversation, DryRun, Response, UnusableReply, json_schema_of

GEMINI_FILE_TTL_S = 48 * 3600
CACHE_READ_TTL_S = 47 * 3600
PROTECTED_AGE_S = 2 * 3600
MAX_STORAGE_BYTES = 19 * 1024**3
UPLOAD_TIMEOUT_S = 300
UPLOAD_CONCURRENCY = 16


def _literal_values(union) -> tuple[str, ...]:
    """The string literals of an SDK open-union type (Union[Literal[...], UnrecognizedStr]) — read from the installed SDK rather than hand-typed, so a value the API adds or drops is never silently wrong."""
    for arg in typing.get_args(union):
        values = typing.get_args(arg)
        if values and all(isinstance(v, str) for v in values):
            return values
    raise TypeError(f"no string literals found in {union!r}")


def media_resolutions() -> tuple[str, ...]:
    """The settable per-image resolution tiers, derived from the installed SDK."""
    from google.genai.interactions import MediaResolution

    return _literal_values(MediaResolution)


# Per-image token cost by resolution tier — the Gemini 3 image column of the
# media-resolution docs (https://ai.google.dev/gemini-api/docs/media-resolution,
# read 2026-07-22); an unset tier resolves to the API default, which the same
# table states equals high. The doc numbers err safe (high measured 1076-1105
# across 64x64 through 1920x921, content- and transport-independent, perfectly
# additive); billed usage is the exact record.
GEMINI_IMAGE_TOKENS = {
    None: 1120,
    "low": 280,
    "medium": 560,
    "high": 1120,
    "ultra_high": 2240,
}


def image_tier_tokens(resolution: str | None) -> int:
    """What one image costs at a resolution tier, whatever its pixels."""
    if resolution not in GEMINI_IMAGE_TOKENS:
        raise ValueError(
            f"unknown media resolution {resolution!r} — tiers: {media_resolutions()}"
        )
    return GEMINI_IMAGE_TOKENS[resolution]


def default_cache_path() -> Path:
    """The upload cache's home: one per machine (locus.evidence.deployment.machine_cache_dir). What it caches — an image's bytes already uploaded to the Files API — belongs to the API key, not to any deployment or working directory, and a cache that moved with the shell's cwd would silently re-upload everything from a different one."""
    from locus.evidence.deployment import machine_cache_dir

    return machine_cache_dir() / "gemini_uploads.db"


def _client(**kwargs):
    """The one Gemini client constructor, so the missing-key case fails the same way from every entry point (countTokens, a conversation). The SDK raises a bare ValueError without a key; this rewrites it into where the key goes and the free local alternative."""
    from google import genai

    try:
        return genai.Client(**kwargs)
    except Exception as error:
        if "api key" in str(error).lower():
            raise RuntimeError(
                "a Gemini model needs a GOOGLE_API_KEY and none is set. Put it in "
                "store/.env beside the Cloudflare token — Locus self-serves "
                "it from there, so it never passes through the agent — or analyze "
                "on a local model instead (free; needs the bundled mlx server and a "
                "capable Apple-Silicon machine). Key: "
                "https://ai.google.dev/gemini-api/docs/api-key"
            ) from error
        raise


class _WhereverStderrIs(logging.Handler):
    """A handler that finds stderr when it writes. It is installed once per process while where stderr goes is decided per run, so one holding the stream it was built on would write into the first run's stream for the life of the process, closed or not."""

    _locus_retry_narration = True

    def emit(self, record) -> None:
        print(self.format(record), file=sys.stderr, flush=True)


def _narrate_retries() -> None:
    """Route the SDK's retry waits to stderr. The client absorbs transient pushback (429s, timeouts) with exponential backoff that can quietly stretch to minutes, and tenacity announces each wait only on the SDK's own logger — unhandled, that stretch reads as a hang to whoever is watching the call. A stderr line per wait makes the backoff visible as the polite retrying it is; terminal failures still raise and speak for themselves. Idempotent, so every conversation can assert it."""
    logger = logging.getLogger("google_genai._api_client")
    if any(getattr(h, "_locus_retry_narration", False) for h in logger.handlers):
        return
    handler = _WhereverStderrIs()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)


def _add_property_ordering(schema):
    """Gemini 3 generates an object's keys in propertyOrdering order and, absent it, can emit malformed structured output; pydantic's json_schema omits the field. Stamp each object with its declared property order (recursing through $defs, nested objects, and array items) so the order the model is told to emit matches the schema it was given."""
    if isinstance(schema, dict):
        if isinstance(schema.get("properties"), dict):
            schema["propertyOrdering"] = list(schema["properties"])
        for value in schema.values():
            _add_property_ordering(value)
    elif isinstance(schema, list):
        for item in schema:
            _add_property_ordering(item)
    return schema


def count_text_tokens(model: str, texts: list[str]) -> int:
    """Exact prompt-token count of text parts from Gemini's own countTokens — free, no generation. The parts are sent as one user content mirroring how composition sends them, so the count is the wire's own arithmetic, not an estimate. Images are deliberately absent: their cost is flat per image within a tier, so callers add image_tier_tokens instead of uploading anything."""
    from google.genai import types

    client = _client()
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=t) for t in texts if t]
    )
    return client.models.count_tokens(model=model, contents=[content]).total_tokens


_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    key_fp       TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    uri        TEXT NOT NULL,
    mime_type  TEXT NOT NULL,
    file_name  TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    uploaded_at REAL NOT NULL,
    PRIMARY KEY (key_fp, content_hash)
);
"""


def _key_fingerprint(client) -> str:
    """The upload cache's scope, derived from the key the constructed client actually authenticates with — whatever env precedence the SDK applied, the fingerprint names that credential, the one whose project owns the Files API storage the cached URIs point into."""
    key = client._api_client.api_key
    if not key:
        raise RuntimeError(
            "the Gemini client resolved no API key — the upload cache scopes its rows "
            "by key and cannot scope without one"
        )
    return hashlib.sha256(f"gemini-uploads:{key}".encode()).hexdigest()[:16]


class UploadCache:
    """Uploaded-file rows scoped to one API key's fingerprint. A Files API file belongs to the project of the key that uploaded it, so a row is only ever a hit for the key that made it: reads, the storage accounting, and reclaim candidates all scope to `key_fp`, while time-based expiry is global — an expired row is dead under every key. A cache file predating key scoping is rebuilt empty, said aloud on stderr: its rows name no owner, and re-uploading is free."""

    def __init__(self, path: str | Path, key_fingerprint: str):
        self.path = str(path)
        self.key_fp = key_fingerprint
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(uploads)")
            }
            if columns and "key_fp" not in columns:
                conn.execute("DROP TABLE uploads")
                print(
                    f"upload cache {self.path} predates key-scoped rows and names no "
                    f"owner for them — rebuilt empty; anything still needed re-uploads "
                    f"(free)",
                    file=sys.stderr,
                )
            conn.executescript(_CACHE_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, content_hash: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM uploads WHERE key_fp = ? AND content_hash = ? AND uploaded_at > ?",
                (self.key_fp, content_hash, (time.time() - CACHE_READ_TTL_S) * 1000),
            ).fetchone()

    def put(
        self,
        content_hash: str,
        uri: str,
        mime_type: str,
        file_name: str,
        size_bytes: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO uploads "
                "(key_fp, content_hash, uri, mime_type, file_name, "
                " size_bytes, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.key_fp,
                    content_hash,
                    uri,
                    mime_type,
                    file_name,
                    size_bytes,
                    int(time.time() * 1000),
                ),
            )

    def forget(self, file_names: list[str]) -> None:
        """By file name, unscoped: a remotely deleted file is gone for every key that could name it."""
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM uploads WHERE file_name = ?", [(n,) for n in file_names]
            )

    def prune_expired(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM uploads WHERE uploaded_at <= ?",
                ((time.time() - GEMINI_FILE_TTL_S) * 1000,),
            )

    def live_bytes(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM uploads WHERE key_fp = ? AND uploaded_at > ?",
                (self.key_fp, (time.time() - GEMINI_FILE_TTL_S) * 1000),
            ).fetchone()
            return row["total"]

    def deletable(self) -> list[sqlite3.Row]:
        """Uploads a reclaim may delete: this key's own — the only ones its credential can delete remotely — still live, and old enough that no conversation in flight is likely to still be referencing them."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM uploads WHERE key_fp = ? AND uploaded_at BETWEEN ? AND ? ORDER BY uploaded_at",
                (
                    self.key_fp,
                    (time.time() - GEMINI_FILE_TTL_S) * 1000,
                    (time.time() - PROTECTED_AGE_S) * 1000,
                ),
            ).fetchall()


class GeminiConversation(Conversation):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        *,
        record_dir: str | Path,
        thinking_level: str | None = None,
        media_resolution: dict | None = None,
        max_output_tokens: int | None = None,
        cache_path: str | Path | None = None,
        timeout_ms: int = 600_000,
        dry_run: bool = False,
    ):
        from google.genai import types

        from .cards import MEDIA_RESOLUTION_KINDS

        for kind, tier in (media_resolution or {}).items():
            if kind not in MEDIA_RESOLUTION_KINDS:
                raise ValueError(
                    f"unknown media_resolution key {kind!r} — the kinds are {list(MEDIA_RESOLUTION_KINDS)}"
                )
            if tier not in media_resolutions():
                raise ValueError(
                    f"media_resolution {kind} tier must be one of {media_resolutions()}"
                )
        super().__init__(record_dir, system_prompt)

        self.model = model
        # The card's declared `model` is what every wire call addresses; the card name (self.model)
        # stays the deployment-facing identity — card lookups, the ledger, the records.
        self.model_id = model_id(model)
        # Every generation call on this backend bills the operator, so the card's declared
        # per-token prices are required here, before anything is spent: they are what the
        # protocol layer's spend wall and ledger compute dollars with (_billing).
        self.prices = pricing(model)
        self.thinking_levels = thinking_levels(model)
        if thinking_level is not None and thinking_level not in self.thinking_levels:
            raise ValueError(
                f"thinking_level {thinking_level!r} is not one the model accepts: {self.thinking_levels}"
            )
        self.thinking_level = thinking_level
        self._reply_thinking_level = thinking_level
        self.media_resolution = media_resolution or {}
        self.max_output_tokens = max_output_tokens
        self.dry_run = dry_run
        _narrate_retries()
        self.client = _client(
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(
                    initial_delay=10, jitter=30, attempts=10
                ),
            )
        )
        self.cache = UploadCache(
            cache_path or default_cache_path(), _key_fingerprint(self.client)
        )
        self._pending: list[dict] = []
        self._interaction_id: str | None = None
        self._reply_interaction_id: str | None = None
        self._call_names: dict[str, str] = {}

    def _pending_user_content(self) -> list:
        if self._pending and self._pending[-1]["type"] == "user_input":
            return self._pending[-1]["content"]
        step = {"type": "user_input", "content": []}
        self._pending.append(step)
        return step["content"]

    def _add_user_text(self, text: str) -> None:
        self._pending_user_content().append({"type": "text", "text": text})

    def _add_user_image(self, image_path: str, pushed: bool) -> str:
        """The image rides at its kind's per-part tier — activity_screenshots for a pushed image, pulled_screenshots for everything else; a kind the card's table leaves out rides the API default. Cost is the tier's flat token count whatever the pixels."""
        uri, mime_type, _ = self._upload(image_path)
        item = {"type": "image", "uri": uri, "mime_type": mime_type}
        tier = self._image_tier(pushed)
        if tier is not None:
            item["resolution"] = tier
        self._pending_user_content().append(item)
        return uri

    def _image_tier(self, pushed: bool) -> str | None:
        if pushed:
            return self.media_resolution.get(
                "activity_screenshots", self.media_resolution.get("pulled_screenshots")
            )
        return self.media_resolution.get("pulled_screenshots")

    def image_tokens(self, image_path: str) -> int:
        return image_tier_tokens(self.media_resolution.get("pulled_screenshots"))

    def context_remaining(self) -> int:
        """Generation claims nothing from the input context on this path, so the whole remainder is the conversation's to spend beyond what it carries: the model's context minus the last response's usage total — the wire's own count of everything the conversation held through that reply, persisted in its meta.json — minus the priced delta of parts appended since it (tool-result text at countTokens' free arithmetic, served screenshots at the pulled_screenshots tier's flat price)."""
        from .cards import context_tokens

        return context_tokens(self.model) - self._wire_anchored_used()

    def _billing(self) -> dict:
        return {"model": self.model, "model_id": self.model_id, "prices": self.prices}

    def _spend_usage(self, usage_metadata: dict) -> dict:
        """The interaction's usage fields into the ledger's vocabulary. total_cached_tokens is a subset of total_input_tokens, and an exhausted-thinking retry's burned first attempt was a billed call too, so its usage folds in whole."""

        def read(usage: dict) -> dict:
            return {
                "input_tokens": usage.get("total_input_tokens") or 0,
                "cached_tokens": usage.get("total_cached_tokens") or 0,
                "output_tokens": usage.get("total_output_tokens") or 0,
                "thoughts_tokens": usage.get("total_thought_tokens") or 0,
            }

        total = read(usage_metadata)
        retry = usage_metadata.get("exhausted_thinking_retry")
        if retry:
            for field, count in read(retry["usage"]).items():
                total[field] += count
        return total

    def _usage_total(self, usage_metadata: dict) -> int | None:
        return usage_metadata.get("total_tokens")

    def _delta_text_tokens(self, texts: list[str]) -> int:
        return count_text_tokens(self.model_id, texts)

    def _add_tool_result(self, call_id: str, content: str) -> None:
        self._pending.append(
            {
                "type": "function_result",
                "call_id": call_id,
                "name": self._call_names[call_id],
                "result": content,
            }
        )

    @staticmethod
    def _content_hash(filepath: str) -> str:
        return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()

    def _upload(self, image_path: str) -> tuple[str, str, str]:
        filepath = str(Path(image_path).resolve())
        content_hash = self._content_hash(filepath)
        cached = self.cache.get(content_hash)
        if cached is not None:
            return cached["uri"], cached["mime_type"], cached["file_name"]

        self._reclaim_storage()
        file = self._wire_upload(filepath)
        self.cache.put(
            content_hash,
            file.uri,
            file.mime_type,
            file.name,
            Path(filepath).stat().st_size,
        )
        return file.uri, file.mime_type, file.name

    def _wire_upload(self, filepath: str):
        """The network half of an upload — no cache, no shared state, so prefetch_images can run it from worker threads."""
        file = self.client.files.upload(file=filepath)
        deadline = time.time() + UPLOAD_TIMEOUT_S
        while "PROCESSING" in str(file.state):
            if time.time() > deadline:
                raise TimeoutError(f"upload stuck PROCESSING: {filepath}")
            time.sleep(2)
            file = self.client.files.get(name=file.name)
        if "ACTIVE" not in str(file.state):
            raise RuntimeError(f"upload ended in state {file.state}: {filepath}")
        return file

    def prefetch_images(self, image_paths) -> None:
        """Warm the upload cache for a batch of images concurrently. Only the wire fans out: cache reads, the storage reclaim (sized against the whole batch it is about to land), and cache writes all stay on the calling thread, so SQLite is never touched concurrently. Every success is recorded even when a sibling fails, and the first failure then raises."""
        misses: dict[str, str] = {}
        for path in image_paths:
            filepath = str(Path(path).resolve())
            content_hash = self._content_hash(filepath)
            if content_hash not in misses and self.cache.get(content_hash) is None:
                misses[content_hash] = filepath
        if not misses:
            return
        self._reclaim_storage(
            pending_bytes=sum(Path(f).stat().st_size for f in misses.values())
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        failures = []
        with ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY) as pool:
            futures = {
                pool.submit(self._wire_upload, filepath): (hash_, filepath)
                for hash_, filepath in misses.items()
            }
            for future in as_completed(futures):
                content_hash, filepath = futures[future]
                try:
                    file = future.result()
                # An image's upload fails anywhere inside the provider client, and every
                # one of those failures has the same disposition here.
                except Exception as error:  # noqa: BLE001
                    failures.append(error)
                    continue
                self.cache.put(
                    content_hash,
                    file.uri,
                    file.mime_type,
                    file.name,
                    Path(filepath).stat().st_size,
                )
        if failures:
            raise failures[0]

    def _reclaim_storage(self, pending_bytes: int = 0) -> int:
        """Keep remote storage under MAX_STORAGE_BYTES, checked on the paths that grow it — an actual upload, or a prefetch batch counted before it lands (pending_bytes). Files younger than PROTECTED_AGE_S are never deleted, so a conversation in flight cannot have its own evidence pulled out from under it, and a cache row is forgotten only when the remote delete succeeded, leaving failed deletes visible for the next attempt. Returns how many files were deleted."""
        self.cache.prune_expired()
        if self.cache.live_bytes() + pending_bytes < MAX_STORAGE_BYTES:
            return 0
        deleted, failures = [], []
        for row in self.cache.deletable():
            try:
                self.client.files.delete(name=row["file_name"])
                deleted.append(row["file_name"])
            # A remote delete fails anywhere inside the provider client, and one file's
            # failure must not abort the sweep over the rest.
            except Exception as e:  # noqa: BLE001
                failures.append((row["file_name"], repr(e)))
        self.cache.forget(deleted)
        if failures:
            raise RuntimeError(
                f"reclaiming Files API storage: {len(deleted)} deleted, "
                f"{len(failures)} failed (kept in cache for retry): {failures[:3]}"
            )
        return len(deleted)

    def _lowest_thinking_level(self) -> str:
        return self.thinking_levels[0]

    def _get_response(
        self,
        *,
        response_schema: Any = None,
        tools: list | None = None,
    ) -> Response:
        self._reply_thinking_level = self.thinking_level
        self._reply_interaction_id = None
        interaction = self._generate(response_schema, self.thinking_level, tools=tools)

        # An overrun of max_output_tokens ends the interaction as status
        # 'incomplete' (measured live 2026-07-23: a 200 cap consumed by 192
        # thought tokens plus a 4-token fragment of output — so the status is
        # the signal, and output tokens can be nonzero when it fires). The
        # retry is a second paid call and stays on the record whole: the
        # burned first attempt's usage rides the persisted usage under
        # exhausted_thinking_retry, and meta.json's config states the level
        # that actually produced this reply (_config_manifest reads
        # _reply_thinking_level), never the declared one it fell back from.
        # The burned interaction is never chained from — the retry re-runs the
        # same input against the same previous interaction, leaving the failed
        # attempt an orphan branch in the server's store.
        first_attempt = None
        if self._thinking_exhausted(interaction):
            first_attempt = {
                "thinking_level": self.thinking_level,
                "interaction_id": getattr(interaction, "id", None),
                "usage": self._usage_dict(interaction),
            }
            self._reply_thinking_level = self._lowest_thinking_level()
            interaction = self._generate(
                response_schema,
                self._reply_thinking_level,
                include_thoughts=False,
                tools=tools,
            )

        self._require_usable(interaction, first_attempt)
        response = self._normalize(interaction)
        if first_attempt is not None:
            response.usage_metadata["exhausted_thinking_retry"] = first_attempt
        self._interaction_id = interaction.id
        self._reply_interaction_id = interaction.id
        self._pending = []
        return response

    def _config_manifest(self) -> dict:
        return {
            "model": self.model_id,
            "thinking_level": self._reply_thinking_level,
            "media_resolution": self.media_resolution or None,
            "max_output_tokens": self.max_output_tokens,
            "interaction_id": self._reply_interaction_id,
        }

    def _generate(
        self,
        response_schema: Any,
        thinking_level: str | None,
        include_thoughts: bool = True,
        tools: list | None = None,
    ):
        input_steps = [dict(step) for step in self._pending]

        generation_config: dict = {
            "thinking_summaries": "auto" if include_thoughts else "none"
        }
        if thinking_level is not None:
            generation_config["thinking_level"] = thinking_level
        if self.max_output_tokens is not None:
            generation_config["max_output_tokens"] = self.max_output_tokens

        # Safety filtering: the Interactions API exposes no safety settings,
        # and the default block threshold is Off for Gemini 3 models
        # (https://ai.google.dev/gemini-api/docs/safety-settings, read
        # 2026-07-23) — so the arbitrary web content recordings carry is not
        # silently filtered.
        request: dict = {
            "model": self.model_id,
            "input": input_steps,
            "store": True,
            "generation_config": generation_config,
        }
        if self.system_prompt:
            request["system_instruction"] = self.system_prompt
        if self._interaction_id is not None:
            request["previous_interaction_id"] = self._interaction_id
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["parameters"],
                }
                for t in tools
            ]
        if response_schema is not None:
            request["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": _add_property_ordering(json_schema_of(response_schema)),
            }
        if self.dry_run:
            raise DryRun(
                f"dry run: request to {self.model_id} assembled and persisted, not sent"
            )
        return self.client.interactions.create(**request)

    @staticmethod
    def _usage_dict(interaction) -> dict:
        usage = interaction.usage
        return usage.model_dump(exclude_none=True) if usage is not None else {}

    def _thinking_exhausted(self, interaction) -> bool:
        """An 'incomplete' interaction burned its max_output_tokens before finishing — with the card's ceiling sized ~3x the observed largest visible output, that spend is thinking, so a retry at the lowest thinking level is the recovery. A conversation already at the lowest level has no level to fall to and fails loud instead (_require_usable)."""
        return (
            getattr(interaction, "status", None) == "incomplete"
            and self.thinking_level != self._lowest_thinking_level()
        )

    def _require_usable(self, interaction, first_attempt: dict | None = None) -> None:
        """completed carries the reply and requires_action carries tool calls; every other terminal status is a reply the caller must not mistake for one — raised as UnusableReply carrying the usage the wire reported (a burned exhausted-thinking attempt folded in), so the failed call's spend still reaches the meta and the ledger."""
        status = getattr(interaction, "status", None)
        if status not in ("completed", "requires_action"):
            usage = self._usage_dict(interaction)
            if first_attempt is not None:
                usage["exhausted_thinking_retry"] = first_attempt
            raise UnusableReply(
                f"interaction {getattr(interaction, 'id', None)!r} ended {status!r} — see the persisted payloads",
                usage,
            )

    def _normalize(self, interaction) -> Response:
        """The reply out of the interaction's steps. steps can echo the input's own steps back (include_input on a get), so only what follows the last user_input step reads as this reply."""
        from .protocol import ToolCall

        texts: list[str] = []
        thoughts: list[str] = []
        tool_calls: list = []
        for step in interaction.steps or []:
            kind = getattr(step, "type", None)
            if kind == "user_input":
                texts, thoughts, tool_calls = [], [], []
            elif kind == "model_output":
                for item in step.content or []:
                    if getattr(item, "type", None) == "text" and item.text:
                        texts.append(item.text)
            elif kind == "thought":
                for item in step.summary or []:
                    text = getattr(item, "text", None)
                    if text:
                        thoughts.append(text)
            elif kind == "function_call":
                self._call_names[step.id] = step.name
                tool_calls.append(
                    ToolCall(
                        id=step.id, name=step.name, arguments=dict(step.arguments or {})
                    )
                )
        return Response(
            text="".join(texts),
            thoughts_text="\n".join(thoughts),
            usage_metadata=self._usage_dict(interaction),
            tool_calls=tool_calls,
        )

    @property
    def execution_mode(self) -> str:
        return "interactive"
