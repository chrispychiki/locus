"""Spend governance — the operator's dollar walls over paid analysis, and the ledger they are enforced against.

Money moves in exactly one place — a billed generation call at the conversation layer (model/protocol.py) — so that layer is where both halves of governance live, inherited by every paid path by construction: before each billed call the declared caps are checked and a hit cap refuses loud (SpendWall), and after each billed call the spend lands on the ledger, non-optionally, computed from the call's persisted usage and the model card's declared per-token prices. The card decides what is billed, the same way its `conversation` declaration decides the client: a card that declares a pricing block has every call governed at those rates, and a card without one — the shipped local cards, plus everything that never generates (countTokens, Files API traffic) — is free, and nothing here ever runs for it.

**The declaration** is `config/spend.toml` under the deployment root — shipped with the repo, so a fresh clone is walled by its own declaration from the first minute; operator-edited in place (never a dashboard), read fresh at every check so an edit takes effect on the next paid call.
Its keys are the dollar caps PERIOD_KEYS names, each over the UTC calendar period `period_bounds` defines; the shipped file's own comment carries the operator-facing explanation.
The file is the only truth: what it declares is the walls, and nothing else has an opinion.
Each key is optional; a missing key — or a missing file — is no cap on that period, because the safety story is the shipped default, not machinery guarding absence: an operator who wants no walls deletes the caps.
An unknown key refuses loud rather than silently not limiting: a typo'd cap is exactly the runaway this module exists to prevent. A cap of 0 refuses every paid call — the kill switch.

**The wall's arithmetic is check-then-spend**: a call is refused when the period's recorded spend has already reached the cap, so one in-flight call can carry spend past a cap by its own cost and the next paid call refuses. Nothing is estimated, degraded, or retried around — the refusal names the cap, the recorded spend, the period, and the declaration to edit. Concurrent analyses race the same read-then-append honestly: each may pass a check the other's spend would have failed, so the overshoot bound is the cost of the calls in flight.

**The ledger** is `spend.jsonl` beside the deployment's events.db — append-only JSON lines, one per billed call, each carrying the UTC epoch-ms timestamp, model, dollars, the token breakdown (fresh input, cached input, output, thoughts; an exhausted-thinking retry's burned attempt folded in), and the path of the persisted meta.json its usage came from. Appends are single O_APPEND writes, safe under concurrent analyses; the ledger records what the wire reported, nothing more and nothing less — a call that dies before the wire reports usage leaves no entry, and a call that fails after the wire reported usage (a terminal status with no usable reply) lands that reported usage all the same. A torn or corrupt line fails every read loud, naming the file and line: spend truth that cannot be read must never silently read as less spend.

**Dollars are usage × declared card prices** (the card's `pricing` block in config/cards/, USD per million tokens: `input_per_mtok`, `cached_input_per_mtok`, `output_per_mtok`; thinking tokens bill at the output rate).
The card is a price cache, and enforcement reads only it — but it is verified best-effort against LiteLLM's public price registry wherever a price is stated (the price report, the first paid call of a process, `locus doctor`): the fetch is memoized per UTC day per machine, a mismatch is named loud with both numbers, and an absent key or unreachable registry reports honestly as unverified, never as wrong.
"""

import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tomllib

SPEND_FILE = "spend.toml"
LEDGER_FILE = "spend.jsonl"

# key in spend.toml → the period it caps; the order here is the order walls report in.
PERIOD_KEYS = {"day_usd": "day", "week_usd": "week", "month_usd": "month"}

REGISTRY_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
REGISTRY_TIMEOUT_S = 10

# Models whose declared prices this process has already verified — the once-per-process throttle on
# the paid-call verification (the per-day throttle on the network is the memo file).
_VERIFIED: set[str] = set()


class SpendWall(RuntimeError):
    """A declared spend cap is reached: the paid call is refused before its request goes to the wire. The message names the cap, the recorded spend, the period, and the declaration whose edit raises it."""


def _root() -> Path:
    from locus.evidence.deployment import deployment_root

    return deployment_root()


def _ledger_path(root: Path) -> Path:
    """The ledger's home under the deployment root: data/spend.jsonl, beside the events.db whose analyses it accounts (deployment.py owns the data directory's name).
    The declaration lives in config/ — it ships with the repo; the ledger is accreted."""
    from locus.evidence.deployment import DATA_DIR

    return root / DATA_DIR / LEDGER_FILE


def _declaration_path(root: Path) -> Path:
    """The declaration's home under the deployment root: config/spend.toml, with the other deployment-wide declarations (deployment.py owns the config directory's name)."""
    from locus.evidence.deployment import CONFIG_DIR

    return root / CONFIG_DIR / SPEND_FILE


def _now_ms() -> int:
    return int(time.time() * 1000)


def load_declaration(root: Path | None = None) -> dict:
    """The deployment's spend.toml as a validated dict — {} when the file does not exist or declares nothing (no caps declared, no walls: the file is the only truth, and the safety story is that the repo ships it with a default cap).
    Unknown keys and non-numeric or negative caps refuse loud: a declaration that silently fails to limit is the failure mode this feature exists against."""
    path = _declaration_path(root or _root())
    if not path.exists():
        return {}
    declared = tomllib.loads(path.read_text())
    unknown = sorted(set(declared) - set(PERIOD_KEYS))
    if unknown:
        raise ValueError(
            f"{path} declares unknown key(s) {unknown} — the spend walls are "
            f"{sorted(PERIOD_KEYS)} (USD caps over UTC calendar periods); an "
            f"unrecognized key would silently cap nothing"
        )
    for key, value in declared.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"{path} declares {key} = {value!r} — a cap is a finite "
                f"number of dollars, 0 or more (0 refuses every paid call)"
            )
    return declared


def period_bounds(period: str, now_ms: int) -> tuple[int, int]:
    """[start, end) of the UTC calendar period containing now_ms, in epoch ms. day: the UTC calendar day. week: the ISO week — Monday 00:00 UTC to the next Monday. month: the UTC calendar month — the 1st to the next 1st."""
    now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "day":
        start, end = midnight, midnight + timedelta(days=1)
    elif period == "week":
        start = midnight - timedelta(days=now.weekday())
        end = start + timedelta(days=7)
    elif period == "month":
        start = midnight.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1)
    else:
        raise ValueError(
            f"unknown period {period!r} — one of {sorted(set(PERIOD_KEYS.values()))}"
        )
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def read_ledger(root: Path | None = None) -> list[dict]:
    """Every entry, oldest first — the whole file is small (one line per billed call). A line that does not parse fails loud with its number: a torn write must never silently read as less spend."""
    path = _ledger_path(root or _root())
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path} line {number} is not valid JSON ({error}) — a torn "
                f"or hand-edited ledger line; every paid call refuses until "
                f"the line is repaired or removed, because spend truth that "
                f"cannot be read must not read as less spend"
            ) from error
    return entries


def _append_ledger(root: Path, entry: dict) -> None:
    path = _ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(entry, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


# The ledger's token vocabulary — the one legal field set for a priced usage dict. Each backend's
# _spend_usage reads its own wire shape into exactly these, stating an unbilled component as an
# explicit 0; anything outside the set refuses loud, because a key the pricing arithmetic does not
# read would silently price its component at $0 on the ledger and against the walls.
USAGE_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "thoughts_tokens")


def validate_usage(usage: dict, backend: str) -> dict:
    """A backend-normalized usage dict checked whole against USAGE_FIELDS before it is priced and recorded — unknown and missing keys refuse loud, naming the key and the backend that produced it."""
    missing = [field for field in USAGE_FIELDS if field not in usage]
    unknown = sorted(set(usage) - set(USAGE_FIELDS))
    if missing or unknown:
        raise ValueError(
            f"{backend} produced a spend usage record outside the ledger's "
            f"vocabulary — missing {missing}, unknown {unknown}; the fields "
            f"are exactly {list(USAGE_FIELDS)} (USAGE_FIELDS in spend.py), "
            f"each an explicit token count, 0 when the wire bills none — a "
            f"key outside them would silently price its component at $0"
        )
    return usage


def call_cost(spend_usage: dict, prices: dict) -> float:
    """One billed call's dollars from its usage in the ledger's token vocabulary (USAGE_FIELDS) — mechanical: fresh input (input minus cached) at the input rate, cached input at the cached rate, output and thinking at the output rate. Every field is required — a backend states an unbilled component as an explicit 0, and a missing field refuses loud rather than pricing as $0. Extra keys are ignored here so a ledger entry, which carries its own metadata beside the counts, re-prices as written."""
    missing = [field for field in USAGE_FIELDS if field not in spend_usage]
    if missing:
        raise ValueError(
            f"usage record is missing {missing} — the ledger's vocabulary is "
            f"exactly {list(USAGE_FIELDS)} (USAGE_FIELDS in spend.py), and a "
            f"missing field must never price as $0"
        )
    fresh = spend_usage["input_tokens"] - spend_usage["cached_tokens"]
    return round(
        (
            fresh * prices["input_per_mtok"]
            + spend_usage["cached_tokens"] * prices["cached_input_per_mtok"]
            + (spend_usage["output_tokens"] + spend_usage["thoughts_tokens"])
            * prices["output_per_mtok"]
        )
        / 1e6,
        8,
    )


def guard_paid_call(billing: dict) -> None:
    """Runs before every billed generation request: the walls, then — once per process per model — the best-effort price verification, loud on drift and silent otherwise (the full verification status is the price report's)."""
    check_walls()
    model = billing["model"]
    if model not in _VERIFIED:
        _VERIFIED.add(model)
        verdict = verify_prices(billing["model_id"], billing["prices"])
        if verdict["status"] == "drift":
            print(
                f"price drift on {model}: {verdict['mismatches']} — the "
                f"ledger and walls run on the declared prices; update the "
                f"card's pricing block in config/cards/{model}.toml if the registry "
                f"is right",
                file=sys.stderr,
            )


def record_paid_call(
    billing: dict,
    *,
    label: str | None,
    usage: dict,
    backend: str,
    record: str | None,
) -> None:
    """One billed call onto the ledger, non-optionally — the backend-normalized usage validated whole against USAGE_FIELDS (`backend` names its producer in the refusal), dollars from call_cost over it, `record` naming the persisted meta.json that usage came from (stored root-relative when it sits under the deployment)."""
    validate_usage(usage, backend)
    root = _root()
    if record is not None:
        try:
            record = str(Path(record).resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    _append_ledger(
        root,
        {
            "ts": _now_ms(),
            "model": billing["model"],
            "label": label,
            "usd": call_cost(usage, billing["prices"]),
            **usage,
            "record": record,
        },
    )


def check_walls(now_ms: int | None = None) -> None:
    """Every declared cap against the ledger's recorded spend in its current period; the first cap already reached refuses with SpendWall. No caps declared, no walls."""
    root = _root()
    declaration = load_declaration(root)
    if not declaration:
        return
    now = _now_ms() if now_ms is None else now_ms
    entries = read_ledger(root)
    for key, period in PERIOD_KEYS.items():
        cap = declaration.get(key)
        if cap is None:
            continue
        start, end = period_bounds(period, now)
        # Rounded exactly as spend_report states it, so a cap set to the
        # reported figure is reached by the wall's own arithmetic — a raw float
        # sum can sit one ulp under the rounded number the operator copied.
        spent = round(sum(e["usd"] for e in entries if start <= e["ts"] < end), 8)
        if spent >= cap:
            raise SpendWall(
                f"{period} spend cap reached: ${spent:.4f} of the ${cap:g} "
                f"{key} cap declared in {_declaration_path(root)} is already spent "
                f"this UTC {period} ({_iso(start)} → {_iso(end)}; ledger "
                f"{_ledger_path(root)}). Paid calls refuse until the period "
                f"turns; raising or removing the cap is an edit to that "
                f"declaration — the operator's call"
            )


def _iso(ms: int) -> str:
    from locus.evidence.clock import utc_stamp

    return utc_stamp(ms)


def spend_report(now_ms: int | None = None) -> dict:
    """Recorded spend against the declaration, per period — the agent's cheap read for budgeting an investigation inside the operator's walls. Reports every period whether or not it is capped; when nothing is declared, says so and names the keys, so the surface that reports spend also teaches the wall."""
    root = _root()
    declaration = load_declaration(root)
    entries = read_ledger(root)
    now = _now_ms() if now_ms is None else now_ms
    periods = {}
    for key, period in PERIOD_KEYS.items():
        start, end = period_bounds(period, now)
        spent = round(sum(e["usd"] for e in entries if start <= e["ts"] < end), 8)
        cap = declaration.get(key)
        periods[period] = {
            "bounds": [_iso(start), _iso(end)],
            "spent_usd": spent,
            "cap_usd": cap,
            **({"remaining_usd": round(cap - spent, 8)} if cap is not None else {}),
        }
    report = {
        "ledger": str(_ledger_path(root)),
        "entries": len(entries),
        "lifetime_usd": round(sum(e["usd"] for e in entries), 8),
        "declaration": (
            str(_declaration_path(root)) if _declaration_path(root).exists() else None
        ),
        "periods": periods,
    }
    if not declaration:
        report["note"] = (
            f"no spend caps declared — paid analysis is unwalled. Declare "
            f"any of {sorted(PERIOD_KEYS)} (USD over UTC calendar periods: "
            f"day = the UTC date, week = Monday 00:00 UTC onward, month = "
            f"the calendar month) in {_declaration_path(root)} and every paid "
            f"call enforces them"
        )
    return report


def _memo_path() -> Path:
    """The registry memo's home: one per machine (locus.evidence.deployment.machine_cache_dir) — the registry's prices belong to no deployment, and the memo exists to hold the fetch to once per UTC day."""
    from locus.evidence.deployment import machine_cache_dir

    return machine_cache_dir() / "price_registry_memo.json"


def _fetch_registry() -> dict:
    """The gemini/-prefixed subset of LiteLLM's registry — the only keys a Gemini card can verify against; the rest of the file is other providers."""
    with urllib.request.urlopen(REGISTRY_URL, timeout=REGISTRY_TIMEOUT_S) as response:
        full = json.load(response)
    return {key: value for key, value in full.items() if key.startswith("gemini/")}


def _registry_today() -> tuple[dict | None, str]:
    """(registry subset, provenance) — memoized per UTC day, failures included, so an offline machine pays one attempt a day, not one per priced call. The memo stores the attempt's full timestamp (the day is derived at read), and the provenance states that timestamp and the refresh convention, so a reader knows exactly what evidence backs the verdict and how stale it can be."""
    path = _memo_path()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if path.exists():
        try:
            memo = json.loads(path.read_text())
        except json.JSONDecodeError:
            memo = None
        if memo and str(memo.get("fetched", ""))[:10] == today:
            if memo.get("registry") is not None:
                return memo["registry"], (
                    f"registry snapshot fetched {memo['fetched']} (refetched at most once per UTC day)"
                )
            return None, (
                f"registry unreachable when tried at "
                f"{memo['fetched']} (retried at most once per UTC "
                f"day): {memo.get('error')}"
            )
    from locus.evidence.clock import utc_stamp

    tried = utc_stamp()
    try:
        registry, error = _fetch_registry(), None
    # Fetching a third-party URL fails as a socket error, an HTTP protocol error, or a
    # body that will not parse, and every one of them is the same verdict — no registry
    # today — which verify_prices reports as `unverified` rather than raising.
    except Exception as fetch_error:  # noqa: BLE001
        registry, error = None, repr(fetch_error)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fetched": tried, "registry": registry, "error": error}))
    tmp.replace(path)
    if registry is None:
        return None, (
            f"registry unreachable when tried at {tried} (retried at most once per UTC day): {error}"
        )
    return registry, (
        f"registry snapshot fetched {tried} (refetched at most once per UTC day)"
    )


def verify_prices(model: str, prices: dict) -> dict:
    """A card's declared prices against LiteLLM's registry entry for `gemini/<model>` — exact-key lookup, no fuzzy matching. Returns status `verified`, `drift` (each mismatch named with both numbers), or `unverified` (key absent, field absent, or registry unreachable — honestly unchecked, never wrong). Never raises and never enforces: the walls and the ledger run on the declaration alone."""
    registry, provenance = _registry_today()
    source = {"source": REGISTRY_URL, "provenance": provenance}
    if registry is None:
        return {"status": "unverified", "reason": provenance, "source": REGISTRY_URL}
    entry = registry.get(f"gemini/{model}")
    if entry is None:
        return {
            "status": "unverified",
            "reason": f"no key gemini/{model} in the registry — the declared prices stand unchecked",
            **source,
        }
    checks = [
        ("input_per_mtok", entry.get("input_cost_per_token")),
        ("cached_input_per_mtok", entry.get("cache_read_input_token_cost")),
        ("output_per_mtok", entry.get("output_cost_per_token")),
        # Thinking bills at the card's one output rate, so a registry that
        # prices reasoning apart is drift against that rate, named as such.
        ("output_per_mtok (reasoning)", entry.get("output_cost_per_reasoning_token")),
    ]
    mismatches, checked = [], 0
    for field, per_token in checks:
        if per_token is None:
            continue
        checked += 1
        declared = prices[field.split(" ")[0]]
        registry_mtok = per_token * 1e6
        if not math.isclose(declared, registry_mtok, rel_tol=1e-6):
            mismatches.append(
                {
                    "field": field,
                    "declared_usd_per_mtok": declared,
                    "registry_usd_per_mtok": registry_mtok,
                }
            )
    if not checked:
        return {
            "status": "unverified",
            "reason": f"gemini/{model} is in the registry but carries none of the cost fields to check against",
            **source,
        }
    if mismatches:
        return {"status": "drift", "mismatches": mismatches, **source}
    return {"status": "verified", **source}
