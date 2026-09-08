"""`locus ae` — SQL against the deployment's Analytics Engine, answered inside a self-describing envelope.

The connection is analytics_engine.py's; this module owns the verb's judgment aids: the `ms('<UTC instant>')` expansion for the epoch-ms double columns, the sampling probe every result carries, and the zero diagnosis — a zero-shaped answer looks identical whether the query matched nothing, the plane is lagging or down, or a position was misnamed, so the envelope leads with the facts that disambiguate it: the dataset's latest row, the declared row layout, and what a zero cannot say for itself.
"""

import re
from datetime import datetime, timezone

from locus.evidence.analytics_engine import TELEMETRY_SCHEMA, telemetry_schema
from locus.evidence.clock import utc_stamp
from locus.evidence.deployment import ae_dataset as _ae_dataset
from locus.evidence.deployment import deployment_root

# What a zero cannot say for itself, at the end of the zero diagnosis: the two live
# readings a fresh latest-row figure cannot rule out.
AE_ZERO_NOTE = (
    "a datapoint is queryable ~30–60s after its event; a stale "
    "latest row is `locus usage` territory — it counts worker "
    "invocations from outside the worker, telling quiet traffic "
    "from dead ingestion"
)


def _utc_ms(value: str) -> int:
    """A UTC instant as epoch ms: bare digits pass through as ms (13 digits — epoch seconds would silently read as 1970, so they are refused); anything else parses as ISO (date or datetime), naive values read as UTC."""
    if value.isdigit():
        if len(value) < 13:
            raise SystemExit(
                f"{value!r} is too short for epoch milliseconds — pass ms (13 digits) or an ISO date/datetime"
            )
        return int(value)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"{value!r} is not an ISO date/datetime or epoch milliseconds")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _zero_shaped(rows: list) -> bool:
    """A zero-shaped result — the answer reads as absence: no rows at all, or rows carrying no nonzero figure and no non-numeric content. `SELECT count()` against nothing that exists returns one well-formed row holding 0, the same absence as an empty set."""
    for row in rows:
        for value in row.values():
            if isinstance(value, (int, float)):
                if value != 0:
                    return False
            elif isinstance(value, str) and value.strip():
                try:
                    if float(value) != 0:
                        return False
                except ValueError:
                    return False
    return True


def _ae_layout() -> list[str]:
    """The dataset's row layout, rendered from its one declaration — the same file the worker writes rows by. Served with a zero-shaped result because a position error manufactures exactly that shape."""
    schema = telemetry_schema(deployment_root())
    positioned = {
        int(pos): name for pos, name in schema["blobs"].items() if pos.isdigit()
    }
    # The uniform head is the contiguous numeric run from position 1; the gap the
    # schema leaves for each metric's own blobs is where the head ends, and the
    # numeric keys past it are the uniform tail — derived from the declaration's
    # own shape, so widening either region never leaves a stale boundary here.
    positions = sorted(positioned)
    head_len = 1
    while (
        head_len < len(positions) and positions[head_len] == positions[head_len - 1] + 1
    ):
        head_len += 1
    head = [f"blob{pos}={positioned[pos]}" for pos in positions[:head_len]]
    tail = [f"blob{pos}={positioned[pos]}" for pos in positions[head_len:]]
    lines = [
        f"layout ({TELEMETRY_SCHEMA}): "
        f"index1={schema['index1']}; "
        + " ".join(head)
        + " on every row, then per metric:"
    ]
    domains = schema.get("blob_domains", {})
    for metric, decl in schema["metrics"].items():
        parts = [
            f"blob{pos}={name}" + (f" [{domains[name]}]" if name in domains else "")
            for pos, name in sorted(decl["blobs"].items(), key=lambda kv: int(kv[0]))
        ] + [
            f"double{pos}={name}"
            for pos, name in sorted(decl["doubles"].items(), key=lambda kv: int(kv[0]))
        ]
        lines.append(
            f"  {metric}: "
            + (" ".join(parts) if parts else "(uniform blobs only)")
            + (f" — {decl['note']}" if "note" in decl else "")
        )
    lines.append(
        "  every metric: "
        + " ".join(tail)
        + f"; unmeasured double = {schema['unmeasured_sentinel']}"
    )
    return lines


def _ae_latest(dataset: str) -> str:
    """The dataset's most recent row, phrased for a zero-result answer — the fact that separates "matched nothing on a live plane" from "the plane itself is quiet or down"."""
    from locus.evidence.analytics_engine import ae_query

    probe = ae_query(f"SELECT max(timestamp) AS latest FROM {dataset}")
    latest = (probe.get("data") or [{}])[0].get("latest")
    if not latest:
        return "none — no rows anywhere in this dataset's retention"
    then = datetime.strptime(latest, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    seconds = max(0, int((datetime.now(timezone.utc) - then).total_seconds()))
    ago = (
        f"{seconds}s"
        if seconds < 120
        else f"{seconds // 60}m"
        if seconds < 7200
        else f"{seconds // 3600}h"
        if seconds < 172800
        else f"{seconds // 86400}d"
    )
    return f"{latest} UTC ({ago} ago)"


def run_ae(sql: str) -> dict:
    """One `locus ae` invocation's whole result: the query sent (ms() expanded client-side), and the envelope the rows ride in — anchor and query first, the zero diagnosis when the shape calls for one, the sampling state, the layout pointer, the rows last."""
    from locus.evidence.analytics_engine import ae_query

    expanded = re.sub(
        r"\bms\(\s*(['\"])(.*?)\1\s*\)", lambda m: str(_utc_ms(m.group(2))), sql
    )
    result = ae_query(expanded)
    # One envelope: the anchor, the query, and the sampling state ride inside the
    # same artifact as the rows, so however the file travels the facts that judge
    # the numbers travel with it — and they lead it, because a number read
    # without them can be wrong by an unknowable factor.
    envelope: dict = {"current_timestamp": utc_stamp(), "sql": sql}
    dataset = _ae_dataset()
    rows = result.get("data", [])
    # A zero-shaped answer is a diagnosis, not a measurement: it looks identical
    # whether the query matched nothing, the plane is lagging or down, or a
    # position was misnamed. The disambiguating facts ride ahead of the rows.
    zero = dataset is not None and _zero_shaped(rows)
    if zero:
        envelope["zero"] = {
            "latest_row": _ae_latest(dataset),
            "layout": _ae_layout(),
            "note": AE_ZERO_NOTE,
        }
    if dataset is None:
        envelope["sampling"] = (
            "unprobed — this deployment declares no AE dataset to probe"
        )
    else:
        # AE samples under load, and a sampled row stands for _sample_interval
        # datapoints. The query is the caller's — its window is unknowable
        # here — so the probe asks the only question that holds for any query:
        # has this dataset sampled anywhere in what it still retains?
        probe = ae_query(f"SELECT max(_sample_interval) AS m FROM {dataset}")
        interval = int((probe.get("data") or [{}])[0].get("m") or 1)
        envelope["sampling"] = (
            f"this dataset has sampled within its retention (max "
            f"_sample_interval {interval}) — counting rows undercounts; count "
            f"with sum(_sample_interval) unless your window is known unsampled"
            if interval > 1
            else "none anywhere in this dataset's retention — rows are the datapoints"
        )
    envelope["schema"] = (
        f"blob/double positions are named only in {deployment_root() / TELEMETRY_SCHEMA} — "
        "a position read without it is a guess"
    )
    headline = (
        f"zero-shaped result ({len(rows)} rows, no nonzero figure) — "
        f"latest row in {dataset}: {envelope['zero']['latest_row']}"
        if zero
        else f"{len(rows)} rows"
    )
    return {
        "current_timestamp": envelope.pop("current_timestamp"),
        "headline": headline,
        **envelope,
        "rows": rows,
    }
