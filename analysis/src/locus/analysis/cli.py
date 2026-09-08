"""The locus CLI.

One events.db in the deployment's `data/` directory — the home of everything the deployment accretes — found from anywhere in the clone. The store is the deployment's own R2 bucket, declared in store/wrangler.toml with S3 keys self-served from store/.env. Time is UTC throughout: a store key's date part is the slice's open day in UTC, and `ae` filters on UTC timestamps.

A slice id is `<padded-open-ms>-<disambiguator>`, stable across loads and re-derived databases.

Every output's path is derived from what it is — occasion and subject for an analysis, the slice set for a replay payload — never chosen by the caller. An analysis directory is written once and refuses to overwrite: its contents are the evidence for a finding, so a rerun makes a new one rather than editing the evidence behind an answer someone already read. Replay material is the opposite: deterministic derivation, not evidence, so `browse open` refreshes it in place. `browse screenshot` is the one export whose destination belongs to the caller, and that file is the caller's to overwrite.

A run speaks into its own log.
Its first line — `started <stamp> <path>` — is the log's address; everything after lands there, readable while the run still fills it, long spans beating on a clock.
A run that ends clean with a small say repeats the whole of it inline, so the hop to the file is only ever paid for size; a run's result, or its error, lands in the log; a log with neither is a run still working — or one that died mid-work.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from locus.evidence.clock import utc_stamp
from locus.evidence.db import connect
from locus.evidence.deployment import (
    DATA_DIR,
    definitions,
    deployment_root,
    export_declared,
)
from locus.evidence.deployment import ae_dataset as _ae_dataset
from locus.evidence.deployment import store as _store
from locus.evidence.derive import derivations_stale, finish_derivations, stale_stage
from locus.evidence.retention import horizon_cutoff
from locus.evidence.speak import speaking, speaking_into, started

from .ae import TELEMETRY_SCHEMA
from .select_screenshots import SCREENSHOT_INTERVAL_MS

DB_NAME = "events.db"


def _data_root() -> Path:
    """The deployment's `data/` (deployment.py owns the name): the db and every family beside it."""
    return deployment_root() / DATA_DIR


def _definition() -> dict:
    """The operator's session definition, from the deployment's config/definitions.toml — what the derivation path runs under."""
    return definitions(deployment_root())


def _cutoff() -> str:
    """The deployment's retention horizon as a date, from the same declaration the deploy reads — what the load's intake and the sweep both hold the db to (retention.py)."""
    return horizon_cutoff(deployment_root())


def _expire(conn, db: str) -> dict:
    """The retention sweep, run wherever the db is reconciled: recordings past the horizon dropped, with the replay cache beside the db, under the same lock the derivation takes so no pass is reading what this deletes."""
    from locus.evidence.derive import derivation_lock
    from locus.evidence.retention import expire

    with derivation_lock(db):
        return expire(conn, _cutoff(), _definition(), _derived_root(db) / "pages")


def _resolve_db(*, required=True, create=False) -> str | None:
    """The one events.db, pinned: it lives in the deployment's `data/` and nowhere else, and there is no override. load creates it there; the read commands fail loud when it does not exist yet."""
    path = _data_root() / DB_NAME
    if path.exists() or create:
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    if required:
        raise SystemExit(
            f"no {DB_NAME} at {path} — run `locus load <prefix>` first; it creates the db"
        )
    return None


def _derived_root(db_path: str) -> Path:
    """Where the derived families live: plainly beside the db — `analyses/` and `pages/` as its siblings, never inside a wrapper named after the db file."""
    return Path(db_path).resolve().parent


def _subject(rows) -> str:
    """A slice set's subject: the visitor and which of their slices — the first slice's disambiguator, `_plusN` for the rest. The visitor id rides whole: any id a surface emits pastes into an AE equality or an R2 filter as-is, so no surface ever teaches an abbreviated one."""
    first = rows[0]
    slice_part = first["recorder_slice"].rpartition("-")[2] or first["recorder_slice"]
    subject = f"{first['visitor_id']}-{slice_part}"
    if len(rows) > 1:
        subject += f"_plus{len(rows) - 1}"
    return subject


def _analysis_dir(db: str, rows) -> Path:
    """An analysis's identity is occasion + subject: the UTC moment it ran (first, so the family sorts chronologically), then what it analyzed. Two analyses over the same subject are different artifacts by design; a same-second collision fails loud downstream (analysis dirs are never overwritten), never suffixes silently."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return _derived_root(db) / "analyses" / f"{stamp}_{_subject(rows)}"


def _speaking(verb: str):
    """This run speaking into its own log in `outputs/` beside the db (speak.py owns the discipline)."""
    return speaking(_data_root() / "outputs", verb)


def _result(payload: dict) -> None:
    """One JSON line of a run's output, standing alone so the file reads the same half-written as finished. Key order is reading order — whatever states the shape goes first, the rows it summarizes last."""
    print(json.dumps(payload))


def _local_server(model: str, prompt_tokens: int, analyses_dir: Path) -> dict | None:
    """On a local card the price is also a wait: the server's admission state now and, once calls have completed here, the measured pace of this model on this machine. A paid card has neither."""
    from .model.cards import OPENAI_COMPATIBLE, conversation, declared_card, model_id
    from .model.openai_compat import measured_serving, server_admission

    if conversation(model) != OPENAI_COMPATIBLE:
        return None
    base_url = declared_card(model).get("base_url")
    if not base_url:
        raise SystemExit(
            f"card '{model}' declares no base_url — the OpenAI-compatible endpoint "
            "the local backend dials; declare it in the card's file under config/cards/"
        )
    admission = server_admission(base_url)
    served = measured_serving(analyses_dir, model_id(model))
    if admission is None:
        line = f"; no server answers at {base_url} — a run would wait for one"
    else:
        line = (
            f"; the server has {admission['running']} running, "
            f"{admission['waiting']} waiting, served one at a time"
        )
    if served:
        prefill_min = prompt_tokens / served["prompt_tokens_per_s"] / 60
        line += (
            f"; the last {served['calls']} calls here prefilled at "
            f"{served['prompt_tokens_per_s']} tokens/s (this payload: about "
            f"{prefill_min:.1f} min before the first token) and generated "
            f"{served['generated_tokens_median']} tokens at "
            f"{served['generated_tokens_per_s']} tokens/s — about "
            f"{served['served_s_median'] / 60:.0f} min served per call, plus the wait "
            "for admission"
        )
    else:
        line += "; no completed call here yet to measure this model's pace by"
    return {
        "base_url": base_url,
        "admission": admission,
        "measured": served,
        "prefill_estimate_s": (
            round(prompt_tokens / served["prompt_tokens_per_s"]) if served else None
        ),
        "line": line,
    }


def _main_opens(args) -> bool:
    """Whether main opens this run's log under outputs/. It does not for `browse`, which writes none, nor for an analysis run, which speaks into the directory it makes."""
    return not (args.fn is _browse or (args.fn is _analyze and args.run))


def _resolve(conn, recorder_slice: str, visitor: str | None = None) -> sqlite3.Row:
    """The agent-facing slice name → its row. A name may carry its visitor as `<visitor>/<slice>` — the qualified address the collision refusal prints; `visitor` scopes the lookup the same way for callers that carry a recorded visitor (browse open reading a window.json). The integer primary key is a private rowid that exists only inside this process."""
    if visitor is None and "/" in recorder_slice:
        visitor, _, recorder_slice = recorder_slice.rpartition("/")
    clauses, params = ["recorder_slice = ?"], [recorder_slice]
    if visitor:
        clauses.append("visitor_id LIKE ?")
        params.append(visitor + "%")
    rows = conn.execute(
        f"SELECT * FROM slices WHERE {' AND '.join(clauses)}", params
    ).fetchall()
    if not rows:
        scope = f" for visitor {visitor}" if visitor else ""
        hint = (
            " — slices are addressed by recorder slice id "
            "(slices.recorder_slice); the integer primary key is a "
            "private rowid"
            if recorder_slice.isdigit()
            else ""
        )
        raise SystemExit(f"no slice {recorder_slice}{scope}{hint}")
    if len(rows) > 1:
        addresses = " or ".join(f"{row['visitor_id']}/{recorder_slice}" for row in rows)
        raise SystemExit(
            f"slice id {recorder_slice} appears under visitors "
            f"{[row['visitor_id'] for row in rows]} — a slice id is its open "
            f"millisecond plus a random 4-character draw, so two visitors "
            f"can legally coincide on one id, and a bare id cannot say "
            f"which is meant; address it as {addresses}"
        )
    return rows[0]


def _site_contexts(conn, slice_ids: list[int]) -> dict[str, str]:
    """Every site's business frame for the window, resolved by convention — data/context/<snippet>.md for each snippet the slice set spans, so nothing is threaded per call. The frame is what only the operator can state (what the site is for, what a good and a bad session mean); the evidence already testifies pages and flows. A window may span sites, but no site's slices are ever read under another site's frame: each snippet resolves its own file, and any one missing refuses the whole analysis."""
    placeholders = ",".join("?" * len(slice_ids))
    snippets = {
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT snippet FROM events WHERE slice_id IN ({placeholders})",
            slice_ids,
        )
    }
    if None in snippets or not snippets:
        raise SystemExit(
            "some of these slices carry no snippet id, so no site context "
            "can be resolved for them — the snippet rides in from the "
            "store's object key at load, so a row without one came from an "
            "object outside the store's chunk key layout; the site's frame "
            "is banked at data/context/<snippet>.md per the setup context step"
        )
    missing = sorted(
        s for s in snippets if not (_data_root() / "context" / f"{s}.md").exists()
    )
    if missing:
        raise SystemExit(
            f"no site context at data/context/<snippet>.md for {missing} — it "
            "carries what only the operator can state (what the site is for, "
            "what a good and a bad session mean); the setup skill's context "
            "step banks it there"
        )
    return {
        s: (_data_root() / "context" / f"{s}.md").read_text() for s in sorted(snippets)
    }


def _declared_dataset() -> str | None:
    """The AE dataset name for `--help` to name, or None when there is no deployment declaring one — help has to render from anywhere, including outside a clone."""
    try:
        return _ae_dataset()
    except (SystemExit, OSError):
        return None


def _slice_summary(conn, visitor_ids) -> dict:
    placeholders = ",".join("?" * len(visitor_ids))
    totals = {"replayable": 0, "discarded": 0, "rescued": 0}
    for row in conn.execute(
        f"SELECT status, COUNT(*) n FROM slices WHERE visitor_id IN ({placeholders}) GROUP BY status",
        list(visitor_ids),
    ):
        totals[row["status"]] = row["n"]
    return totals


def _prefix(args) -> str:
    """The key prefix a store verb was given, refused before any LIST when no chunk key can begin with it — the store matches a prefix literally, so an impossible one would otherwise read as an empty store."""
    from locus.evidence.chunk import prefix_skip_reason

    prefix = args.prefix or ""
    reason = prefix_skip_reason(prefix)
    if reason is not None:
        raise SystemExit(f"prefix {prefix!r}: {reason}")
    return prefix


def _ls(args) -> None:
    """A LIST over a real store is minutes of work that discovers slices the whole way, so the log takes them as they come — one JSON object per line, the scope first and the totals last, the rows between them in the order the store gave them up."""
    from locus.evidence.inventory import Inventory

    prefix = _prefix(args)
    local = None
    db = _resolve_db(required=False)
    if db:
        local = {}
        for row in connect(db).execute("SELECT visitor_id, recorder_slice FROM slices"):
            local.setdefault(row["visitor_id"], set()).add(row["recorder_slice"])

    listing = Inventory(_store(), local, prefix=prefix, cutoff=_cutoff())
    _result(
        {
            "current_timestamp": utc_stamp(),
            "scope": prefix,
            "loaded_known": local is not None,
        }
    )
    for row in listing:
        _result(row)
    t = listing.totals
    new = f" ({t['new']} new)" if local is not None else ""
    expired = (
        f", {t['expired']} past the retention horizon and never loadable"
        if t["expired"]
        else ""
    )
    dates = (
        f", {utc_stamp(t['first_ms'])} → {utc_stamp(t['last_ms'])}"
        if t["slices"]
        else ""
    )
    foreign = (
        f"; {t['foreign_keys']} foreign object(s) ({t['foreign_bytes'] / 1e6:.1f} MB) don't match the chunk layout"
        if t["foreign_keys"]
        else ""
    )
    _result(
        {
            "headline": (
                f"{t['visitors']} visitors, {t['slices']} slices{new}, {t['bytes'] / 1e6:.1f} MB{dates}{expired}{foreign}"
            ),
            "totals": t,
        }
    )


def _load_and_derive(db: str, prefix: str) -> None:
    """The load itself: its own accounting, whatever the store made it skip, the derivation's progress, the retention sweep, and last the outcome."""
    from locus.evidence.load import LoadStats, load_chunks

    conn = connect(db)
    loaded = load_chunks(conn, _store(), prefix, stats=LoadStats(), cutoff=_cutoff())
    skipped = loaded["skipped_chunks"]
    if loaded["stats"].counts["chunks"] or skipped:
        print(loaded["stats"].final())

    # A widening to a full re-derivation is said out loud, before the long pass runs.
    definition = _definition()
    stage = stale_stage(conn, definition)
    if stage is not None:
        print(
            f"the derivation changed since this db's surfaces were derived "
            f"— re-deriving every visitor from the {stage} stage on, so "
            f"nothing reads mixed-vintage"
        )
    done = finish_derivations(conn, db, definition)
    expired = _expire(conn, db)
    dirty, rescue = done["dirty"], done["rescue"]
    counts = loaded["stats"].counts
    unclean = loaded["stats"].unclean()
    unclean = f", {unclean}" if unclean else ""
    rederived = (
        f"; re-derived every visitor from the {done['from']} stage on because the derivation changed"
        if done["from"]
        else ""
    )
    scope = f" under {prefix!r}" if prefix else ""
    drained = counts["chunks"]
    totals = _slice_summary(conn, dirty) if dirty else None
    if dirty:
        headline = (
            f"loaded {loaded['inserted']} new events across {len(dirty)} "
            f"visitors{unclean}; their slices now: {totals['replayable']} replayable, "
            f"{totals['discarded']} discarded, {totals['rescued']} rescued{rederived}"
        )
    elif loaded["fetched_chunks"]:
        headline = f"loaded 0 new events; the fetched chunks{scope} were already current{unclean}{rederived}"
    elif drained:
        held = f", {skipped} already loaded" if skipped else ""
        headline = f"loaded 0 new events; {drained} chunks{scope} fetched{unclean}{held}{rederived}"
    elif skipped:
        headline = f"loaded 0 new events; all {skipped} chunks{scope} already loaded{rederived}"
    else:
        headline = f"loaded 0 new events; the store holds nothing{scope}{rederived}"
    if expired["slices"] or expired["chunks"]:
        cache = ", replay cache cleared" if expired["pages_cleared"] else ""
        headline += (
            f"; retention dropped {expired['slices']} slices and {expired['events']} "
            f"events across {expired['visitors']} visitors — recordings that opened "
            f"before {expired['cutoff']} are past the horizon{cache}"
        )
    _result(
        {
            "current_timestamp": utc_stamp(),
            "headline": headline,
            "inserted": loaded["inserted"],
            "visitors": len(dirty),
            "slices": totals,
            "rescue": rescue or None,
            "chunks": {
                "fetched": drained,
                "already_loaded": skipped,
                "corrupt": counts["corrupt"],
                "lossy": counts["lossy"],
            },
            "rederived_from": done["from"],
            "retention": expired,
        }
    )


def _load(args) -> None:
    """The chunks, then every derivation they left dirty — the two halves of making the store queryable, in the one verb that does both."""
    _load_and_derive(_resolve_db(create=True), _prefix(args))


def _doctor(args) -> None:
    """The heal narrates for as long as a corpus-wide re-derivation takes, and the report lands last — every check, each failing one carrying its fix text. The exit code is the one thing the caller is told without reading: nonzero means something in there is unhealthy."""
    from .doctor import run_doctor

    report = run_doctor(deployment_root(), _resolve_db(required=False), _definition())
    _result(report)
    if not report["clean"]:
        raise SystemExit(1)


def _age(ms: int) -> str:
    """A duration as the operator reads one — `3d 4h`, `5h 12m`, `9m` — for how far behind the store a site's loaded data stands."""
    minutes = max(0, ms) // 60_000
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _status(args) -> None:
    """Corpus accounting, and it must reconcile: analyzable + dropped + unmaterialized == events, every dropped event belonging to a discarded slice whose row names its reason. A dropped event with no named cause is exactly the silent loss this command exists to make impossible; the reasons themselves are a query over `slices`. Paid-analysis accounting rides along: the spend block reports the ledger's recorded dollars against the deployment's declared caps, period by period."""
    from .spend import spend_report

    conn = connect(_resolve_db())
    # Every figure here comes off an index or off `slices`, never off a walk of the
    # events table: at corpus scale one walk of its wide rows takes minutes. Visitors
    # are counted over slices; a visitor with events but no slice yet is inside the
    # unmaterialized count.
    total = conn.execute("SELECT COUNT(*) n FROM events").fetchone()["n"]
    undistilled = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE type_str IS NULL"
    ).fetchone()["n"]
    visitors = conn.execute(
        "SELECT COUNT(DISTINCT visitor_id) v FROM slices"
    ).fetchone()["v"]
    slices = conn.execute(
        "SELECT status, COUNT(*) n, SUM(n_events) ev FROM slices GROUP BY status"
    ).fetchall()

    # The sites, in the operator's words: a snippet id is the store's name for a site,
    # and the hosts its recordings actually opened on are the domains the operator knows
    # it by — one snippet can span several (a storefront and its www, a staging host).
    # Names only: how much each host carries is a query over `slices`.
    sites: dict[str, list] = {}
    for row in conn.execute(
        "SELECT DISTINCT snippet, url FROM slices WHERE url IS NOT NULL ORDER BY snippet"
    ):
        host = urlsplit(row["url"]).hostname
        if host and host not in sites.setdefault(row["snippet"], []):
            sites[row["snippet"]].append(host)
    # How far behind the store each site's loaded data stands: the newest upload instant
    # among the chunks the db holds for that snippet, so no network read.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    sites_block = []
    for snippet, found in sites.items():
        through = conn.execute(
            "SELECT MAX(uploaded_ms) m FROM loaded_chunks WHERE key LIKE ?",
            (f"{snippet}/%",),
        ).fetchone()["m"]
        sites_block.append(
            {
                "snippet": snippet,
                "hosts": sorted(found),
                "loaded_through": utc_stamp(through) if through else None,
                "behind": _age(now_ms - through) if through else None,
            }
        )

    analyzable = conn.execute(
        "SELECT COUNT(*) n FROM events e JOIN slices s ON e.slice_id = s.id WHERE s.status = 'replayable'"
    ).fetchone()["n"]
    # Events that landed but whose slices were never materialized — a load that died between
    # hydrating a visitor and materializing its slices. Named here because the discard reasons cannot
    # account for them (they belong to no slices row at all), and the difference would otherwise show
    # up only as coverage that won't add up.
    unmaterialized = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE slice_id IS NULL"
    ).fetchone()["n"]
    dropped = total - analyzable
    stale = derivations_stale(conn, _definition())
    dropped_fraction = round(dropped / total, 4) if total else 0.0
    payload = {
        "current_timestamp": utc_stamp(),
        "headline": (
            f"{total} events, {visitors} visitors, {analyzable} analyzable ({dropped_fraction * 100:.2f}% dropped)"
        ),
        "events": total,
        "visitors": visitors,
        "sites": sites_block,
        "sites_note": (
            "loaded_through is the newest upload this db holds for the site; nothing "
            "the store received after it is here — `locus load <snippet>/` brings it"
        ),
        "distilled_events": total - undistilled,
        "derivations": {
            "stale": stale,
            **(
                {
                    "note": "the derived surfaces — timestamps, slices, flat "
                    "columns, projections, sessions — were derived by older code "
                    "or an older session definition — `locus doctor` re-derives "
                    "them (any `locus load` also repairs this)"
                }
                if stale
                else {}
            ),
        },
        "slices": {
            row["status"]: {"count": row["n"], "events": row["ev"]} for row in slices
        },
        "coverage": {
            "analyzable_events": analyzable,
            "dropped_events": dropped,
            "dropped_fraction": dropped_fraction,
            "unmaterialized_events": unmaterialized,
            "unmaterialized_note": (
                "hydrated but slices never materialized — a load that did not finish; "
                "`locus doctor` finishes the derivation (so does any "
                "`locus load`)"
            ),
        },
        "spend": spend_report(),
    }
    _result(payload)


def _materialized_page(build) -> Path:
    """Run one page composition, narrating what it materialized — the printed paths are how the caller learns where the material lives — and turning a payload refusal into a clean exit."""
    try:
        written, page = build()
    except ValueError as refusal:
        raise SystemExit(str(refusal))
    for path in written:
        print(f"materialized {path}")
    print(f"composed {page}")
    return page


def _compose_set_page(names: list[str]) -> Path:
    """A slice set's default page: within each visitor's slices, one payload and one mount per time-disjoint lane — a sequential lane plays through on one player, concurrent page contexts (tabs) mount side by side — all on the shared clock."""
    from locus.evidence.replay import (
        COMPONENT_NAME,
        default_page,
        disjoint_lanes,
        materialize,
        set_name,
    )

    db = _resolve_db()
    conn = connect(db)
    try:
        rows = [_resolve(conn, s) for s in names]
    except SystemExit as miss:
        raise SystemExit(
            f"{miss} — open takes a URL, a composed page or analysis directory path, or a slice set from events.db"
        )
    rows.sort(key=lambda r: r["start_ts"])
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["visitor_id"], []).append(row)
    pages = _derived_root(db) / "pages"

    def build():
        # Each mount's topbar states its own visitor, so raw sets need no extra
        # label; labels are an analysis's own citation vocabulary.
        written, mounts = [pages / COMPONENT_NAME], []
        for group in groups.values():
            for lane in disjoint_lanes(group):
                m = materialize(conn, pages, lane)
                written.append(m["payload"])
                mounts.append((m["payload"], None))
        return written, default_page(pages, set_name(rows), mounts)

    return _materialized_page(build)


def _compose_analysis_page(analysis_dir: Path) -> Path:
    """An analysis directory's default page, from window.json's slice table: consecutive same-visitor slices that don't overlap in time share one mount and play through as one stream, labeled by their label range (S1–S3); a visitor change or a time overlap (a concurrent lane) starts the next mount."""
    from locus.evidence.replay import (
        COMPONENT_NAME,
        analysis_slices,
        default_page,
        materialize,
    )

    db = _resolve_db()
    conn = connect(db)
    pages = _derived_root(db) / "pages"

    def build():
        table = analysis_slices(analysis_dir)
        runs: list[list[dict]] = []
        for row in table:
            run = runs[-1] if runs else None
            if (
                run
                and run[-1]["visitor"] == row["visitor"]
                and row["start_ts"] >= max(s["end_ts"] for s in run)
            ):
                run.append(row)
            else:
                runs.append([row])
        written, mounts = [pages / COMPONENT_NAME], []
        for run in runs:
            try:
                rows = [_resolve(conn, s["slice"], s["visitor"]) for s in run]
            except SystemExit as gone:
                # The analysis is a finding and is kept; the recording it cites is not, and
                # the horizon takes it on its own schedule. So a window naming a slice the db
                # no longer holds is the expected end of a citation, not a corruption.
                raise SystemExit(
                    f"{gone} — this analysis cites a recording the deployment no "
                    f"longer holds. Findings outlive the recordings behind them: "
                    f"the retention horizon (store/deploy.config.toml) expired it "
                    f"from the bucket and the db alike, so it cannot be replayed. "
                    f"The analysis's own text, citations, and screenshots stand"
                )
            m = materialize(conn, pages, rows)
            written.append(m["payload"])
            label = (
                run[0]["label"]
                if len(run) == 1
                else f"{run[0]['label']}–{run[-1]['label']}"
            )
            mounts.append((m["payload"], label))
        return written, default_page(pages, analysis_dir.name, mounts)

    return _materialized_page(build)


def _resolve_open(argv: list[str]) -> list[str]:
    """`browse open`'s target may be a URL or a composed page, passed through to the browser as-is — or a slice set or an analysis directory, which first materializes the replay material and composes the default page in pages/ beside the db, printing what it wrote; the browser then opens the one page. A moment rides any of those as a `#t=` fragment and lands on the page that gets opened, so a slice and the moment to see it at are one target."""
    from urllib.parse import urlparse

    rest = list(argv[1:])
    window = []
    if rest and rest[0].startswith("w") and rest[0][1:].isdigit():
        window = [rest.pop(0)]  # the window to open into, the browser's to interpret
    if not rest:
        return argv  # the browser says what open needs
    flags = [t for t in rest if t.startswith("-")]
    if flags:
        raise SystemExit(
            f"open takes no flags ({' '.join(flags)}): `open [wN] <target>` — "
            "the window, when one is named, comes first"
        )
    # One page is opened, so it is addressed at one moment: the fragment is lifted off
    # whichever name carries it and re-attached to the page composed from them all.
    moments = {t.partition("#")[2] for t in rest} - {""}
    if len(moments) > 1:
        raise SystemExit(
            "one page opens at one moment — these disagree: "
            + ", ".join(f"#{m}" for m in sorted(moments))
        )
    fragment = f"#{moments.pop()}" if moments else ""
    rest = [t.partition("#")[0] for t in rest]
    if len(rest) == 1:
        target = rest[0]
        if urlparse(target).scheme in ("http", "https", "file", "data", "about"):
            return ["open", *window, target + fragment]
        path = Path(target)
        if path.is_file():
            return ["open", *window, target + fragment]
        if path.is_dir():
            return ["open", *window, str(_compose_analysis_page(path)) + fragment]
    # No slice id holds a space, so a set quoted into one shell word is still that set.
    rest = [piece for token in rest for piece in token.split()]
    return ["open", *window, str(_compose_set_page(rest)) + fragment]


def _browse(args) -> None:
    """The anchor every prose surface owes: a browse reply gets cited back at a moment, and whoever reads it later has to be able to date it."""
    from locus.evidence.browse import is_help, run

    argv = list(args.args)
    # Help documents the browser rather than driving it, so it resolves no deployment root and
    # answers from anywhere; past this branch an invocation has a verb and a window to reach.
    if is_help(argv):
        raise SystemExit(run(None, argv))
    print(f"current timestamp: {utc_stamp()}")
    if argv[0] == "open":
        argv = _resolve_open(argv)
    raise SystemExit(run(_data_root() / "browse", argv))


def _inline_or_file(value: str | None) -> str | None:
    """A value that may be the text itself or the path to a file holding it. The filesystem answers the question, and a value it cannot even ask about — too long for a filename, holding a newline — is text by that very fact."""
    if not value:
        return value
    try:
        path = Path(value)
        return path.read_text() if path.is_file() else value
    except OSError:
        return value


def _resolve_address(conn, name: str) -> tuple[sqlite3.Row, int | None, int | None]:
    """A slice address → (row, window_start, window_end). A bare id is the whole slice; `<slice>#<k>` is its k-th route-boundary piece, the structural unit a price report offers when the slice alone out-prices the model's context."""
    base, sep, k = name.partition("#")
    row = _resolve(conn, base)
    if not sep:
        return row, None, None
    from .window import slice_pieces

    bounds = slice_pieces(conn, row["id"])
    if len(bounds) < 2:
        raise SystemExit(
            f"slice {base} has no internal route boundaries — there are no "
            f"pieces to address; widen --screenshot-interval instead"
        )
    if not k.isdigit() or not 1 <= int(k) <= len(bounds):
        raise SystemExit(
            f"slice {base} has pieces #1..#{len(bounds)}; {name!r} names "
            f"none of them — price the slice to see them listed"
        )
    _, ws, we = bounds[int(k) - 1]
    return row, ws, we


def _analyze(args) -> None:
    from .engine import (
        PayloadOverflow,
        home_analysis,
        price_analysis,
        run_analysis,
    )

    db = _resolve_db()
    conn = connect(db)
    if conn.execute(
        "SELECT 1 FROM slices WHERE recorder_slice = ?",
        (args.question.rpartition("/")[2],),
    ).fetchone():
        raise SystemExit(
            f"{args.question!r} is a slice id — the last positional argument "
            f"is the question; write the question itself (or the path to a "
            f"file holding it)"
        )
    question = _inline_or_file(args.question)
    addressed = [_resolve_address(conn, s) for s in args.slices]
    window_start = window_end = None
    if any(ws is not None or we is not None for _, ws, we in addressed):
        if len(addressed) > 1:
            raise SystemExit(
                "a piece address is an analysis of its own — name it alone"
            )
        _, window_start, window_end = addressed[0]
    rows = [row for row, _, _ in addressed]
    slice_ids = [row["id"] for row in rows]
    from .window import resolve_window

    window = resolve_window(conn, slice_ids, window_start, window_end)
    if args.model:
        from .model.cards import declared_card

        try:
            declared_card(args.model)
        except ValueError as undeclared:
            raise SystemExit(str(undeclared))
        model = args.model
    else:
        from .model.cards import default_card

        model = default_card()

    common = {
        "question": question,
        "model": model,
        "site_contexts": _site_contexts(conn, slice_ids),
        "screenshot_interval_ms": args.screenshot_interval,
    }

    def report_price(price: dict) -> None:
        """The fit-and-cost headline leads the result; the whittling surfaces (per-slice, pieces) follow it. A price that fits names the spending step."""
        turn1 = price["turn1"]
        cost = (
            f", turn-1 input ${price['pricing']['turn1_input_usd']}"
            if "pricing" in price
            else ""
        )
        server = _local_server(
            model, turn1["total_tokens"], _derived_root(db) / "analyses"
        )
        _result(
            {
                "current_timestamp": utc_stamp(),
                "headline": (
                    f"{price['model']}: "
                    f"{'fits' if turn1['fits'] else 'does not fit'} — "
                    f"{turn1['total_tokens']} tokens ({turn1['text_tokens']} text "
                    f"+ {turn1['screenshot_tokens']} screenshots), {turn1['context_pct']}% "
                    f"of context{cost}"
                    + (server["line"] if server else "")
                    + ("; add --run to send it" if turn1["fits"] else "")
                ),
                **price,
                **({"local_server": server} if server else {}),
            }
        )

    if not args.run:
        report_price(price_analysis(conn, window, **common))
        return
    from .model.cards import declared_pricing
    from .spend import SpendWall, check_walls

    # The wall is a ledger read, asked before the analysis has a home: a capped
    # deployment refuses in milliseconds and leaves nothing in analyses/. The
    # price reads the corpus and takes seconds on a large window, so the home
    # is named and announced first and a price refusal says why in the log the
    # caller was already pointed at.
    if declared_pricing(model):
        try:
            check_walls()
        except SpendWall as refusal:
            raise SystemExit(str(refusal))
    out_dir = home_analysis(_analysis_dir(db, rows))
    started(out_dir)
    with speaking_into(out_dir / "analysis.log"):
        _last_word_on_termination(out_dir)
        try:
            result = run_analysis(conn, window, out_dir=out_dir, **common)
        except PayloadOverflow as refused:
            report_price(refused.price)
            raise SystemExit(str(refused))
        except SpendWall as refusal:
            raise SystemExit(str(refusal))
        _result(
            {
                "current_timestamp": utc_stamp(),
                "headline": f"answered — {Path(result['response_path']).name}",
                **result,
            }
        )


def _last_word_on_termination(out_dir: Path) -> None:
    """A run told to stop says so in its own log before it goes: a log that ends on a heartbeat reads as a death with no cause, and SIGTERM is the one stop a process can still speak through (a SIGKILL leaves nothing to catch). Who sent it is unknowable, so the line says only what is: the directory keeps the transcripts written so far, and an answer stands only if an `answered` line precedes the stop. Raising SystemExit lets the run's finally blocks close what it holds."""
    import signal

    from locus.evidence.speak import say

    def stopped(signum, frame):
        say(
            f"stopped by {signal.Signals(signum).name}; {out_dir} keeps the transcripts "
            "written so far; no validated answer unless an `answered` line precedes "
            "this one — an answer takes a new run"
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stopped)


def _ae(args) -> None:
    from .ae import run_ae

    _result(run_ae(args.sql))


def _usage(args) -> None:
    from locus.evidence.usage import usage_query

    # The anchor and the query wrap Cloudflare's response rather than injecting
    # into it: `response` is the API's shape untouched, and the envelope keeps
    # the facts attached wherever the file travels.
    _result(
        {
            "current_timestamp": utc_stamp(),
            "query": args.query,
            "response": usage_query(args.query),
        }
    )


def main(argv=None) -> None:
    export_declared()
    parser = argparse.ArgumentParser(
        prog="locus",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def declare(name, help, **kw):
        """A subcommand declared once: the same sentence is its line under `locus --help` and the body of its own `--help`, so a command carrying no arguments still says what it is when asked directly."""
        return sub.add_parser(name, help=help, description=help, **kw)

    p = declare(
        "ls",
        help="the store's slices, one row each, marked loaded or not against events.db and marked when past the retention horizon",
    )
    p.add_argument(
        "prefix",
        nargs="?",
        help="key prefix scoping the listing — any leading "
        "substring of the key (e.g. snippetid/ or "
        "snippetid/2026-06-25/); empty lists the whole store, "
        "and anything finer than a prefix is a filter over "
        "the returned rows",
    )
    p.set_defaults(fn=_ls)

    p = declare(
        "load",
        help="load chunks from the store and leave events.db queryable — hydrate, materialize slices, distill, then drop what the retention horizon has expired; incremental",
    )
    p.add_argument(
        "prefix",
        nargs="?",
        help="key prefix to load (e.g. snippetid/, snippetid/date/visitor/); empty loads the whole store",
    )
    p.set_defaults(fn=_load)

    p = declare(
        "doctor",
        help="check and heal the deployment's mechanical integrity — "
        "derivation currency, runtimes, the retention horizon over "
        "the db, store drift, card price drift; non-zero exit while "
        "anything is unhealthy",
    )
    p.set_defaults(fn=_doctor)

    p = declare(
        "status",
        help="corpus accounting — events, visitors, each site's "
        "hosts, slices by status, a coverage block that reconciles "
        "every recorded event to a fate, whether derivations are "
        "stale, and paid-analysis spend against the deployment's "
        "declared caps",
    )
    p.set_defaults(fn=_status)

    # The tail is the browser's own grammar, flags and all, so nothing in it may look
    # like an option here: a prefix character no one can type leaves every token —
    # `--css`, `--full`, `--help` — to the browser that owns them.
    p = declare(
        "browse",
        prefix_chars="\0",
        add_help=False,
        help="the deployment's browser — headed windows, one per instance "
        "(w1, w2, …), state surviving between invocations; `open` also takes "
        "a slice set or an analysis directory, materializing its replay page "
        "into data/pages/ first",
    )
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=_browse)

    p = declare(
        "analyze",
        help="put a question to the model over recorded evidence — "
        "an invocation prices the composed model-payload exactly, "
        "spending nothing, and a price that fits names the "
        "second, deliberate step that sends it",
    )
    p.add_argument(
        "slices",
        nargs="+",
        help="whole slices — several visitors and time-overlapping "
        "recordings legal, composed as labeled slices on one "
        "shared window clock — or exactly one route-boundary "
        "piece address (<slice>#2) from a price report",
    )
    p.add_argument(
        "question",
        help="the entire question, in your words — inline text, or a path to a file holding it",
    )
    p.add_argument(
        "--model",
        help="the model this invocation prices (and, if sent, runs) on, by the "
        "name of its card — the declaration one file per model in "
        "config/cards/ under the deployment root (default: the card marked "
        "`default = true`)",
    )
    # Deliberately absent from --help: the spend step is learned from a price
    # that fits, never from the flag list — a first-contact agent cannot
    # compose a spending invocation before it has held the price.
    p.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--screenshot-interval",
        type=int,
        default=SCREENSHOT_INTERVAL_MS,
        help="ms between activity-screenshots — the screenshot-cost "
        "lever (default: %(default)s); per-image wire cost is the "
        "model card's",
    )
    p.set_defaults(fn=_analyze)

    dataset = _declared_dataset()
    p = declare(
        "ae",
        help="run SQL against the deployment's Analytics Engine"
        + (f" (dataset: {dataset})" if dataset else " (this deployment declares none)")
        + f"; the row layout lives in {TELEMETRY_SCHEMA}, "
        "field semantics in store/src/telemetry.js",
    )
    p.add_argument(
        "sql",
        help='the AE SQL, e.g. "SELECT blob1 AS metric, '
        "sum(_sample_interval) n FROM "
        f'{dataset or "<dataset>"} GROUP BY metric"; '
        "ms('<UTC date or datetime>') anywhere in it "
        "expands to epoch ms before the query is sent — for the epoch-ms "
        "double columns; the server-clock `timestamp` column is a DateTime "
        "and compares only against toDateTime('<UTC datetime>')",
    )
    p.set_defaults(fn=_ae)

    p = declare(
        "usage",
        help="run GraphQL against Cloudflare's usage analytics — "
        "worker requests, R2 storage/operations; $account "
        "binds in the query",
    )
    p.add_argument(
        "query",
        help="the GraphQL, e.g. 'query($account: String!) { viewer "
        "{ accounts(filter: {accountTag: $account}) { "
        "r2OperationsAdaptiveGroups(limit: 10, filter: "
        '{date_geq: "2026-07-01", date_leq: "2026-07-31"}) '
        "{ sum { requests } dimensions { actionType } } } } }'",
    )
    p.set_defaults(fn=_usage)

    args = parser.parse_args(argv)
    if not _main_opens(args):
        args.fn(args)
        return
    with _speaking(args.command):
        args.fn(args)


if __name__ == "__main__":
    main()
