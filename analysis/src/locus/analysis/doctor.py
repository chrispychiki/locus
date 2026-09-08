"""Doctor — the deployment's mechanical integrity, diagnosed and healed in one run.

Doctor checks the judgment-free invariants — facts with exactly one correct answer — and only those:

- **Derivation currency.** The db records the identity of what derived its surfaces — timestamps, slices, flat columns, projections, sessions — the deriving code and, for sessions, the operator's declared numbers; a recorded vintage differing from the installed identity means a change left every derived value behind, and rows hydration landed but materialization, distillation, or the session derivation never reached (a load that died partway) are stranded. Both heal mechanically, and doctor heals them itself, through the same derivation path a load drives — timestamps, slice, rescue, re-distill from raw_json, sessions.
- **Runtimes.** bun and Playwright's Chromium are runtimes the CLI shells into. Installing software is not doctor's to do; a missing runtime is named with its install.
- **Retention.** The deployment declares one horizon and it governs every copy of a recording, the local db included — a db nothing ever swept would keep forever what a load pulled, while the bucket's copy expired. So doctor applies it here as `locus load` does, dropping the recordings past it and saying what went; a horizon the declaration does not state as a whole number of days is unhealthy, because nothing can be swept by it.
- **Store drift.** The repo's declaration is the source of truth for the live deployment and a deploy reconciles the live state back to it, so live state that differs is drift that will silently revert. What "matches" means is owned by the store package's own doctor (store/scripts/doctor.js, beside the provision that converges it); this check runs that script against the live Cloudflare API and carries its verdict, each drifted setting named with the command that fixes it.
- **Profile size.** The operator profile (`data/operator.md`) is read before every piece of work, so it has a hard byte cap (PROFILE_CAP_BYTES, declared here). Shrinking it takes judgment — consolidation — so doctor names the consolidate-notes skill and never touches the file.
- **Price drift.** A Gemini card's declared per-token prices are what every billed call's ledger entry and every spend wall computes dollars from, and the provider can reprice under them. The check is spend.verify_prices — the same best-effort registry comparison every price report runs — swept here across every Gemini card: a mismatch is drift naming both numbers and the card to edit; an absent registry key or unreachable registry is honestly unverified, never unhealthy. A Gemini card declaring no pricing block at all is unhealthy outright — every paid call on it refuses at construction.

Everything that takes judgment — what capture is costing, whether PII is leaking, whether the deployment is healthy in any semantic sense — belongs to the agent composing over the observability primitives (`ae`, `usage`, the store listings), never to a verb.

Safe to run anytime: reads, plus the idempotent derivation heal and the retention sweep — which deletes, and is meant to, on the deployment's own declared horizon. The report is JSON; a clean deployment reports every check ok, and the CLI exits non-zero while anything stays unhealthy.
"""

import os
import subprocess
from pathlib import Path

from locus.evidence.db import connect
from locus.evidence.deployment import DATA_DIR
from locus.evidence.derive import (
    derivations_stale,
    finish_derivations,
    stranded_events,
)

PROFILE_CAP_BYTES = 4096


def _profile_check(deployment_root: Path) -> dict:
    profile = deployment_root / DATA_DIR / "operator.md"
    if not profile.exists():
        return {
            "check": "profile_size",
            "ok": True,
            "note": "no operator.md yet — nothing to check",
        }
    size = profile.stat().st_size
    if size <= PROFILE_CAP_BYTES:
        return {
            "check": "profile_size",
            "ok": True,
            "bytes": size,
            "cap": PROFILE_CAP_BYTES,
        }
    return {
        "check": "profile_size",
        "ok": False,
        "bytes": size,
        "cap": PROFILE_CAP_BYTES,
        "finding": f"the operator profile is {size} bytes; the cap is "
        f"{PROFILE_CAP_BYTES}. It is read before every piece of work, so it "
        f"has to stay small",
        "fix": "consolidate it (the consolidate-notes skill): merge, "
        "generalize, or drop claims until it fits. Doctor never edits it",
    }


def _bun_check() -> dict:
    from locus.evidence.slices import require_bun

    try:
        require_bun()
        return {"check": "bun", "ok": True}
    except SystemExit as missing:
        return {"check": "bun", "ok": False, "fix": str(missing)}


def _browser_check() -> dict:
    """Asks playwright itself for the Chromium path — but in a child process: the throwaway driver session tears down with a harmless-but-noisy asyncio race on this interpreter (a pending connection task destroyed at exit), and the child keeps that noise off doctor's own stderr. A failed probe surfaces the child's stderr whole."""
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from playwright.sync_api import sync_playwright\n"
                "with sync_playwright() as p:\n"
                "    print(p.chromium.executable_path)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    executable = Path(probe.stdout.strip()) if probe.stdout.strip() else None
    if probe.returncode == 0 and executable and executable.exists():
        return {"check": "browser", "ok": True}
    finding = (
        f"Playwright's Chromium is not installed (nothing at {executable})"
        if executable
        else f"the Playwright probe failed: {probe.stderr.strip()}"
    )
    return {
        "check": "browser",
        "ok": False,
        "finding": finding,
        "fix": "./install.sh from the deployment root installs it",
    }


def _derivations_check(db: str | None, bun_ok: bool, definition: dict) -> dict:
    if db is None:
        return {
            "check": "derivations",
            "ok": True,
            "note": "no events.db yet — nothing derived to check",
        }
    conn = connect(db)
    found = {
        "stale_vintage": derivations_stale(conn, definition),
        "stranded_events": stranded_events(conn),
    }
    if not (found["stale_vintage"] or found["stranded_events"]):
        conn.close()
        return {"check": "derivations", "ok": True, **found}
    if not bun_ok:
        conn.close()
        return {
            "check": "derivations",
            "ok": False,
            "found": found,
            "fix": "the heal is the distillation pass, a bun script — "
            "install bun (the bun check names it) and rerun doctor",
        }
    done = finish_derivations(conn, db, definition)
    healed = {
        "re_derived": (
            f"every visitor, from the {done['from']} stage on — "
            + {
                "timestamps": "timestamps, slices, rescue, distillation, sessions",
                "slices": "slices, rescue, distillation, sessions",
                "distill": "distillation, sessions",
                "sessions": "sessions",
            }[done["from"]]
            if done["stale"]
            else f"slices re-materialized, events distilled, and sessions derived for {len(done['dirty'])} visitor(s)"
        ),
        **({"rescue": done["rescue"]} if any(done["rescue"].values()) else {}),
    }
    still = {
        "stale_vintage": derivations_stale(conn, definition),
        "stranded_events": stranded_events(conn),
    }
    conn.close()
    if still["stale_vintage"] or still["stranded_events"]:
        return {
            "check": "derivations",
            "ok": False,
            "found": found,
            "healed": healed,
            "still_unhealthy_after_heal": still,
        }
    return {"check": "derivations", "ok": True, "found": found, "healed": healed}


def _retention_check(deployment_root: Path, db: str | None, definition: dict) -> dict:
    """The horizon over the local copy. The declaration governs every copy of a recording, and the db is the one that would otherwise keep forever what a load pulled, so doctor sweeps it here — the same drop `locus load` makes, so a deployment that is only ever read still honours the horizon. What went is reported, never silent: these are recordings, and the line is the only record they were here."""
    from locus.evidence.db import connect
    from locus.evidence.derive import derivation_lock
    from locus.evidence.retention import expire, horizon_cutoff, horizon_days

    try:
        days = horizon_days(deployment_root)
        cutoff = horizon_cutoff(deployment_root)
    except SystemExit as refusal:
        return {"check": "retention", "ok": False, "finding": str(refusal)}
    if db is None:
        return {
            "check": "retention",
            "ok": True,
            "horizon_days": days,
            "note": "no events.db yet — nothing loaded to expire",
        }
    conn = connect(db)
    try:
        with derivation_lock(db):
            dropped = expire(
                conn, cutoff, definition, Path(db).resolve().parent / "pages"
            )
    finally:
        conn.close()
    return {"check": "retention", "ok": True, "horizon_days": days, "expired": dropped}


def _store_check(deployment_root: Path, bun_ok: bool) -> dict:
    store = deployment_root / "store"
    if not (store / "scripts" / "doctor.js").exists():
        return {
            "check": "store",
            "ok": False,
            "finding": f"no doctor.js at {store / 'scripts'} — the store "
            f"package owns the declared-vs-live comparison; "
            f"restore the clone",
        }
    if not bun_ok:
        return {
            "check": "store",
            "ok": False,
            "finding": "cannot ask — the declared-vs-live comparison is "
            "the store's own doctor, a bun script; install bun "
            "(the bun check names it) and rerun doctor",
        }
    if not (
        os.environ.get("CLOUDFLARE_API_TOKEN")
        and os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    ):
        return {
            "check": "store",
            "ok": False,
            "finding": "cannot ask — no Cloudflare credentials",
            "fix": "put CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in store/.env",
        }
    proc = subprocess.run(
        ["bun", "scripts/doctor.js"],
        cwd=store,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = (proc.stdout + proc.stderr).strip().splitlines()
    if proc.returncode == 0:
        return {"check": "store", "ok": True, "detail": detail}
    if proc.returncode == 1:
        return {
            "check": "store",
            "ok": False,
            "finding": "the live deployment does not match the repo's "
            "declaration — each DRIFT line names the setting "
            "and the command that fixes it",
            "detail": detail,
        }
    return {
        "check": "store",
        "ok": False,
        "finding": "the live state could not be read",
        "detail": detail,
    }


def _pricing_check() -> dict:
    from .model.cards import GEMINI, PRICING_FIELDS, declared_pricing, models_speaking
    from .spend import verify_prices

    models = models_speaking(GEMINI)
    if not models:
        return {
            "check": "model_prices",
            "ok": True,
            "note": "no Gemini cards declared — nothing bills, nothing to verify",
        }
    cards = []
    ok = True
    for model in models:
        prices = declared_pricing(model)
        if prices is None:
            ok = False
            cards.append(
                {
                    "model": model,
                    "status": "undeclared",
                    "finding": "no pricing block — every paid call on this card refuses at construction",
                    "fix": f"declare {list(PRICING_FIELDS)} (USD per million "
                    f"tokens, from the model's pricing page) under "
                    f"[pricing] in config/cards/{model}.toml",
                }
            )
            continue
        verdict = verify_prices(model, prices)
        if verdict["status"] == "drift":
            ok = False
            cards.append(
                {
                    "model": model,
                    "status": "drift",
                    "mismatches": verdict["mismatches"],
                    "fix": f"the declaration is the price of record — check the "
                    f"model's pricing page and edit the card's pricing "
                    f"block in config/cards/{model}.toml to match it",
                }
            )
        else:
            cards.append({"model": model, **verdict})
    return {"check": "model_prices", "ok": ok, "cards": cards}


def run_doctor(deployment_root: Path, db: str | None, definition: dict) -> dict:
    from locus.evidence.clock import utc_stamp

    bun = _bun_check()
    checks = [
        bun,
        _browser_check(),
        _derivations_check(db, bun["ok"], definition),
        _retention_check(deployment_root, db, definition),
        _store_check(deployment_root, bun["ok"]),
        _profile_check(deployment_root),
        _pricing_check(),
    ]
    return {
        "current_timestamp": utc_stamp(),
        "clean": all(c["ok"] for c in checks),
        "checks": checks,
    }
