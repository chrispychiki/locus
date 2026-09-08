"""The deployment — the one anchor everything per-deployment hangs off, and its declared facts.

The root is the directory holding store/wrangler.toml, found from cwd upward because the agent always runs inside the clone.
Two directories beneath it are named once here.
CONFIG_DIR holds the deployment-wide operator declarations — the spend walls, the session definition, the card roster; package-owned declarations (store/wrangler.toml, deploy.config.toml, the facade) stay with the machinery that reconciles them.
DATA_DIR holds everything the deployment accretes: the single events.db and the families derived beside it, the run logs, the spend ledger, the deployment's browser, and the operator's context and memory files.
Consumers append their file and family names to `deployment_root() / CONFIG_DIR` and `deployment_root() / DATA_DIR`; nothing creates a directory eagerly — each writer makes the parents it needs.

Every fact the deployment declares resolves through here, from the artifact that owns it: credentials from the `.env` files on disk, the bucket and worker name from store/wrangler.toml, the AE dataset from its binding there, the worker's origin from the worker name on the account's own workers.dev subdomain. Consumers call these instead of restating a value or requiring a hand-set variable, so a credentialed affordance works the same from the CLI, a library caller, or a test.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import tomllib

from .cloudflare import CF_API

DATA_DIR = "data"
CONFIG_DIR = "config"


def machine_cache_dir() -> Path:
    """The machine-level locus cache — DATA_DIR's counterpart for facts that belong to the machine or a credential rather than to any one deployment (the Gemini upload cache, the R2 token-id lookup, the price-registry memo, the mlx server's one-resident-model pidfile). XDG-resolved: $XDG_CACHE_HOME when set, else ~/.cache, `locus/` beneath; consumers append their file names and make the parents they need."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "locus"


def deployment_root() -> Path:
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / "store" / "wrangler.toml").exists():
            return directory
    raise SystemExit(
        f"not inside a Locus deployment: no store/wrangler.toml found from "
        f"{Path.cwd()} upward — run this from inside the clone; the setup "
        f"skill has the steps"
    )


def parse_env_file(path: Path) -> dict[str, str]:
    """One .env file's declarations: KEY=VALUE lines, blanks and # comments skipped, surrounding quotes stripped, the first value seen for a name winning — the one reading every consumer of these files shares, so a quoted token means the same thing to all of them."""
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        found.setdefault(key.strip(), value)
    return found


def env_from_files() -> dict[str, str]:
    """The credentials the clone declares on disk, read but not exported. Every level from cwd upward is read — `.env.local`, then `.env`, then the store package's `store/.env`, nearest level first — and the first value seen for a name wins. The deploy keeps the Cloudflare token in `store/.env` (wrangler auto-reads a `.env` in its own directory) and Locus reads that same file for R2 access."""
    found: dict[str, str] = {}
    for directory in (Path.cwd(), *Path.cwd().parents):
        for env_file in (
            directory / ".env.local",
            directory / ".env",
            directory / "store" / ".env",
        ):
            if not env_file.exists():
                continue
            for key, value in parse_env_file(env_file).items():
                found.setdefault(key, value)
    return found


def export_declared() -> None:
    """Export what the clone declares, so an operator-given token reaches the tools with no manual export. An already-set environment variable wins over every file."""
    for key, value in env_from_files().items():
        os.environ.setdefault(key, value)


def _wrangler() -> dict:
    return tomllib.loads((deployment_root() / "store" / "wrangler.toml").read_text())


def bucket() -> str:
    """The deployment's R2 bucket, named once in store/wrangler.toml — the single declaration `provision` reads, so Locus and the deploy never drift off the same source."""
    buckets = _wrangler().get("r2_buckets") or []
    if buckets and buckets[0].get("bucket_name"):
        return buckets[0]["bucket_name"]
    raise SystemExit(
        f"{deployment_root() / 'store' / 'wrangler.toml'} declares no r2_buckets[0].bucket_name"
    )


def store_url() -> str:
    """The store Locus reads is the deployment's own R2 bucket: named by store/wrangler.toml, with endpoint, account, and S3 keys self-served from store/.env."""
    return f"s3://{bucket()}"


def store():
    """The deployment's store, ready to read: the declared credentials exported, the bucket resolved, the S3 client built — the one call behind every consumer that reads chunks."""
    from .store import store_for

    export_declared()
    return store_for(store_url())


def ae_dataset() -> str | None:
    """The Analytics Engine dataset name, read from its one declaration — the CAPTURE binding in store/wrangler.toml. None when the deployment runs without AE."""
    datasets = _wrangler().get("analytics_engine_datasets") or []
    if not datasets:
        return None
    if "dataset" not in datasets[0]:
        raise SystemExit(
            f"{deployment_root() / 'store' / 'wrangler.toml'} declares "
            f"analytics_engine_datasets[0] without a dataset name"
        )
    return datasets[0]["dataset"]


DEFINITIONS_FILE = "definitions.toml"

_DEFINITION_SHAPE = {
    "session": {"inactivity_minutes": (int, float)},
    "engaged": {"min_seconds": (int, float), "min_pageviews": (int,)},
}


def definitions(root: Path | None = None) -> dict:
    """The operator's definition of a session and of engaged, as declared in config/definitions.toml under the deployment root (`root`, or the one found from cwd), validated: the sessions table is derived by exactly these values (sessions.py), so a declaration that silently failed to shape it is the one failure this read refuses — an unknown table or key, a missing key, a value that is not a non-negative number, each named with the file."""
    path = (root or deployment_root()) / CONFIG_DIR / DEFINITIONS_FILE
    if not path.exists():
        raise SystemExit(
            f"no {path}: the operator's session definition ships with the repo "
            f"— restore the file"
        )
    declared = tomllib.loads(path.read_text())
    unknown = set(declared) - set(_DEFINITION_SHAPE)
    if unknown:
        raise SystemExit(f"{path} declares unknown tables {sorted(unknown)}")
    for table, keys in _DEFINITION_SHAPE.items():
        block = declared.get(table)
        if not isinstance(block, dict):
            raise SystemExit(f"{path} declares no [{table}] table")
        stray = set(block) - set(keys)
        if stray:
            raise SystemExit(f"{path}: [{table}] declares unknown keys {sorted(stray)}")
        for key, kinds in keys.items():
            value = block.get(key)
            if not isinstance(value, kinds) or isinstance(value, bool) or value < 0:
                raise SystemExit(
                    f"{path}: [{table}].{key} must be a non-negative number, not {value!r}"
                )
    return {
        "session": dict(declared["session"]),
        "engaged": dict(declared["engaged"]),
        "path": str(path),
    }


def worker_name() -> str:
    """The worker's name, from its one declaration in store/wrangler.toml."""
    name = _wrangler().get("name")
    if not name:
        raise SystemExit(
            f"{deployment_root() / 'store' / 'wrangler.toml'} declares no worker name"
        )
    return name


def worker_url() -> str:
    """The deployed worker's origin — the script origin and upload endpoint the deploy prints. Derived, never stored: the worker name from store/wrangler.toml on the account's workers.dev subdomain, asked of the Cloudflare API with the deployment's own credentials (the subdomain is account state, declared nowhere in the repo)."""
    export_declared()
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account or not token:
        raise SystemExit(
            "resolving the worker URL needs CLOUDFLARE_ACCOUNT_ID and "
            "CLOUDFLARE_API_TOKEN (store/.env) — the workers.dev subdomain is "
            "account state, readable only with the deployment's credentials"
        )
    request = urllib.request.Request(
        f"{CF_API}/accounts/{account}/workers/subdomain",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"the Cloudflare API refused the workers.dev subdomain read "
            f"({error.code}): {error.read().decode(errors='replace')[:500]} — "
            f"the token in store/.env needs a Workers scope to read it"
        )
    except urllib.error.URLError as error:
        raise SystemExit(
            f"could not reach the Cloudflare API to resolve the workers.dev subdomain: {error.reason}"
        )
    if not body.get("success") or not (body.get("result") or {}).get("subdomain"):
        raise SystemExit(
            f"the Cloudflare API returned no workers.dev subdomain for the account: {body.get('errors') or body}"
        )
    return f"https://{worker_name()}.{body['result']['subdomain']}.workers.dev"
