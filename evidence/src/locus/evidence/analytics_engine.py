"""Read the deployment's Analytics Engine — the worker's telemetry rates plane.

The query is yours: which blob/double is which lives in `store/src/telemetry-schema.json`, field semantics in `store/src/telemetry.js` — this owns only the connection. Account and token resolve exactly as the R2 reads resolve them (cloudflare.py). Analytics Engine has no first-party CLI, so this POSTs SQL to the REST endpoint.

Whether a deployment has AE at all is the `CAPTURE` binding in `store/wrangler.toml`; with it absent the worker writes nothing and there is no rates plane: a query here fails loud — Cloudflare's SQL API errors on a dataset that never wrote — or returns empty only once an old dataset drains. Read against R2's exact plane instead.
"""

import json
import urllib.error
import urllib.request

from .cloudflare import CF_API, account_and_token
from .deployment import deployment_root

# The row layout's one declaration — the file the worker writes rows by; deployment-root relative.
TELEMETRY_SCHEMA = "store/src/telemetry-schema.json"


def telemetry_schema(root=None) -> dict:
    """The declaration itself, read from the deployment root (or the root a caller resolved)."""
    return json.loads(((root or deployment_root()) / TELEMETRY_SCHEMA).read_text())


def ae_query(sql: str) -> dict:
    account_id, token = account_and_token("Analytics Engine")
    request = urllib.request.Request(
        f"{CF_API}/accounts/{account_id}/analytics_engine/sql",
        data=sql.encode(),
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        message = (
            f"AE query failed ({error.code}): {error.read().decode(errors='replace')}"
        )
        if error.code == 403:
            message += "\nthe token (store/.env) must carry Account Analytics: Read"
        raise SystemExit(message)
