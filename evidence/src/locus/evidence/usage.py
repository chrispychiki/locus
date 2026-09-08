"""Read Cloudflare's usage-analytics plane — worker requests, R2 storage and operation counts — via the GraphQL API (https://developers.cloudflare.com/analytics/graphql-api/).

The query is yours: datasets, dimensions, and filters are Cloudflare's (workersInvocationsAdaptive, r2OperationsAdaptiveGroups, r2StorageAdaptiveGroups, …) — this owns only the connection. Write `$account` in the query and it binds to the deployment's account id, resolved the same way every other plane resolves it (cloudflare.py). Usage numbers, never dollars: the invoice math (usage × published rate − free tier) stays with the reader.
"""

import json
import urllib.error
import urllib.request

from .cloudflare import CF_API, account_and_token


def usage_query(query: str):
    account_id, token = account_and_token("usage analytics")
    body = {"query": query}
    if "$account" in query:
        body["variables"] = {"account": account_id}
    request = urllib.request.Request(
        f"{CF_API}/graphql",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"usage query failed ({error.code}): {error.read().decode(errors='replace')}"
        )
    if payload.get("errors"):
        message = json.dumps(payload["errors"], indent=2)
        if "auth" in message.lower():
            message += "\nthe token (store/.env) must carry Account Analytics: Read"
        raise SystemExit(f"usage query failed: {message}")
    return payload.get("data")
