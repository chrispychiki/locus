"""The Cloudflare side of a deployment: the declared account, the one token, and the S3 credentials R2 derives from it.

account_and_token is what every Cloudflare plane needs before it can ask anything, both from store/.env (export_declared) — so a deployment that can read its bucket can read its telemetry, with nothing else to configure.

r2_s3_credentials derives the S3 key pair mechanically from the Cloudflare API token (access key id = the token's id, secret = SHA-256 of the token value — https://developers.cloudflare.com/r2/api/tokens/: "Access Key ID: The `id` of the API token. Secret Access Key: The SHA-256 hash of the API token `value`."), so setup adds no manual credential step.
"""

import hashlib
import json
import os
from pathlib import Path

# The Cloudflare REST API root — every plane's endpoint composes from it.
CF_API = "https://api.cloudflare.com/client/v4"


def account_and_token(plane: str) -> tuple[str, str]:
    account_id = declared_account()
    if account_id is None:
        raise SystemExit(
            f"CLOUDFLARE_ACCOUNT_ID is not set (store/.env); {plane} is a "
            f"Cloudflare-only plane"
        )
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise SystemExit(
            f"CLOUDFLARE_API_TOKEN is not set (store/.env); reading {plane} "
            f"needs the token to carry Account Analytics: Read"
        )
    return account_id, token


def declared_account() -> str | None:
    """The deployment's Cloudflare account id, from CLOUDFLARE_ACCOUNT_ID (store/.env, exported by export_declared) — the deployment handle the bucket shares with the Analytics Engine plane, so every plane resolves its account from the one declaration. None when the environment declares no account."""
    return os.environ.get("CLOUDFLARE_ACCOUNT_ID")


def r2_endpoint(account_id: str) -> str:
    """R2's S3 endpoint for an account — the one composition of the `{account}.r2.cloudflarestorage.com` URL. r2_s3_credentials returns it beside the keys it derives (the consumers with derived keys read it there); the credless paths — an AWS-chain store, the profile credential_process writes — compose it here."""
    return f"https://{account_id}.r2.cloudflarestorage.com"


def r2_s3_credentials(account_id: str, api_token: str) -> dict:
    """Derive R2's S3 credentials from a Cloudflare API token with R2 access: access key id = the token's id, secret = SHA-256 of the token value. The id is fetched once via the token-verify API and cached (it is stable and not secret); the secret is recomputed locally every call, so nothing derived is persisted. The S3 client itself (store.py) knows nothing about Cloudflare."""
    secret = hashlib.sha256(api_token.encode()).hexdigest()
    access_key_id = _cached_r2_token_id(account_id, api_token)
    if access_key_id is None:
        import urllib.request

        request = urllib.request.Request(
            f"{CF_API}/accounts/{account_id}/tokens/verify",
            headers={"Authorization": f"Bearer {api_token}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            verify = json.loads(response.read())
        if not verify.get("success"):
            raise RuntimeError(f"token verify failed: {verify.get('errors')}")
        access_key_id = verify["result"]["id"]
        _cache_r2_token_id(account_id, api_token, access_key_id)
    return {
        "endpoint": r2_endpoint(account_id),
        "access_key_id": access_key_id,
        "secret_access_key": secret,
    }


def _r2_token_id_cache_path() -> Path:
    from .deployment import machine_cache_dir

    return machine_cache_dir() / "r2_token_ids.json"


def _r2_token_fingerprint(account_id: str, api_token: str) -> str:
    return hashlib.sha256(f"r2-token-id:{account_id}:{api_token}".encode()).hexdigest()[
        :16
    ]


def _cached_r2_token_id(account_id: str, api_token: str) -> str | None:
    try:
        cache = json.loads(_r2_token_id_cache_path().read_text())
    except (OSError, ValueError):
        return None
    return cache.get(_r2_token_fingerprint(account_id, api_token))


def _cache_r2_token_id(account_id: str, api_token: str, token_id: str) -> None:
    path = _r2_token_id_cache_path()
    try:
        cache = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        cache = {}
    cache[_r2_token_fingerprint(account_id, api_token)] = token_id
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    except OSError:
        pass
