# The R2 credential self-serve: one stored Cloudflare token, and every S3 read
# derives its own key pair. Nothing is exported by hand and nothing derived is
# persisted, so these are the guarantees the whole no-ceremony story rests on.

import hashlib
import urllib.request

import pytest
from _support import FakeVerify
from locus.evidence.cloudflare import (
    account_and_token,
    r2_s3_credentials,
)

ACCOUNT = "acc1234567890"
TOKEN = "cf-token-value"


def _isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_the_r2_secret_is_the_token_hashed_locally(monkeypatch, tmp_path):
    _isolate_cache(monkeypatch, tmp_path)
    verify = FakeVerify({"success": True, "result": {"id": "tokenid42"}})
    monkeypatch.setattr(urllib.request, "urlopen", verify)

    creds = r2_s3_credentials(ACCOUNT, TOKEN)
    assert creds == {
        "endpoint": f"https://{ACCOUNT}.r2.cloudflarestorage.com",
        "access_key_id": "tokenid42",
        "secret_access_key": hashlib.sha256(TOKEN.encode()).hexdigest(),
    }
    url, auth = verify.requests[0]
    assert url == (
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/tokens/verify"
    )
    assert auth == f"Bearer {TOKEN}"


def test_the_token_id_is_fetched_once_and_the_secret_never_persisted(
    monkeypatch, tmp_path
):
    from locus.evidence.cloudflare import _r2_token_id_cache_path

    _isolate_cache(monkeypatch, tmp_path)
    verify = FakeVerify({"success": True, "result": {"id": "tokenid42"}})
    monkeypatch.setattr(urllib.request, "urlopen", verify)

    first = r2_s3_credentials(ACCOUNT, TOKEN)
    second = r2_s3_credentials(ACCOUNT, TOKEN)
    assert first == second
    assert len(verify.requests) == 1, "the id is stable and not secret — cache it"

    cached = _r2_token_id_cache_path().read_text()
    assert "tokenid42" in cached
    assert TOKEN not in cached
    assert first["secret_access_key"] not in cached, (
        "the secret is recomputed every call; nothing derived is written to disk"
    )


def test_a_different_token_under_the_same_account_is_a_cache_miss(
    monkeypatch, tmp_path
):
    _isolate_cache(monkeypatch, tmp_path)
    ids = iter(["idA", "idB"])
    calls = []

    class Rotating(FakeVerify):
        def __call__(self, request, timeout=None):
            calls.append(request.full_url)
            self.body = {"success": True, "result": {"id": next(ids)}}
            return self

    monkeypatch.setattr(
        urllib.request, "urlopen", Rotating({"success": True, "result": {}})
    )
    assert r2_s3_credentials(ACCOUNT, "token-one")["access_key_id"] == "idA"
    assert r2_s3_credentials(ACCOUNT, "token-two")["access_key_id"] == "idB"
    assert len(calls) == 2, "a rotated token must never reuse the old key id"


def test_a_rejected_token_fails_loud(monkeypatch, tmp_path):
    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        FakeVerify({"success": False, "errors": [{"message": "Invalid token"}]}),
    )
    with pytest.raises(RuntimeError, match="token verify failed"):
        r2_s3_credentials(ACCOUNT, TOKEN)


def test_a_cloudflare_only_plane_fails_loud_on_a_missing_declaration(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    with pytest.raises(SystemExit, match="Cloudflare-only plane"):
        account_and_token("telemetry")

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    with pytest.raises(SystemExit, match="CLOUDFLARE_API_TOKEN is not set"):
        account_and_token("telemetry")
