"""The telemetry plane's connection. The query is the caller's; what this owns is reaching the right account with the right token and failing loud when it can't — the same contract usage.py holds, from the same resolution."""

import io
import json
import urllib.error

import pytest
from locus.evidence import analytics_engine as ae


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ae_query_posts_the_sql_to_the_deployment_account(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = request.data.decode()
        sent["auth"] = request.headers.get("Authorization")
        return FakeResponse({"data": [{"n": "2"}], "rows": 1})

    monkeypatch.setattr(ae.urllib.request, "urlopen", fake_urlopen)

    sql = "SELECT blob1 AS metric FROM locus_capture"
    assert ae.ae_query(sql)["data"] == [{"n": "2"}]
    assert sent["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/acct1/analytics_engine/sql"
    )
    assert sent["auth"] == "Bearer tok"
    assert sent["body"] == sql, "the SQL goes to the wire as written"


def test_ae_query_fails_loud_without_token_or_account(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")
    with pytest.raises(SystemExit, match="CLOUDFLARE_API_TOKEN"):
        ae.ae_query("SELECT 1")

    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(SystemExit, match="CLOUDFLARE_ACCOUNT_ID is not set"):
        ae.ae_query("SELECT 1")


def test_ae_query_surfaces_http_errors_with_the_body(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 400, "bad", None, io.BytesIO(b"syntax error")
        )

    monkeypatch.setattr(ae.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit, match="400.*syntax error"):
        ae.ae_query("SELECT nonsense")


def test_ae_query_403_names_the_missing_scope(monkeypatch):
    """A token minted without the analytics read scope 403s here — the error names the remedy."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "forbidden",
            None,
            io.BytesIO(b"authentication error"),
        )

    monkeypatch.setattr(ae.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit) as caught:
        ae.ae_query("SELECT 1")
    assert "Account Analytics: Read" in str(caught.value)


def test_both_planes_resolve_the_account_and_token_the_same_way(monkeypatch):
    """One deployment, one account, one token — a plane that resolved them its own way could read a
    different deployment's telemetry than the one whose bucket the load just read."""
    from locus.evidence.cloudflare import account_and_token

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")
    assert account_and_token("Analytics Engine") == ("acct1", "tok")
    assert account_and_token("usage") == ("acct1", "tok")
