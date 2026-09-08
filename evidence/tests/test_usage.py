import io
import json
import urllib.error

import pytest
from locus.evidence import usage


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_usage_query_binds_the_account_and_returns_data(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        sent["auth"] = request.headers.get("Authorization")
        return FakeResponse({"data": {"viewer": {}}, "errors": None})

    monkeypatch.setattr(usage.urllib.request, "urlopen", fake_urlopen)

    query = "query($account: String!) { viewer { accounts } }"
    assert usage.usage_query(query) == {"viewer": {}}
    assert sent["url"] == "https://api.cloudflare.com/client/v4/graphql"
    assert sent["auth"] == "Bearer tok"
    assert sent["body"] == {"query": query, "variables": {"account": "acct1"}}

    usage.usage_query("query { viewer { budgets } }")
    assert "variables" not in sent["body"], (
        "no $account in the query means no variables are sent"
    )


def test_usage_query_surfaces_graphql_errors_and_names_the_scope(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")
    monkeypatch.setattr(
        usage.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            {
                "data": None,
                "errors": [{"message": "not authorized to access this account"}],
            }
        ),
    )
    with pytest.raises(SystemExit) as caught:
        usage.usage_query("query { viewer }")
    assert "not authorized" in str(caught.value)
    assert "Account Analytics: Read" in str(caught.value)


def test_usage_query_fails_loud_without_token_or_account(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")
    with pytest.raises(SystemExit, match="CLOUDFLARE_API_TOKEN"):
        usage.usage_query("query { viewer }")

    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(SystemExit, match="CLOUDFLARE_ACCOUNT_ID is not set"):
        usage.usage_query("query { viewer }")


def test_usage_query_surfaces_http_errors(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct1")

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 500, "boom", None, io.BytesIO(b"server broke")
        )

    monkeypatch.setattr(usage.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit, match="500.*server broke"):
        usage.usage_query("query { viewer }")
