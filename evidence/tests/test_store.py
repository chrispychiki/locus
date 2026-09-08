import hashlib
import urllib.request

import pytest
from _support import FakeVerify
from locus.evidence.store import S3Store, store_for

ACCOUNT = "acc1234567890"
TOKEN = "cf-token-value"


def _isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_store_for_interprets_urls_and_fails_loud(monkeypatch):
    # A machine environment carrying Cloudflare identity would flip the bare-bucket case
    # onto the R2 self-serve path — and its live token-verify call; this test is about
    # URL interpretation alone.
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    store = store_for("s3://my-chunks")
    assert isinstance(store, S3Store)
    assert store.bucket == "my-chunks"

    with pytest.raises(ValueError, match="the store is the whole bucket"):
        store_for("s3://my-chunks/abc123")

    with pytest.raises(ValueError, match="unknown store scheme 'file'"):
        store_for("file:///tmp/chunks")
    with pytest.raises(ValueError, match="names no bucket"):
        store_for("s3://")
    with pytest.raises(ValueError, match="carries a query"):
        store_for("s3://my-chunks?endpoint=https://acc.example.com")


def test_an_r2_store_self_serves_keys_and_endpoint_from_the_declaration(
    monkeypatch, tmp_path
):
    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        FakeVerify({"success": True, "result": {"id": "tokenid42"}}),
    )

    store = store_for("s3://locus-chunks")
    creds = store.client._request_signer._credentials
    assert creds.access_key == "tokenid42"
    assert creds.secret_key == hashlib.sha256(TOKEN.encode()).hexdigest()
    assert (
        store.client.meta.endpoint_url == f"https://{ACCOUNT}.r2.cloudflarestorage.com"
    )


def test_an_operators_own_aws_keys_are_never_overridden(monkeypatch, tmp_path):
    from locus.evidence import store as store_module

    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAOPERATOR")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "operator-secret")

    def never(*args, **kwargs):
        raise AssertionError(
            "an already-credentialed environment must not reach for the Cloudflare token"
        )

    monkeypatch.setattr(urllib.request, "urlopen", never)
    passed = {}
    monkeypatch.setattr(
        store_module,
        "S3Store",
        lambda bucket, **kwargs: passed.update(bucket=bucket, **kwargs),
    )

    store_module.store_for("s3://locus-chunks")
    assert passed["access_key"] is None and passed["secret"] is None, (
        "an already-credentialed environment falls through to boto3's own chain"
    )


class _FakePaginator:
    """list_objects_v2 pagination semantics over an in-memory key set, Delimiter included —
    what the sharded listing actually leans on."""

    def __init__(self, blobs):
        self.blobs = blobs

    def paginate(self, Bucket=None, Prefix="", Delimiter=None):
        contents, prefixes = [], []
        seen = set()
        from datetime import datetime, timezone

        stamp = datetime.fromtimestamp(0, tz=timezone.utc)
        for key in sorted(self.blobs):
            if not key.startswith(Prefix):
                continue
            rest = key[len(Prefix) :]
            if Delimiter and Delimiter in rest:
                shared = Prefix + rest.split(Delimiter)[0] + Delimiter
                if shared not in seen:
                    seen.add(shared)
                    prefixes.append({"Prefix": shared})
            else:
                contents.append(
                    {
                        "Key": key,
                        "Size": len(self.blobs[key]),
                        "LastModified": stamp,
                        "ETag": '"x"',
                    }
                )
        yield {"Contents": contents, "CommonPrefixes": prefixes}


class _FakeClient:
    def __init__(self, blobs):
        self.blobs = blobs

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self.blobs)


def test_sharded_listing_yields_every_object_exactly_once():
    # The listing shards itself over discovered '/' levels and paginates them in
    # parallel; the contract is set equality with the flat listing — every key
    # once, unordered, shallow stragglers included.
    blobs = {
        f"snip{s}/2026-07-{d:02d}/v{v}/obj": b"x" * (s + d + v)
        for s in range(3)
        for d in range(1, 8)
        for v in range(4)
    }
    blobs["shallow-object"] = b"y"
    blobs["snip0/stray"] = b"z"
    store = object.__new__(S3Store)
    store.bucket = "b"
    store.client = _FakeClient(blobs)
    listed = list(store.objects(""))
    assert sorted(k for k, _s, _u, _e in listed) == sorted(blobs)
    assert len(listed) == len(blobs)

    scoped = list(store.objects("snip1/"))
    assert sorted(k for k, _s, _u, _e in scoped) == sorted(
        k for k in blobs if k.startswith("snip1/")
    )


class _SlowPaginator:
    """Shards that paginate the way a store does — one round-trip per page, several pages
    deep — so what a listing hands over mid-shard is observable."""

    PAGES = 3
    SHARDS = 32

    def __init__(self, per_page_s=0.2):
        self.per_page_s = per_page_s
        self.pages = {}

    def paginate(self, Bucket=None, Prefix="", Delimiter=None):
        import time
        from datetime import datetime, timezone

        stamp = datetime.fromtimestamp(0, tz=timezone.utc)
        if Delimiter:
            yield {
                "Contents": [],
                "CommonPrefixes": [
                    {"Prefix": f"s{i:02d}/"} for i in range(self.SHARDS)
                ],
            }
            return
        for page in range(self.PAGES):
            if page:
                time.sleep(self.per_page_s)
            self.pages[Prefix] = page + 1
            yield {
                "Contents": [
                    {
                        "Key": f"{Prefix}{page}",
                        "Size": 1,
                        "LastModified": stamp,
                        "ETag": '"x"',
                    }
                ],
                "CommonPrefixes": [],
            }


def _slow_store(paginator):
    store = object.__new__(S3Store)
    store.bucket = "b"
    store.client = type("C", (), {"get_paginator": lambda _self, _name: paginator})()
    return store


def test_a_shards_keys_leave_as_its_pages_land():
    """Page one's keys are as usable as the last's, so a consumer reaches them without waiting on the pages behind them."""
    paginator = _SlowPaginator()
    listing = _slow_store(paginator).objects("")

    first = next(listing)
    assert max(paginator.pages.values()) < _SlowPaginator.PAGES, (
        "a key reached the consumer before any shard had finished paginating"
    )

    every = [first, *listing]
    assert sorted(k for k, _s, _u, _e in every) == sorted(
        f"s{i:02d}/{page}"
        for i in range(_SlowPaginator.SHARDS)
        for page in range(_SlowPaginator.PAGES)
    )


def test_a_listing_nobody_is_draining_lets_go(monkeypatch):
    """A consumer stops early on a failed GET or an interrupt; its shard threads must not stay parked on a queue with no reader, because the pool joins before the listing can return."""
    import threading

    from locus.evidence import store as store_module

    # One slot, so a shard thread is genuinely parked on a full queue when the
    # consumer walks away — the state the abandonment signal exists for, and one
    # a backlog deeper than the listing never reaches.
    monkeypatch.setattr(store_module, "_LISTING_BACKLOG", 1)
    listing = _slow_store(_SlowPaginator(per_page_s=0.01)).objects("")
    next(listing)
    let_go = threading.Thread(target=listing.close, daemon=True)
    let_go.start()
    let_go.join(timeout=15)
    assert not let_go.is_alive(), (
        "shard threads are still parked on a queue nobody drains"
    )
