"""The store read path: the operator's own bucket, read directly.

S3Store reads the deployment's R2 bucket over R2's S3 endpoint via boto3, the access-key pair self-served from the Cloudflare API token (cloudflare.py). The protocol is standard S3, so a different S3-compatible bucket works in principle — boto3's standard AWS chain supplies its keys and endpoint — but the shipped deployment is R2 end to end, and swapping the bucket swaps only the read path: the worker that writes it is R2's. Locus reads the bucket directly with the operator's credentials; the deployment's worker is the *write* gate (browser-facing auth/throttling/validation) and has no role on the read path. store_for() is the only place a store URL is interpreted into a store: `s3://bucket`, fail-loud on anything else.
"""

import os
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from queue import Full, Queue

from .chunk import decode_or_skip, parse_chunk_key
from .cloudflare import declared_account, r2_endpoint, r2_s3_credentials

GET_CONCURRENCY = 64
LIST_CONCURRENCY = 24
# Both are network-latency windows, not CPU workers — a chunk GET and a LIST page each cost a
# round-trip (~a quarter second against R2) regardless of the machine, so the right window is what
# the store tolerates, measured: GETs scaled cleanly through 64 in-flight (~450 chunks/s against
# ~40 at 10), and a whole-bucket LIST sharded over its discovered prefixes at 24 ran ~11× the
# serial pagination. CPU-bound work (distillation) sizes itself from the machine instead.

_MIN_SHARDS = LIST_CONCURRENCY
_MAX_SHARDS = 512
# Slack between the shards listing and whoever is consuming them — deep enough that a
# consumer pausing over one key never idles the LIST window, shallow enough that the
# listing cannot run arbitrarily far ahead and hold a corpus of keys in memory.
_LISTING_BACKLOG = 4096


class S3Store:
    def __init__(
        self,
        bucket: str,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret: str | None = None,
    ):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        # The connection pool must fit the GET window, or the extra in-flight requests
        # each open and discard an unpooled connection.
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret,
            region_name="auto" if endpoint else None,
            config=Config(max_pool_connections=GET_CONCURRENCY),
        )

    def _pages(self, prefix: str) -> Iterator[tuple[str, int, int, str]]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield (
                    obj["Key"],
                    obj["Size"],
                    int(obj["LastModified"].timestamp() * 1000),
                    obj["ETag"].strip('"'),
                )

    def _shards(self, prefix: str) -> tuple[list[str], list[tuple]]:
        """Subtree prefixes that partition the listing, discovered with delimiter LISTs — layout-blind, so any key shape shards by whatever '/' levels it actually has. Returns (shard prefixes, the objects already enumerated during discovery, which no shard covers). Expansion stops once the shards can fill the LIST window; a level that would explode past _MAX_SHARDS is discarded whole — thousands of one-page LISTs are slower than the pages they replace, and its half-enumerated objects with it, so nothing is ever listed twice. Empty shards with spill means discovery already enumerated everything; the original prefix coming back as the lone shard means its sublevel was the explosive kind — the serial case."""
        paginator = self.client.get_paginator("list_objects_v2")

        def expand(shard: str) -> tuple[list[str], list[tuple]]:
            subs, contents = [], []
            for page in paginator.paginate(
                Bucket=self.bucket, Prefix=shard, Delimiter="/"
            ):
                subs += [c["Prefix"] for c in page.get("CommonPrefixes", [])]
                contents += [
                    (
                        obj["Key"],
                        obj["Size"],
                        int(obj["LastModified"].timestamp() * 1000),
                        obj["ETag"].strip('"'),
                    )
                    for obj in page.get("Contents", [])
                ]
            return subs, contents

        shards, spill = [prefix], []
        for _level in range(2):
            if len(shards) >= _MIN_SHARDS:
                break
            with ThreadPoolExecutor(max_workers=LIST_CONCURRENCY) as pool:
                rounds = list(pool.map(expand, shards))
            next_shards = [sub for subs, _ in rounds for sub in subs]
            if len(next_shards) > _MAX_SHARDS:
                break
            spill += [
                (shard, obj)
                for shard, (_subs, contents) in zip(shards, rounds)
                for obj in contents
            ]
            if not next_shards:
                return [], spill
            shards = next_shards
        return shards, spill

    def shard_rows(self, prefix: str = "") -> Iterator[tuple[str, tuple]]:
        """Each object paired with the shard that listed it. One shard's keys arrive in the store's own lexicographic order, because a shard is paginated serially — so the shard is what tells a consumer folding keys together that no more of a group can arrive. Shards themselves interleave, so the pairing rides every row."""
        shards, spill = self._shards(prefix)
        yield from spill
        if not shards:
            return
        if len(shards) == 1:
            for row in self._pages(shards[0]):
                yield shards[0], row
            return

        # A shard's pages are serial round-trips, so its keys leave as each page
        # lands: a consumer waiting on the whole shard is a fetch window idling
        # through pagination it could have been working through.
        rows: Queue = Queue(maxsize=_LISTING_BACKLOG)
        drained = object()
        abandoned = threading.Event()

        def offer(row) -> bool:
            """Hand one row over, giving up if the consumer has gone. A listing whose reader stopped early — a failed GET, an interrupt — must not leave its shard threads parked on a queue nobody drains."""
            while not abandoned.is_set():
                try:
                    rows.put(row, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def list_shard(shard: str) -> None:
            try:
                for row in self._pages(shard):
                    if not offer((shard, row)):
                        return
            finally:
                offer(drained)

        pool = ThreadPoolExecutor(max_workers=LIST_CONCURRENCY)
        futures = [pool.submit(list_shard, shard) for shard in shards]
        try:
            listing = len(shards)
            while listing:
                row = rows.get()
                if row is drained:
                    listing -= 1
                else:
                    yield row
            # A shard reports itself drained on its way out however it left, so
            # the count alone cannot tell a listed shard from a failed one.
            for future in futures:
                future.result()
        finally:
            abandoned.set()
            # A shard that never started has nothing to abandon, and its first page
            # is a round-trip nobody is waiting for.
            pool.shutdown(cancel_futures=True)

    def objects(self, prefix: str = "") -> Iterator[tuple[str, int, int, str]]:
        """(key, bytes, uploaded_ms, etag) for every object under the prefix, in no particular order — the listing is sharded by discovered subtree prefixes and paginated in parallel, because LIST pages are serial round-trips and a real corpus is hundreds of them. The upload time is epoch-ms UTC, distinct from the slice's open time — a slice opens on the visitor's clock and its chunks land whenever the device sends them, so the pair reads delivery latency. The ETag is the object body's MD5 (chunks are single-request writes), the content fingerprint a later load skips an unchanged GET on."""
        for _shard, row in self.shard_rows(prefix):
            yield row

    def keys(self, prefix: str = "") -> Iterator[str]:
        for key, _size, _uploaded, _etag in self.objects(prefix):
            yield key

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def events_under(
        self,
        prefix: str = "",
        concurrency: int = GET_CONCURRENCY,
        on_errors=None,
        loaded=None,
        on_fetched=None,
        expired=None,
        stats=None,
    ) -> Iterator[tuple[str, dict]]:
        """Every chunk under an opaque key prefix, decoded to (visitor, event) pairs — the load's read side. R2 has no batch GET, so chunks are fetched concurrently (a sliding window of `concurrency` in-flight GETs) and decoded as they land; order doesn't matter because hydration dedups by content hash and orders at query time. Identity rides in the chunk (the payload carries visitorId/sliceId), except the snippet id, which lives only in the key. A key that is not a chunk key is fetched and decoded anyway, so decode_or_skip names the garbage rather than this loop passing over it in silence. on_errors receives each decodable chunk carrying payload `errors` (see decode_or_skip).

        loaded(key, etag) -> bool skips the GET for an object already held, so only new or changed objects cross the network. on_fetched(key, etag, uploaded_ms) fires for a chunk that decoded cleanly (no lost events), after its events are yielded — a lossy or garbage object is left unrecorded so a later load re-fetches and re-reports it. uploaded_ms is the object's store arrival time from the LIST: the store's copy dies with the object at the retention horizon, so the load is the one chance to keep it.

        expired(key) -> bool passes over an object holding a recording past the deployment's retention horizon, before the GET: what the horizon has dropped must never land in the db, and hydrating it only to sweep it again would re-fetch it on every load. The horizon is a deployment policy, so it arrives as the caller's predicate (retention.py) — the read path holds no policy of its own.

        stats, when given, is a LoadStats (load.py): GET and decode time land in it per chunk (thread-seconds — divide by the window to compare against wall), and each drained or skipped chunk ticks its counters, which its heartbeat reads on its own clock."""

        def fetch(key, etag, uploaded_ms):
            parsed = parse_chunk_key(key)
            clean = []
            t0 = time.perf_counter()
            blob = self.get(key)
            t1 = time.perf_counter()
            events = decode_or_skip(
                blob,
                key,
                parsed["snippet"] if parsed else None,
                on_errors=on_errors,
                on_clean=lambda _k: clean.append(True),
            )
            if stats is not None:
                stats.add(
                    get_s=t1 - t0, decode_s=time.perf_counter() - t1, bytes=len(blob)
                )
            return key, etag, uploaded_ms, events, bool(clean)

        def drain(future):
            key, etag, uploaded_ms, events, clean = future.result()
            yield from events
            if on_fetched is not None and clean:
                on_fetched(key, etag, uploaded_ms)
            if stats is not None:
                stats.add(chunks=1, events=len(events))
                if not clean:
                    stats.add(corrupt=1 if not events else 0, lossy=1 if events else 0)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            window = deque()
            for key, _size, uploaded_ms, etag in self.objects(prefix):
                if expired is not None and expired(key):
                    if stats is not None:
                        stats.add(expired=1)
                    continue
                if loaded is not None and loaded(key, etag):
                    if stats is not None:
                        stats.add(skipped=1)
                    continue
                window.append(pool.submit(fetch, key, etag, uploaded_ms))
                if len(window) >= concurrency * 2:
                    yield from drain(window.popleft())
            while window:
                yield from drain(window.popleft())


def store_for(url: str) -> S3Store:
    """The only place a store URL is interpreted into a store. The store wraps the whole bucket — `s3://bucket`, nothing else in the URL — and the read path parses the snippet-id-first key layout (`{snippetId}/{date}/{visitor}/{slice}/{chunk}`) directly, so one db spans every snippet in the operator's bucket.

    Credentials self-serve: the S3 key pair is derived from CLOUDFLARE_API_TOKEN on the spot and the endpoint from the declared account (cloudflare.py), so the operator sets store/.env and nothing exports keys by hand. An already-set AWS access key falls through to boto3's standard chain untouched."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "s3":
        raise ValueError(
            f"unknown store scheme {parsed.scheme!r} in {url!r}; the one "
            f"form that exists is s3://bucket"
        )
    if not parsed.netloc:
        raise ValueError(f"store url {url!r} names no bucket")
    if parsed.path.strip("/"):
        raise ValueError(
            f"store url {url!r} has a path; the store is the whole bucket — "
            f"drop the snippet, use s3://bucket"
        )
    if parsed.query:
        raise ValueError(
            f"store url {url!r} carries a query; the one form that exists is "
            f"s3://bucket — endpoint and credentials self-serve (cloudflare.py)"
        )
    account = declared_account()
    endpoint = access_key = secret = None
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if account and token and not os.environ.get("AWS_ACCESS_KEY_ID"):
        creds = r2_s3_credentials(account, token)
        access_key, secret = creds["access_key_id"], creds["secret_access_key"]
        endpoint = creds["endpoint"]
    elif account:
        # The AWS-chain path against the R2 store: the keys come from the
        # environment, the endpoint from the account.
        endpoint = r2_endpoint(account)
    return S3Store(
        parsed.netloc, endpoint=endpoint, access_key=access_key, secret=secret
    )
