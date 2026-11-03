import pytest
from _support import (
    SNIPPET,
    T0,
    VISITOR,
    FakeStore,
    chunk_at,
    chunk_blob,
    counted,
    meta,
)
from locus.evidence.chunk import slice_date
from locus.evidence.inventory import Inventory

META = counted(meta(T0), 1)


def listed(store, local=None, prefix=""):
    """An Inventory drained, the way its caller drains it: the rows as they came, then the totals it left behind."""
    listing = Inventory(store, local, prefix)
    return list(listing), listing.totals


def test_inventory_groups_and_diffs_the_listing():
    rid_a = f"0{T0}-aaaa"
    rid_b = f"0{T0 + 90_000}-bbbb"
    blob = chunk_blob(rid_a, [META])
    date = slice_date(rid_a)
    store = FakeStore(
        {
            f"{SNIPPET}/{date}/{VISITOR}/{rid_a}/{T0}000001.json.gz": blob,
            f"{SNIPPET}/{date}/{VISITOR}/{rid_a}/{T0}000002.json.gz": blob,
            f"{SNIPPET}/{date}/{VISITOR}/{rid_b}/{T0}000003.json.gz": blob,
            f"{SNIPPET}/{date}/other-visitor/{rid_a}/{T0}000001.json.gz": blob,
            f"othersnip/{date}/third-visitor/{rid_a}/{T0}000001.json.gz": blob,
        }
    )

    slices, totals = listed(store, {VISITOR: {rid_a}})
    assert totals["slices"] == 4 and totals["new"] == 3
    rows = {(g["visitor_id"], g["slice_id"]): g for g in slices}
    assert set(rows) == {
        (VISITOR, rid_a),
        (VISITOR, rid_b),
        ("other-visitor", rid_a),
        ("third-visitor", rid_a),
    }

    mine = rows[(VISITOR, rid_a)]
    assert mine["snippet"] == SNIPPET
    assert mine["chunks"] == 2 and mine["bytes"] == 2 * len(blob)
    assert mine["open_ms"] == T0
    assert mine["loaded"] is True

    assert rows[(VISITOR, rid_b)]["loaded"] is False
    assert rows[("other-visitor", rid_a)]["loaded"] is False, (
        "no local presence means everything is new"
    )
    assert rows[("third-visitor", rid_a)]["snippet"] == "othersnip", (
        "a whole-store slice listing attributes each row to its snippet"
    )


class _Listing:
    """A store whose listing hands rows over one at a time, each tagged with the shard it came from, recording what it has handed over so far."""

    def __init__(self, rows):
        self.rows = rows
        self.handed = []

    def shard_rows(self, prefix=""):
        for shard, key in self.rows:
            self.handed.append(key)
            yield shard, (key, 1, 0, "etag")


def _key(slice_id, n=1):
    return f"{SNIPPET}/{slice_date(slice_id)}/{VISITOR}/{slice_id}/{T0}00000{n}.json.gz"


def test_a_slice_leaves_before_the_listing_has_finished():
    # A whole-store LIST is minutes of round-trips, so a row that waited on the
    # last of them would reach its reader only once the whole listing had.
    ids = [f"0{T0 + i}-aaa{i}" for i in range(4)]
    shard = f"{SNIPPET}/{slice_date(ids[0])}/"
    listing = _Listing([(shard, _key(i)) for i in ids])

    first = next(iter(Inventory(listing, {})))
    assert first["slice_id"] == ids[0]
    assert len(listing.handed) < len(ids), (
        "the first slice is handed over while the rest are still being listed"
    )


def test_a_slice_is_not_closed_by_a_key_from_another_shard():
    # Shards are paginated in parallel and interleave in the stream, so "the next
    # key belongs to a different slice" says nothing on its own. Only the shard
    # that owns a slice can report it done, and closing early would publish a row
    # counting a fraction of its chunks.
    mine, theirs = f"0{T0}-aaaa", f"0{T0 + 90_000}-bbbb"
    later = f"0{T0 + 180_000}-cccc"
    a, b = f"{SNIPPET}/{slice_date(mine)}/", f"other/{slice_date(theirs)}/"
    listing = _Listing(
        [
            (a, _key(mine, 1)),
            (b, _key(theirs, 1)),
            (a, _key(mine, 2)),
            (b, _key(theirs, 2)),
            (a, _key(later, 1)),
        ]
    )

    rows = {row["slice_id"]: row for row in Inventory(listing, {})}
    assert rows[mine]["chunks"] == 2 and rows[theirs]["chunks"] == 2
    assert rows[later]["chunks"] == 1


def test_a_slice_straddling_shards_fails_the_listing_loud():
    # The fold's precondition — one shard holds all of a slice's keys — is the
    # key layout's guarantee; a violation must fail the listing rather than
    # publish two partial rows for one slice.
    rid, other = f"0{T0}-aaaa", f"0{T0 + 90_000}-bbbb"
    shard_a, shard_b = "a/", "b/"
    listing = _Listing(
        [(shard_a, _key(rid, 1)), (shard_a, _key(other, 1)), (shard_b, _key(rid, 2))]
    )

    with pytest.raises(RuntimeError, match="span more than one listing shard"):
        list(Inventory(listing, {}))


def test_every_slice_is_returned_however_large_the_scope():
    # The rows are the arrivals plane, and an arrival only answers a birth if the two
    # can be joined — on (snippet, visitor, slice). A listing that rolls up past some
    # size withholds exactly that key, and the reader goes to the raw bucket to get it
    # back. So there is no size at which the rows stop coming.
    n = 200
    blobs = dict(chunk_at(f"v{i:04d}-visitor", T0 + i * 86_400_000) for i in range(n))

    slices, _totals = listed(FakeStore(blobs))
    assert len(slices) == n
    assert len({r["slice_id"] for r in slices}) == n


def test_a_slice_carries_when_its_chunks_actually_landed():
    # The slice opens on the visitor's clock; its chunks land whenever the device
    # manages to send them, which can be days later. Both facts, in UTC ms, because
    # the gap between them is what tells a late delivery from a lost one — and the
    # only other way to get it is the raw bucket, whose listing is in local time.
    rid = f"0{T0}-aaaa"
    date = slice_date(rid)
    k1 = f"{SNIPPET}/{date}/{VISITOR}/{rid}/{T0}000001.json.gz"
    k2 = f"{SNIPPET}/{date}/{VISITOR}/{rid}/{T0}000002.json.gz"
    blob = chunk_blob(rid, [META])
    day = 86_400_000
    store = FakeStore({k1: blob, k2: blob}, uploaded={k1: T0 + 3_000, k2: T0 + 2 * day})

    [row], _totals = listed(store)
    assert row["open_ms"] == T0
    assert row["first_upload_ms"] == T0 + 3_000
    assert row["last_upload_ms"] == T0 + 2 * day


def test_inventory_counts_foreign_objects_instead_of_hiding_them():
    rid = f"0{T0}-aaaa"
    blob = chunk_blob(rid, [META])
    store = FakeStore(
        {
            f"{SNIPPET}/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz": blob,
            "backup.tar.gz": b"x" * 10,
            f"{SNIPPET}/notes/readme.txt": b"y" * 5,
            f"{SNIPPET}/2026-01-01/{VISITOR}/not-a-slice/{T0}000001.json.gz": b"z",
        }
    )
    _slices, totals = listed(store, {})
    assert totals["slices"] == 1
    assert totals["foreign_keys"] == 3
    assert totals["foreign_bytes"] == 16


def test_inventory_scopes_to_a_key_prefix():
    rid_day1 = f"0{T0}-aaaa"
    rid_day2 = f"0{T0 + 86_400_000}-bbbb"
    day1, day2 = slice_date(rid_day1), slice_date(rid_day2)
    assert day1 != day2
    store = FakeStore(
        {
            f"{SNIPPET}/{day1}/{VISITOR}/{rid_day1}/{T0}000001.json.gz": chunk_blob(
                rid_day1, [META]
            ),
            f"{SNIPPET}/{day2}/{VISITOR}/{rid_day2}/{T0 + 86_400_000}000001.json.gz": chunk_blob(
                rid_day2, [META]
            ),
        }
    )

    assert listed(store)[1]["slices"] == 2
    assert listed(store, prefix=f"{SNIPPET}/")[1]["slices"] == 2

    one, one_totals = listed(store, prefix=f"{SNIPPET}/{day1}/")
    assert one_totals["slices"] == 1 and one[0]["slice_id"] == rid_day1

    two, two_totals = listed(store, prefix=f"{SNIPPET}/{day2}/")
    assert two_totals["slices"] == 1 and two[0]["slice_id"] == rid_day2


def test_a_slice_past_the_horizon_is_neither_loaded_nor_new():
    """The store keeps an expired object until R2's own lifecycle pass takes it, and no load will ever fetch it — so a listing that called it new would send its reader after something that is never coming."""
    day = 86_400_000
    old, recent = f"0{T0 - 60 * day}-aaaa", f"0{T0}-bbbb"
    shards = [(f"{SNIPPET}/{slice_date(i)}/", _key(i)) for i in (old, recent)]
    cutoff = slice_date(f"0{T0 - 30 * day}-cccc")

    listing = Inventory(_Listing(shards), {}, cutoff=cutoff)
    rows = {row["slice_id"]: row for row in listing}

    assert rows[old]["expired"] and not rows[old]["loaded"]
    assert not rows[recent]["expired"]
    assert listing.totals["slices"] == 2
    assert listing.totals["new"] == 1, "only the recording still inside the horizon"
    assert listing.totals["expired"] == 1
