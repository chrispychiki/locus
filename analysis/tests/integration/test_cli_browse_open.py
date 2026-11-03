import json
from pathlib import Path

import pytest
from _cli_support import (
    data_home,
    loaded_fixture_db,
    local_default_card,  # noqa: F401 — autouse fixture, registered by import
    offline_count,  # noqa: F401 — autouse fixture, registered by import
    shipped_definition,  # noqa: F401 — autouse fixture, registered by import
)


def test_browse_open_materializes_beside_the_db_at_derived_names(
    tmp_path, capsys, monkeypatch
):
    from locus.analysis.cli import _resolve_open

    _db, _conn, rows = loaded_fixture_db(tmp_path, monkeypatch)
    rid = rows[0]["recorder_slice"]

    argv = _resolve_open(["open", rid])
    capsys.readouterr()
    page = Path(argv[1])
    assert page.parent == data_home(tmp_path) / "pages", (
        "replay material is a derived family beside the db, never inside a wrapper dir"
    )
    assert page.name == f"{rows[0]['visitor_id']}_{rid}.html"
    payload = page.with_suffix(".js")
    assert payload.exists()
    assert f'<script src="./{payload.name}">' in page.read_text(), (
        "the default page embeds its own set's payload by relative src"
    )

    # Deterministic derivation, not evidence: re-opening refreshes in place.
    before = payload.read_text()
    assert _resolve_open(["open", rid])[1] == str(page)
    capsys.readouterr()
    assert payload.read_text() == before

    # Two sets sharing a first slice and a count are different payloads; the digest
    # keeps one from silently replacing the other under a page that embeds it.
    first_pair = _resolve_open(["open", rid, rows[1]["recorder_slice"]])[1]
    second_pair = _resolve_open(["open", rid, rows[2]["recorder_slice"]])[1]
    capsys.readouterr()
    assert first_pair != second_pair
    assert Path(first_pair).exists() and Path(second_pair).exists()


def test_browse_open_takes_a_slice_and_the_moment_to_see_it_at(
    tmp_path, capsys, monkeypatch
):
    # "This slice at this moment" is one target: the moment rides the slice name and lands
    # on the page composed from it, so opening a citation never costs a first open to learn
    # the composed page's derived name.
    from locus.analysis.cli import _resolve_open

    _db, _conn, rows = loaded_fixture_db(tmp_path, monkeypatch)
    rid = rows[0]["recorder_slice"]
    page = _resolve_open(["open", rid])[1]
    capsys.readouterr()

    assert _resolve_open(["open", f"{rid}#t=123"])[1] == f"{page}#t=123"
    capsys.readouterr()

    # A set composes one page, so one moment addresses it, riding whichever name carries it.
    pair = _resolve_open(["open", rid, f"{rows[1]['recorder_slice']}#t=456"])
    capsys.readouterr()
    assert pair[1].endswith("#t=456")
    with pytest.raises(SystemExit, match="one moment"):
        _resolve_open(["open", f"{rid}#t=1", f"{rows[1]['recorder_slice']}#t=2"])


def test_browse_open_names_every_form_when_the_target_is_none_of_them(
    tmp_path, capsys, monkeypatch
):
    from locus.analysis.cli import _resolve_open

    _db, _conn, _rows = loaded_fixture_db(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="URL.*analysis directory.*slice set"):
        _resolve_open(["open", "no-such-thing"])


def test_browse_open_composes_an_analysis_directory_from_its_slice_table(
    tmp_path, capsys, monkeypatch
):
    # The seam: window.json's slice table (as the engine writes it) drives
    # composition. Consecutive same-visitor slices that don't overlap share
    # one mount and play through as one stream, labeled by their range; a time
    # overlap is a concurrent lane and starts its own mount.
    from locus.analysis.cli import _resolve_open

    _db, _conn, rows = loaded_fixture_db(tmp_path, monkeypatch)
    visitor = rows[0]["visitor_id"]

    def seg(label, row, start, end):
        # opening consumes label/visitor/slice; slice_id belongs to the writer
        return {
            "label": label,
            "visitor": visitor,
            "slice": row["recorder_slice"],
            "slice_id": 0,
            "start_ts": start,
            "end_ts": end,
        }

    analysis_dir = data_home(tmp_path) / "analyses" / "2026-07-18T00-00-00Z_demo-0001"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "window.json").write_text(
        json.dumps({"slices": [seg("S1", rows[0], 0, 10), seg("S2", rows[1], 20, 30)]})
    )

    argv = _resolve_open(["open", f"{analysis_dir}#t=1781254694465"])
    narration = capsys.readouterr().out
    page_ref, _, fragment = argv[1].partition("#")
    page = Path(page_ref)
    assert fragment == "t=1781254694465", "the citation fragment rides through"
    assert page.name == f"{analysis_dir.name}.html"
    body = page.read_text()
    assert body.count("<section") == 1, (
        "a sequential same-visitor window is one stream on one player"
    )
    assert 'data-label="S1–S2"' in body
    assert narration.count("materialized") >= 1 and str(page) in narration

    concurrent = data_home(tmp_path) / "analyses" / "2026-07-18T00-00-01Z_demo-0002"
    concurrent.mkdir()
    (concurrent / "window.json").write_text(
        json.dumps({"slices": [seg("S1", rows[0], 0, 10), seg("S2", rows[1], 5, 15)]})
    )
    argv = _resolve_open(["open", str(concurrent)])
    capsys.readouterr()
    body = Path(argv[1].partition("#")[0]).read_text()
    assert body.count("<section") == 2, "a concurrent lane gets its own mount"
    assert 'data-label="S1"' in body and 'data-label="S2"' in body

    empty = data_home(tmp_path) / "analyses" / "not-an-analysis"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no window.json"):
        _resolve_open(["open", str(empty)])
