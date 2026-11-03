import ast
from pathlib import Path

from locus.analysis import engine
from locus.analysis.prompts import (
    request_screenshots_tool,
)


def test_request_screenshots_tool_has_one_shape_for_every_window():
    tool = request_screenshots_tool()
    items = tool["parameters"]["properties"]["screenshots"]["items"]
    assert items["required"] == ["timestamp"], (
        "a moment is an offset; the label rides only where the window has several slices"
    )
    assert "label" in items["properties"] and "enum" not in items["properties"]["label"]


def _imported_modules(path):
    tree = ast.parse(Path(path).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
    return mods


def test_policy_measurement_boundary_holds():
    from locus.analysis import oracle

    assert not any("oracle" in m for m in _imported_modules(engine.__file__)), (
        "the engine (policy) must not import the oracle (measurement)"
    )
    assert not any(m.endswith("engine") for m in _imported_modules(oracle.__file__)), (
        "the oracle (measurement) must not import the engine (policy)"
    )


def _locus_import_closure(module: str) -> set[str]:
    """The locus modules a fresh interpreter loads for `import <module>` — the whole transitive closure, which a file-level import scan cannot see."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys, {module}; "
                "print('\\n'.join(m for m in sys.modules if m.startswith('locus')))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def test_the_boundary_holds_across_the_whole_import_closure():
    """An indirect route — the engine importing a module that itself imports the oracle, or the grounding internals the oracle judges with — reads the same forbidden material through a middleman. The import closure is the fact the file-level scan approximates."""
    engine_side = _locus_import_closure("locus.analysis.engine")
    assert "locus.analysis.oracle" not in engine_side, (
        "loading the engine loaded the oracle — the policy can read its own objective"
    )
    assert not any(m.startswith("locus.analysis.ground") for m in engine_side), (
        "loading the engine loaded the grounding internals — the policy can read "
        "the rubric it is measured by"
    )
    oracle_side = _locus_import_closure("locus.analysis.oracle")
    assert "locus.analysis.engine" not in oracle_side, (
        "loading the oracle loaded the engine — the measurement depends on the policy"
    )


def _table(*rows):
    return [{"label": label, "start_ts": lo, "end_ts": hi} for label, lo, hi in rows]


def test_citation_violations_is_the_format_made_mechanical():
    """The validator holds every citation-shaped span to the citation format, the window's labels, and its bounds, and nothing else — a valid citation of a wrong moment is judgment, not its business."""
    from locus.analysis.window import citation_violations

    single = _table(("S1", 1_000_000, 1_600_000))  # a 10-minute recording

    assert citation_violations("plain prose, no brackets", single, 1_000_000) == []
    assert (
        citation_violations(
            "ok [00:10.500] and a range [00:10.500, 05:00.000]", single, 1_000_000
        )
        == []
    )
    assert citation_violations("aspect ratio [3:2] is prose", single, 1_000_000) == []

    (malformed,) = citation_violations("bad [00:99.123]", single, 1_000_000)
    assert malformed.startswith("[00:99.123]") and "[MM:SS.mmm]" in malformed
    (missing_millis,) = citation_violations("bad [00:10]", single, 1_000_000)
    assert missing_millis.startswith("[00:10]") and "[MM:SS.mmm]" in missing_millis

    (past_end,) = citation_violations("late [59:59.999]", single, 1_000_000)
    assert past_end.startswith("[59:59.999]") and "[10:00.000]" in past_end
    (bad_range,) = citation_violations("swap [00:20.000, 00:10.000]", single, 1_000_000)
    assert (
        bad_range.startswith("[00:20.000, 00:10.000]")
        and "[10:00.000]" not in bad_range
    )
    (labeled,) = citation_violations("label [S1 00:10.000]", single, 1_000_000)
    assert labeled.startswith("[S1 00:10.000]")

    (repeated,) = citation_violations(
        "twice [00:99.123] and again [00:99.123]", single, 1_000_000
    )
    assert repeated == malformed, "one defect however often the span repeats"

    multi = _table(("S1", 1_000_000, 1_300_000), ("S2", 1_200_000, 1_600_000))
    assert citation_violations("fine [S2 04:00.000]", multi, 1_000_000) == [], (
        "a moment inside the labeled slice passes"
    )
    (bare,) = citation_violations("bare [00:10.000]", multi, 1_000_000)
    assert bare.startswith("[00:10.000]") and "S1" in bare and "S2" in bare
    (unknown,) = citation_violations("ghost [S9 00:10.000]", multi, 1_000_000)
    assert unknown.startswith("[S9 00:10.000]") and "S1" in unknown and "S2" in unknown
    (crossed,) = citation_violations("crossed [S1 05:30.000]", multi, 1_000_000)
    assert crossed.startswith("[S1 05:30.000]") and "[05:00.000]" in crossed, (
        "a moment the window holds but the cited slice does not is a defect"
    )

    assert (
        citation_violations("both [S2 04:00.000, S2 04:30.000]", multi, 1_000_000) == []
    ), "a range naming its one slice twice is the form, redundantly"
    (split,) = citation_violations(
        "split [S1 01:00.000, S2 04:00.000]", multi, 1_000_000
    )
    assert split.startswith("[S1 01:00.000, S2 04:00.000]") and "two" in split


def test_a_citation_is_valid_as_far_out_as_the_payload_stamps_it():
    """The payload prints a moment's minutes unbounded — a comparative window's later slices sit days after its start — and the model cites the stamp they were shown, so the citation form admits what the payload emits."""
    from locus.analysis.window import citation_violations, format_offset

    late = 1_000_000 + 2079 * 60_000 + 37_620
    multi = _table(("S1", 1_000_000, 1_300_000), ("S3", late - 60_000, late + 60_000))
    stamp = format_offset(late, 1_000_000, "S3")
    assert stamp == "[S3 2079:37.620]"
    assert citation_violations(f"seen {stamp}", multi, 1_000_000) == []
    (early,) = citation_violations("off [S3 00:10.000]", multi, 1_000_000)
    assert early.startswith("[S3 00:10.000]") and "outside S3" in early
