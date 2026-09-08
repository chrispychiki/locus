import json
import re

import pytest
from _analysis_support import ScriptedConversation as FakeConversation
from locus.analysis.analyze import compose_window, event_stream, screenshot_index
from locus.analysis.select_screenshots import select_screenshots
from locus.analysis.window import format_offset, resolve_window
from locus.evidence.db import connect
from locus.evidence.hydrate import pack_raw


def test_offset_formatting():
    assert format_offset(61_234, 0) == "[01:01.234]"
    assert format_offset(5_000, 5_000) == "[00:00.000]"
    assert format_offset(4_551, 5_000) == "[00:00.000]"
    assert format_offset(61_234, 0, "S2") == "[S2 01:01.234]"


def test_citation_forms_follow_the_window_labels():
    from locus.analysis.window import citation_forms

    bare = citation_forms(["S1"])
    assert bare == {"cite": "[MM:SS.mmm]", "cite_range": "[MM:SS.mmm, MM:SS.mmm]"}
    labeled = citation_forms(["S1", "S2", "S3"])
    assert labeled["cite"] == "[S2 MM:SS.mmm]", (
        "the multi-slice forms are written with the window's real labels"
    )
    assert labeled["cite_range"] == "[S2 MM:SS.mmm, MM:SS.mmm]"


def test_resolve_citation_is_a_pure_table_lookup():
    from locus.analysis.window import resolve_citation

    table = [
        {
            "label": "S1",
            "visitor": "va",
            "slice": "s-a",
            "slice_id": 1,
            "start_ts": 1_000,
            "end_ts": 5_000,
        },
        {
            "label": "S2",
            "visitor": "vb",
            "slice": "s-b",
            "slice_id": 2,
            "start_ts": 2_000,
            "end_ts": 6_000,
        },
    ]
    hit = resolve_citation(
        table, 1_000, label="S2", start_offset_ms=1_500, end_offset_ms=2_500
    )
    assert hit["slice"]["slice"] == "s-b" and hit["slice"]["visitor"] == "vb"
    assert (hit["start_ts"], hit["end_ts"]) == (2_500, 3_500), (
        "offsets resolve to absolute epoch time on the one window clock"
    )

    with pytest.raises(ValueError, match="names its slice's label"):
        resolve_citation(table, 1_000, start_offset_ms=0)
    with pytest.raises(ValueError, match="unknown label"):
        resolve_citation(table, 1_000, label="S9", start_offset_ms=0)

    only = resolve_citation(table[:1], 1_000, start_offset_ms=500)
    assert only["slice"]["label"] == "S1" and only["start_ts"] == 1_500, (
        "a single-slice window resolves bare citations to its one slice"
    )
    assert only["end_ts"] is None


def test_distilled_kinds_are_names_never_numeric_codes(distilled_db):
    conn = connect(distilled_db)
    kinds = [
        r["type_str"] for r in conn.execute("SELECT DISTINCT type_str FROM events")
    ]
    assert kinds
    assert all(kind and not kind.isdigit() for kind in kinds)


def row(ts, kind, **kw):
    base = {
        "id": ts,
        "timestamp": ts,
        "type_str": kind,
        "url": None,
        "tag": None,
        "class": None,
        "text": None,
        "x": None,
        "y": None,
        "input": None,
        "hidden": None,
        "extra": None,
        "md": None,
        "diff": None,
        "raw_json": pack_raw("{}"),
        "device": None,
        "os": None,
        "browser": None,
    }
    base.update(kw)
    return base


def stream_text(rows):
    return [line for _, line in event_stream(rows, rows[0]["timestamp"])]


TRACE = re.compile(r"^\[[^\]]+\] (\d+) ")


def traces(lines):
    """The counted-trace lines: a stamp, then a count where an event line carries its kind."""
    return [l for l in lines if TRACE.match(l)]


def count_of(trace):
    return int(TRACE.match(trace).group(1))


def stamps(line):
    return re.findall(r"\[(?:S\d+ )?\d\d:\d\d\.\d{3}\]", line)


def test_press_events_collapse_into_their_click():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "MouseDown", tag="button", text="Save", x=10.4, y=20.9),
            row(200, "MouseUp", tag="button", text="Save", x=10.4, y=20.9),
            row(201, "Click", tag="button", text="Save", x=10, y=20),
            row(900, "MouseDown", tag="div", text="canvas", x=5, y=5),
        ]
    )
    kinds = [l.split(" ")[1] for l in lines]
    assert kinds == ["Meta", "Click", "MouseDown"]


def test_press_events_collapse_despite_within_click_pointer_drift():
    # The browser stamps the click at the release, so a press/click pair on one
    # element routinely differs by a few pixels — still one interaction.
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "MouseDown", tag="a", text="Categories", x=350, y=180),
            row(200, "MouseUp", tag="a", text="Categories", x=350, y=180),
            row(201, "Click", tag="a", text="Categories", x=350, y=179),
        ]
    )
    kinds = [l.split(" ")[1] for l in lines]
    assert kinds == ["Meta", "Click"]


def test_press_events_on_a_distant_lookalike_do_not_collapse():
    # Two identically-rendered elements sit farther apart than any within-click
    # drift; a press on one never folds into a click on the other.
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "MouseDown", tag="button", text="Save", x=10, y=20),
            row(200, "Click", tag="button", text="Save", x=10, y=220),
        ]
    )
    kinds = [l.split(" ")[1] for l in lines]
    assert kinds == ["Meta", "MouseDown", "Click"]


def test_projected_link_queries_collapse_and_arrival_urls_keep_theirs():
    arrival = "https://x.test/p?utm_source=google&gclid=abc123"
    md = (
        "[Zoom](https://x.test/p?utm_source=google&gclid=abc123#Image9)\n"
        "[Dresses](https://x.test/collections/dresses)\n"
        "![pic](https://x.test/cdn/a_1x1.gif?v=1784917277)"
    )
    lines = stream_text(
        [
            row(0, "Meta", url=arrival),
            row(100, "FullSnapshot", md=md),
            row(200, "PageLoad", url=arrival),
        ]
    )
    joined = "\n".join(lines)
    assert "[Zoom](https://x.test/p?…#Image9)" in joined, (
        "a link target keeps origin, path, and fragment; its query collapses to a residue"
    )
    assert "[Dresses](https://x.test/collections/dresses)" in joined
    assert "![pic](https://x.test/cdn/a_1x1.gif?…)" in joined
    assert joined.count("utm_source=google&gclid=abc123") == 2, (
        "the arrival url keeps its full query on the Meta and PageLoad lines"
    )


def test_consecutive_selections_keep_only_the_final_state():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(10, "Selection", text="The"),
            row(20, "Selection", text="The paper"),
            row(30, "Selection", text="The paper demonstrates"),
            row(500, "Scroll", x=0, y=10),
        ]
    )
    selections = [l for l in lines if "Selection" in l]
    assert len(selections) == 1
    assert "The paper demonstrates" in selections[0]


def test_rapid_same_element_input_churn_settles_to_the_last_value():
    import json as _json

    same = pack_raw(_json.dumps({"data": {"id": 7}}))
    other = pack_raw(_json.dumps({"data": {"id": 8}}))
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(10, "Input", tag="textarea", input="a", raw_json=same),
            row(40, "Input", tag="textarea", input="ab", raw_json=same),
            row(70, "Input", tag="textarea", input="abc", raw_json=same),
            row(80, "Input", tag="input", input="x", raw_json=other),
            row(2000, "Input", tag="textarea", input="abc, then more", raw_json=same),
        ]
    )
    inputs = [l for l in lines if "Input" in l]
    assert [l.split("input=")[1] for l in inputs] == [
        "'abc'",
        "'x'",
        "'abc, then more'",
    ], (
        "same-element churn settles; another element's value never absorbs it, "
        "and a keystroke sitting apart keeps its own line"
    )


def test_hidden_target_input_churn_never_reaches_the_event_stream():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(10, "Input", tag="input", input="fb.1.17847.821367", hidden=1),
            row(1500, "Input", tag="input", input="real typing", hidden=0),
            row(3000, "Input", tag="input", input="unjudged", hidden=None),
        ]
    )
    joined = "\n".join(lines)
    assert "fb.1.17847.821367" not in joined
    assert "real typing" in joined and "unjudged" in joined, (
        "only an affirmative hidden judgment drops — an unresolved target stays"
    )


def test_focus_folds_into_its_own_interaction_and_rides_everywhere_else():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(
                100, "MouseDown", tag="input", text="", x=10, y=20, **{"class": "email"}
            ),
            row(104, "Focus", tag="input", text="", **{"class": "email"}),
            row(200, "Click", tag="input", text="", x=10, y=20, **{"class": "email"}),
            row(5000, "Focus", tag="input", **{"class": "name"}),
            row(9000, "Blur", tag="input", **{"class": "name"}),
        ]
    )
    kinds = [l.split(" ")[1] for l in lines]
    assert kinds == ["Meta", "Click", "Focus", "Blur"], (
        "the pointer-caused Focus folds into its interaction; keyboard focus "
        "and a blur to nowhere are behavior and ride"
    )
    assert "name" in lines[2] and "name" in lines[3]


def test_a_label_click_focusing_its_field_keeps_the_focus_line():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Click", tag="label", text="Email", x=10, y=20),
            row(104, "Focus", tag="input", **{"class": "email"}),
        ]
    )
    assert any(l.split(" ")[1] == "Focus" for l in lines), (
        "a different-target Focus is not duplicate testimony — it names which field actually lit up"
    )


def test_a_blur_shadowing_a_click_elsewhere_folds():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Click", tag="button", text="Save", x=5, y=5),
            row(103, "Blur", tag="textarea", **{"class": "essay"}),
        ]
    )
    assert not any(l.split(" ")[1] == "Blur" for l in lines), (
        "leaving the old element is the click's mechanical shadow — the click "
        "line already testifies where engagement went"
    )


def test_a_blur_just_before_its_pointer_event_folds():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Blur", tag="input", **{"class": "email"}),
            row(102, "MouseDown", tag="button", text="Save", x=5, y=5),
            row(103, "Click", tag="button", text="Save", x=5, y=5),
        ]
    )
    assert not any(l.split(" ")[1] == "Blur" for l in lines), (
        "dispatch order between a blur and its pointer event is not guaranteed; "
        "either side of the window is the same shadow"
    )


def test_a_distant_blur_rides():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Click", tag="button", text="Save", x=5, y=5),
            row(3000, "Blur", tag="textarea", **{"class": "essay"}),
        ]
    )
    assert any(l.split(" ")[1] == "Blur" for l in lines), (
        "a blur seconds from any pointer event was not caused by one — a script steal or an abandoned field is behavior"
    )


def test_a_distant_focus_on_the_same_target_rides():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Click", tag="input", **{"class": "email"}, x=1, y=1),
            row(3000, "Focus", tag="input", **{"class": "email"}),
        ]
    )
    assert any(l.split(" ")[1] == "Focus" for l in lines), (
        "a Focus seconds after the click is its own event, not the click's echo"
    )


def test_invisible_churn_leaves_a_counted_trace():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(10, "Mutation", diff=None),
            row(20, "Input", tag="input", input="pixel.state.9911", hidden=1),
            row(30, "Mutation", diff=None),
            row(1000, "Scroll", x=0, y=10),
            row(2000, "Mutation", diff=None),
        ]
    )
    joined = "\n".join(lines)
    assert "pixel.state.9911" not in joined, "content never rides the trace"
    runs = traces(lines)
    assert len(runs) == 2, "one trace per unbroken run"
    assert count_of(runs[0]) == 3
    assert stamps(runs[0]) == ["[00:00.010]", "[00:00.030]"], (
        "a run spans first to last"
    )
    assert count_of(runs[1]) == 1
    assert stamps(runs[1]) == ["[00:02.000]"], "a run of one has no span"


def test_hidden_focus_churn_joins_the_trace_not_the_stream():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(10, "Focus", tag="input", hidden=1),
            row(20, "Blur", tag="input", hidden=1),
        ]
    )
    assert not any(l.split(" ")[1] in ("Focus", "Blur") for l in lines), (
        "script focus churn on undisplayed elements is machinery"
    )
    assert [count_of(t) for t in traces(lines)] == [2]


def test_viewport_dimensions_print():
    import json as _json

    lines = stream_text(
        [
            row(
                0,
                "Meta",
                url="https://x.test/",
                extra=_json.dumps({"width": 390, "height": 844}),
            ),
            row(
                500, "ViewportResize", extra=_json.dumps({"width": 844, "height": 390})
            ),
        ]
    )
    assert "390×844" in lines[0], "the viewport a context opened at"
    assert "844×390" in lines[1], "what the viewport became"


def test_a_cut_value_testifies_its_true_size():
    import json as _json

    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(
                10,
                "Input",
                tag="textarea",
                input="CHAPTER ONE…",
                extra=_json.dumps({"input_chars": 542103}),
                raw_json=pack_raw('{"data": {"id": 7}}'),
            ),
            row(
                2000,
                "Selection",
                text="The paper…",
                extra=_json.dumps({"text_chars": 42000}),
            ),
        ]
    )
    assert "(first 11 of 542103 chars)" in lines[1]
    assert "(first 9 of 42000 chars)" in lines[2]


def test_bodies_shed_blank_and_bare_sign_lines():
    from locus.analysis.analyze import stream_body

    assert stream_body("a\n\n+\n- \n+ b\n- c") == "a\n+ b\n- c"


def test_timer_ticks_collapse_to_first_and_last():
    rows = [row(0, "Meta", url="https://x.test/")]
    for i in range(5):
        rows.append(
            row(
                1000 + i * 1000,
                "Mutation",
                diff=f"@@ -3,1 +3,1 @@\n- You're {34 - i} in line\n+ You're {33 - i} in line",
            )
        )
    lines = stream_text(rows)
    mutations = [l for l in lines if "Mutation" in l]
    assert len(mutations) == 1
    assert "5×" in mutations[0]
    assert "You're 34 in line" in mutations[0]
    assert "You're 29 in line" in mutations[0]
    assert not any(f"You're {n} in line" in mutations[0] for n in range(30, 34)), (
        "the values between the first and the last do not ride"
    )


def test_timer_ticks_collapse_across_interleaved_events():
    rows = [row(0, "Meta", url="https://x.test/")]
    for i in range(4):
        rows.append(
            row(
                1000 + i * 1000,
                "Mutation",
                diff=f"@@ -4,1 +4,1 @@\n-     0:{28 - i:02d}\n+     0:{27 - i:02d}",
            )
        )
        rows.append(row(1500 + i * 1000, "MouseMove", x=i, y=i))
    lines = stream_text(rows)
    mutations = [l for l in lines if "Mutation" in l]
    assert len(mutations) == 1
    assert "4×" in mutations[0]
    moves = [l for l in lines if "MouseMove" in l]
    assert len(moves) == 4
    assert lines.index(mutations[0]) < lines.index(moves[0])


def test_unrelated_single_flips_stay_literal():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(1000, "Mutation", diff="@@ -1,1 +1,1 @@\n- Save\n+ Saving…"),
            row(9000, "Mutation", diff="@@ -8,1 +8,1 @@\n- cart (0)\n+ cart (1)"),
        ]
    )
    mutations = [l for l in lines if "Mutation" in l]
    assert len(mutations) == 2
    assert "Saving…" in mutations[0] and "cart (1)" in mutations[1]
    assert not any("×" in m for m in mutations), (
        "two unrelated flips never fold into one"
    )


def test_large_removals_summarize_and_readds_become_references():
    block = "\n".join(f"+ product row {i}" for i in range(12))
    removal = "\n".join(f"- product row {i}" for i in range(12))
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(1000, "Mutation", diff=f"@@ -1,0 +1,12 @@\n{block}"),
            row(2000, "Mutation", diff=f"@@ -1,12 +1,0 @@\n{removal}"),
            row(3000, "Mutation", diff=f"@@ -1,0 +1,12 @@\n{block}"),
        ]
    )
    first, second, third = [l for l in lines if "Mutation" in l]
    assert "product row 0" in first and "product row 11" in first
    assert (
        "12 lines" in second
        and "product row 0" in second
        and "product row 11" in second
    )
    assert "product row 5" not in second, "a removed run is its ends, never its body"
    assert "12 lines" in third and "[00:01.000]" in third
    assert "product row" not in third, (
        "a run seen before is the moment it first appeared"
    )


def test_removed_lines_print_as_references_where_shorter():
    row_text = "| 3 | [teacher](https://x.test/?_gl=abc#/t/5) | Celine | " + " | ".join(
        f"[course {i}](https://x.test/?_gl=abc#/c/{i}) | 8 | 735" for i in range(40)
    )
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(1000, "Mutation", diff=f"@@ -1,0 +1,1 @@\n+ {row_text}"),
            row(2000, "Mutation", diff=f"@@ -1,1 +1,0 @@\n- {row_text}\n- Loading…"),
        ]
    )
    added, removed = [l for l in lines if "Mutation" in l]
    assert row_text.replace("?_gl=abc#", "?…#") in added
    collapsed = row_text.replace("?_gl=abc#", "?…#")
    assert (
        f"- [{len(collapsed)} chars: {collapsed[:60]!r} … {collapsed[-60:]!r}]"
        in removed
    )
    assert "course 20" not in removed
    assert "- Loading…" in removed


def test_mostly_seen_blocks_emit_only_their_novel_lines():
    block = "\n".join(f"+ product row {i}" for i in range(12))
    grown = block + "\n+ product row NEW"
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(1000, "Mutation", diff=f"@@ -1,0 +1,12 @@\n{block}"),
            row(2000, "Mutation", diff=f"@@ -1,0 +1,13 @@\n{grown}"),
        ]
    )
    second = [l for l in lines if "Mutation" in l][1]
    assert "13 lines" in second and "product row 0" in second
    assert "+ product row NEW" in second
    assert "product row 5" not in second


def test_stream_sorts_by_canonical_timestamps():
    move = row(500, "MouseMove", x=1, y=2)
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(1000, "Click", tag="a", text="go", x=1, y=1),
            move,
        ]
    )
    assert [l.split(" ")[1] for l in lines] == ["Meta", "MouseMove", "Click"]


def test_scroll_target_text_prints_once_per_run():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Scroll", tag="div", text="The paper demonstrates", x=0, y=1),
            row(200, "Scroll", tag="div", text="The paper demonstrates", x=0, y=50),
            row(300, "Scroll", tag="div", text="The paper demonstrates", x=0, y=90),
        ]
    )
    scrolls = [l for l in lines if "Scroll" in l]
    assert len(scrolls) == 3
    assert sum("The paper demonstrates" in l for l in scrolls) == 1
    assert all("to offset (0," in l for l in scrolls)
    assert not any("@(" in l for l in scrolls)


def test_touchstart_target_text_prints_once_per_same_target_run():
    blob = "The Poet EmpressA Library Reads pick!"
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "TouchStart", tag="section", text=blob, x=1, y=1),
            row(200, "Scroll", tag="div", text="other", x=0, y=50),
            row(300, "TouchStart", tag="section", text=blob, x=2, y=2),
            row(400, "TouchStart", tag="p", text="a different target", x=3, y=3),
            row(500, "TouchStart", tag="section", text=blob, x=4, y=4),
            row(
                600,
                "Mutation",
                diff="@@ -1,2 +1,2 @@\n- old\n- lines\n+ new\n+ content",
            ),
            row(700, "TouchStart", tag="section", text=blob, x=5, y=5),
        ]
    )
    touches = [l for l in lines if "TouchStart" in l]
    assert len(touches) == 5
    assert sum(blob in l for l in touches) == 3
    assert blob in touches[0]
    assert blob not in touches[1]
    assert "a different target" in touches[2]
    assert blob in touches[3]
    assert blob in touches[4]
    assert all("@(" in l for l in touches)


def test_a_textless_target_is_identified_by_what_names_it():
    # An icon button, an image, an empty field: the tag and class alone say nothing about
    # what the visitor touched, and the model's only recourse is buying a screenshot to look.
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(
                100,
                "Click",
                tag="button",
                x=1,
                y=1,
                extra=json.dumps({"label": "Close menu", "name": "close"}),
            ),
            row(
                200,
                "Input",
                tag="textarea",
                input="an essay",
                extra=json.dumps({"label": "Paste your text here"}),
            ),
            row(
                300,
                "Click",
                tag="a",
                text="Read more",
                x=2,
                y=2,
                extra=json.dumps({"label": "Read more about the collection"}),
            ),
        ]
    )
    assert 'label="Close menu"' in lines[1], (
        "the stamped name is printed as a name — not as text that was on screen"
    )
    assert '"close"' not in lines[1], "only the stamped name identifies the element"
    assert 'label="Paste your text here"' in lines[2], (
        "a typed value says nothing about which field took it"
    )
    assert "label=" not in lines[3] and '"Read more"' in lines[3], (
        "visible text is the better identifier wherever it exists"
    )


def test_typing_names_its_field_once_per_run():
    editor = json.dumps({"label": "Enter the sentence"})
    same = pack_raw(json.dumps({"data": {"id": 7}}))
    other = pack_raw(json.dumps({"data": {"id": 8}}))
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Input", tag="textarea", input="H", extra=editor, raw_json=same),
            row(600, "Input", tag="textarea", input="He", extra=editor, raw_json=same),
            row(
                1100, "Input", tag="textarea", input="Hel", extra=editor, raw_json=same
            ),
            row(
                1600,
                "Mutation",
                diff="@@ -1,2 +1,2 @@\n- old\n- lines\n+ new\n+ content",
            ),
            row(
                2100, "Input", tag="textarea", input="Hell", extra=editor, raw_json=same
            ),
            row(
                2600,
                "Input",
                tag="input",
                input="x",
                extra=json.dumps({"label": "Title"}),
                raw_json=other,
            ),
        ]
    )
    inputs = [l for l in lines if " Input " in l]
    assert len(inputs) == 5, "every keystroke keeps its own line and its own value"
    assert sum("Enter the sentence" in l for l in inputs) == 1, (
        "the field is named once — the content change between keystrokes did not change it"
    )
    assert "Enter the sentence" in inputs[0]
    assert "Title" in inputs[4], "a different field names itself"


def test_a_typing_run_states_its_ends_whole_and_only_what_changed_between():
    field = json.dumps({"label": "朗讀文本"})
    same = pack_raw(json.dumps({"data": {"id": 7}}))
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(
                100,
                "Input",
                tag="textarea",
                input="kind、 violin",
                extra=field,
                raw_json=same,
            ),
            row(
                600,
                "Input",
                tag="textarea",
                input="kind violin",
                extra=field,
                raw_json=same,
            ),
            row(
                1100,
                "Input",
                tag="textarea",
                input="kind\nviolin",
                extra=field,
                raw_json=same,
            ),
            row(
                1600,
                "Input",
                tag="textarea",
                input="kind\nviolin!",
                extra=field,
                raw_json=same,
            ),
        ]
    )
    inputs = [l for l in lines if " Input " in l]
    assert inputs[0].endswith("input='kind、 violin'"), (
        "the run opens on its whole value"
    )
    assert inputs[1].endswith("@4 '、'→''")
    assert inputs[2].endswith("@4 ' '→'\\n'")
    assert inputs[3].endswith("input='kind\\nviolin!'"), (
        "and closes on the finished text, so nothing has to be reconstructed"
    )


def test_distinct_fields_rendering_identically_never_chain_into_one_run():
    # A script populating a form writes several fields that all render to the same
    # element string. Each is its own field: no line may state one field's value as
    # an edit of another's, and each names its element.
    field = json.dumps({"label": None})
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(
                100,
                "Input",
                tag="input",
                **{"class": "form-control"},
                input="DEMO",
                extra=field,
                raw_json=pack_raw(json.dumps({"data": {"id": 7}})),
            ),
            row(
                101,
                "Input",
                tag="input",
                **{"class": "form-control"},
                input="en-US",
                extra=field,
                raw_json=pack_raw(json.dumps({"data": {"id": 8}})),
            ),
            row(
                102,
                "Input",
                tag="input",
                **{"class": "form-control"},
                input="36000",
                extra=field,
                raw_json=pack_raw(json.dumps({"data": {"id": 9}})),
            ),
        ]
    )
    inputs = [l for l in lines if " Input " in l]
    assert len(inputs) == 3
    assert not any("→" in l for l in inputs), (
        "no field's value may be stated as an edit of another field's"
    )
    assert all("<input" in l and "input=" in l for l in inputs), (
        "each field states its whole value and names its element"
    )


def test_a_run_does_not_outlive_the_document_it_was_typed_in():
    # One component renders the same editor on two routes, so the field after a navigation
    # renders identically to the one before it — and starts empty. Carrying the run across
    # would state the whole previous page's text as deleted, which never happened.
    field = json.dumps({"label": "朗讀文本"})
    same = pack_raw(json.dumps({"data": {"id": 7}}))
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/a"),
            row(
                100,
                "Input",
                tag="textarea",
                input="kind violin",
                extra=field,
                raw_json=same,
            ),
            row(
                600,
                "Input",
                tag="textarea",
                input="kind violins",
                extra=field,
                raw_json=same,
            ),
            row(1000, "Meta", url="https://x.test/b"),
            row(1100, "FullSnapshot", md="a new page"),
            row(1200, "Input", tag="textarea", input="", extra=field, raw_json=same),
            row(1700, "Input", tag="textarea", input="x", extra=field, raw_json=same),
        ]
    )
    inputs = [l for l in lines if " Input " in l]
    assert not any("→''" in l for l in inputs), (
        "no line may state the previous page's text as deleted"
    )
    assert inputs[2].endswith("input=''"), (
        "the field on the new page opens its own run, whole"
    )
    assert 'label="朗讀文本"' in inputs[2], "and names itself again"


def test_a_value_that_did_not_change_says_only_what_can_be_shown():
    field = json.dumps({"label": "Notes"})
    same = pack_raw(json.dumps({"data": {"id": 7}}))
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Input", tag="textarea", input="abc", extra=field, raw_json=same),
            row(600, "Input", tag="textarea", input="abc", extra=field, raw_json=same),
            row(
                1100, "Input", tag="textarea", input="abcd", extra=field, raw_json=same
            ),
        ]
    )
    assert len([l for l in lines if " Input " in l]) == 2, (
        "a re-set to the value already there changed nothing the recording holds"
    )

    capped = "x" * 60 + "…"
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "Input", tag="textarea", input=capped, extra=field, raw_json=same),
            row(600, "Input", tag="textarea", input=capped, extra=field, raw_json=same),
            row(
                1100, "Input", tag="textarea", input=capped, extra=field, raw_json=same
            ),
        ]
    )
    inputs = [l for l in lines if " Input " in l]
    assert len(inputs) == 3, (
        "past the recorded prefix an edit is invisible, not absent — the line stays"
    )
    assert inputs[1].endswith(" …")


def test_arrivals_split_on_navigation_never_on_checkout():
    from locus.analysis.analyze import recording_facts

    facts = recording_facts(
        [
            row(0, "Meta"),
            row(10_000, "Click", device="mobile", os="iOS", browser="Safari"),
        ],
        slice_pages=[
            ("https://x.test/a", "page-load", 0),
            ("https://x.test/a", "unattested", 1_000),
            ("https://x.test/a", "page-load", 2_000),
            ("https://x.test/b", "page-load", 3_000),
        ],
    )
    pages = next(l for l in facts.splitlines() if l.startswith("Pages Visited"))
    assert re.search(r"\b2\b", pages) and re.search(r"\b3\b", pages), (
        "two unique pages, three arrivals: a checkout on /a is not an arrival"
    )
    a = next(l for l in facts.splitlines() if "https://x.test/a" in l)
    assert re.search(r"\b2\b", a), "/a was arrived at twice"
    assert not re.search(
        r"\d", next(l for l in facts.splitlines() if "https://x.test/b" in l)
    )
    assert "mobile" in facts and "iOS" in facts and "Safari" in facts


def test_compose_window_shape(distilled_db, tmp_path):
    conn = connect(distilled_db)
    ordered = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE status='replayable' ORDER BY start_ts"
        )
    ]
    biggest = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    at = ordered.index(biggest)
    slice_ids = (
        ordered[at : at + 2] if at + 1 < len(ordered) else ordered[at - 1 : at + 1]
    )

    selection = select_screenshots(conn, slice_ids[0])
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    for ts in selection.timestamps:
        (screenshot_dir / f"S1_screenshot_{ts}.png").write_bytes(b"\x89PNG fake")

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, slice_ids),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "An example site."},
    )

    kinds = [kind for kind, _ in conversation.parts]
    texts = [t for k, t in conversation.parts if k == "text"]
    assert any(t.startswith("<WEBSITE>") for t in texts)
    recording = next(t for t in texts if t.startswith("<SUMMARY>"))
    assert "2026-06-12T08:58:14Z" in recording and "2026-06-12T08:58:26Z" in recording
    assert "Visitor:" not in recording, (
        "one visitor's window needs no attribution labels"
    )
    slices_line = next(l for l in recording.splitlines() if l.startswith("Slices:"))
    assert "S1" in slices_line and "S2" in slices_line, (
        "a multi-slice group declares its labels and their bounds"
    )

    single = FakeConversation()
    single_manifest = compose_window(
        conn,
        single,
        resolve_window(conn, slice_ids[:1]),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "An example site."},
    )
    single_recording = next(
        t for k, t in single.parts if k == "text" and t.startswith("<SUMMARY>")
    )
    assert "Slices:" not in single_recording, (
        "a single-slice window carries no label apparatus"
    )
    assert [s.label for s in single_manifest.slices] == ["S1"]
    assert any(t.startswith("<SESSION_CONTEXT>") for t in texts)
    assert not any("<QUESTION>" in text for _, text in conversation.parts), (
        "the question is never composed into the window; it is sent last, "
        "on the turn that writes the answer, and never recorded"
    )
    assert kinds.count("image") == len(selection.timestamps) == manifest.n_screenshots

    stream_lines = "\n".join(
        t
        for t in texts
        if not t.startswith(("<WEBSITE", "<SUMMARY>", "<SESSION_CONTEXT>"))
    )
    assert "[S1 00:00.000]" in stream_lines, (
        "a multi-slice window stamps every stream line with its slice label"
    )
    assert "\n[00:" not in stream_lines, "no line is left bare-stamped"
    assert "FullSnapshot" in stream_lines

    single_stream = "\n".join(
        t
        for k, t in single.parts
        if k == "text"
        and not t.startswith(("<WEBSITE", "<SUMMARY>", "<SESSION_CONTEXT>"))
    )
    assert "[00:00.000]" in single_stream and "[S1 " not in single_stream, (
        "a single-slice window stamps bare offsets"
    )

    assert [s.label for s in manifest.slices] == ["S1", "S2"]
    assert [s.slice_id for s in manifest.slices] == slice_ids, (
        "labels run S1..Sn in time order"
    )
    placeholders = ",".join("?" * len(slice_ids))
    expected_events = conn.execute(
        f"SELECT SUM(n_events) total FROM slices WHERE id IN ({placeholders})",
        slice_ids,
    ).fetchone()["total"]
    assert manifest.n_events == expected_events
    first_event_ts = conn.execute(
        "SELECT MIN(timestamp) t FROM events WHERE slice_id = ?", (slice_ids[0],)
    ).fetchone()["t"]
    assert manifest.window_start_ts == first_event_ts


def test_the_screenshot_plane_opens_at_the_covering_snapshot(distilled_db, tmp_path):
    """A slice opens before rrweb has a DOM to capture, so its first instants hold no
    recorded page state — a screenshot there would be stamped with a time whose pixels were
    never recorded. The window's screenshot-addressable bounds open at the covering
    snapshot; the events ahead of it keep their place in the stream."""
    from locus.analysis.window import slice_table
    from locus.evidence.rrweb_constants import EventType

    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    snapshot_ts = conn.execute(
        "SELECT MIN(timestamp) t FROM events WHERE slice_id = ? AND type = ?",
        (slice_id, EventType.FullSnapshot),
    ).fetchone()["t"]

    row = slice_table(conn, [slice_id])[0]
    assert (
        row.snippet
        == conn.execute(
            "SELECT snippet FROM slices WHERE id = ?", (slice_id,)
        ).fetchone()["snippet"]
    ), "the table names the site each slice was recorded on"
    assert row.start_ts < snapshot_ts, (
        "this recording really does open before its snapshot"
    )
    assert row.screenshot_start_ts == snapshot_ts
    assert row.end_ts > snapshot_ts

    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    (screenshot_dir / f"S1_screenshot_{row.start_ts}.png").write_bytes(b"\x89PNG fake")
    with pytest.raises(ValueError, match="screenshot-addressable"):
        compose_window(
            conn,
            FakeConversation(),
            resolve_window(conn, [slice_id]),
            screenshot_index([screenshot_dir]),
            site_contexts={"site": "x"},
        )

    events_before = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE slice_id = ? AND timestamp < ?",
        (slice_id, snapshot_ts),
    ).fetchone()["n"]
    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [slice_id]),
        {},
        site_contexts={"site": "x"},
    )
    assert manifest.n_events > events_before > 0, (
        "the events ahead of the snapshot still ride the stream"
    )


def test_a_window_holding_only_a_slices_lead_is_refused(distilled_db):
    from locus.analysis.window import slice_table
    from locus.evidence.rrweb_constants import EventType

    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    snapshot_ts = conn.execute(
        "SELECT MIN(timestamp) t FROM events WHERE slice_id = ? AND type = ?",
        (slice_id, EventType.FullSnapshot),
    ).fetchone()["t"]
    with pytest.raises(ValueError, match="ends before slice"):
        slice_table(conn, [slice_id], None, snapshot_ts)


def test_every_screenshot_has_a_text_anchor_at_its_timestamp(distilled_db, tmp_path):
    # Every screenshot's moment is stated in the stream's own clock, on its own
    # Screenshot line — never inferred from adjacency to whatever event happens
    # to share its timestamp, and owed even at a moment whose event prints no
    # line of its own.
    conn = connect(distilled_db)
    suppressed = conn.execute(
        "SELECT e.slice_id s, e.timestamp t FROM events e "
        "JOIN slices ON slices.id = e.slice_id "
        "WHERE e.type_str = 'Focus' AND slices.status = 'replayable' "
        "LIMIT 1"
    ).fetchone()
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    (screenshot_dir / f"S1_screenshot_{suppressed['t']}.png").write_bytes(
        b"\x89PNG fake"
    )

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [suppressed["s"]]),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "x"},
    )

    image_at = next(
        i for i, (kind, _) in enumerate(conversation.parts) if kind == "image"
    )
    anchor = conversation.parts[image_at - 1][1].splitlines()[-1]
    offset = format_offset(suppressed["t"], manifest.window_start_ts)
    assert anchor == f"{offset} Screenshot"


def test_compose_prefetches_the_window_screenshots_before_adding_any(
    distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    selection = select_screenshots(conn, slice_id)
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    for ts in selection.timestamps:
        (screenshot_dir / f"S1_screenshot_{ts}.png").write_bytes(b"\x89PNG fake")

    class PrefetchSpy(FakeConversation):
        def __init__(self):
            super().__init__()
            self.prefetched = []

        def prefetch_images(self, image_paths):
            assert not self.images, "prefetch must precede every image add"
            self.prefetched.append(sorted(str(p) for p in image_paths))

    conversation = PrefetchSpy()
    compose_window(
        conn,
        conversation,
        resolve_window(conn, [slice_id]),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "x"},
    )

    assert conversation.prefetched == [
        sorted(
            str(screenshot_dir / f"S1_screenshot_{ts}.png")
            for ts in selection.timestamps
        )
    ]
    assert conversation.prefetched[0] == sorted(conversation.images), (
        "exactly the screenshots the window sends, no more"
    )


def test_window_bounds_split_a_slice_into_pieces(distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    bounds = conn.execute(
        "SELECT MIN(timestamp) lo, MAX(timestamp) hi, COUNT(*) n FROM events WHERE slice_id = ?",
        (slice_id,),
    ).fetchone()
    midpoint = (bounds["lo"] + bounds["hi"]) // 2

    first = FakeConversation()
    head = compose_window(
        conn,
        first,
        resolve_window(conn, [slice_id], None, midpoint),
        {},
        site_contexts={"site": "x"},
    )
    second = FakeConversation()
    tail = compose_window(
        conn,
        second,
        resolve_window(conn, [slice_id], midpoint, None),
        {},
        site_contexts={"site": "x"},
    )

    assert head.n_events + tail.n_events == bounds["n"]
    assert head.window_start_ts == bounds["lo"]
    assert tail.window_start_ts >= midpoint

    head_stream = "\n".join(t for k, t in first.parts if k == "text")
    tail_stream = "\n".join(t for k, t in second.parts if k == "text")
    assert "FullSnapshot" in head_stream
    assert "FullSnapshot" not in tail_stream
    assert "\n[00:00." in tail_stream

    with pytest.raises(ValueError, match="holds no events"):
        resolve_window(conn, [slice_id], None, bounds["lo"])


def test_undistilled_slice_fails_loud(tmp_path):
    from _support import FIXTURE, read_recording
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "raw.db")
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    slice_id = conn.execute("SELECT id FROM slices LIMIT 1").fetchone()["id"]
    with pytest.raises(ValueError, match="no distillation"):
        compose_window(
            conn,
            FakeConversation(),
            resolve_window(conn, [slice_id]),
            {},
            site_contexts={"site": "x"},
        )


def _overlap_db(tmp_path):
    """The same recording under two visitors — two slices occupying the very same instants, the concurrency case the window clock must carry honestly."""
    import subprocess

    from _support import FIXTURE, read_recording
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "overlap.db")
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    hydrate(conn, "concurrent-visitor", events)
    materialize_slices(conn, "concurrent-visitor")
    from _support import DISTILL as distill

    subprocess.run(
        ["bun", str(distill), str(tmp_path / "overlap.db")],
        check=True,
        capture_output=True,
    )
    first = conn.execute(
        "SELECT id FROM slices WHERE visitor_id = ? AND status='replayable' ORDER BY start_ts LIMIT 1",
        (visitor_id,),
    ).fetchone()["id"]
    second = conn.execute(
        "SELECT id FROM slices WHERE visitor_id = 'concurrent-visitor' "
        "AND status='replayable' ORDER BY start_ts LIMIT 1"
    ).fetchone()["id"]
    return conn, first, second


def test_overlapping_recordings_compose_on_one_shared_clock(tmp_path):
    """Two slices covering the same instants compose as two labeled slices
    whose offsets genuinely overlap — the concurrency lives in the numbers,
    never serialized apart."""
    conn, first, second = _overlap_db(tmp_path)

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [first, second]),
        {},
        site_contexts={"site": "x"},
    )

    assert [s.label for s in manifest.slices] == ["S1", "S2"]
    starts = [s.start_ts for s in manifest.slices]
    ends = [s.end_ts for s in manifest.slices]
    assert starts[1] <= ends[0], "the fixture's slices truly overlap"
    assert manifest.window_start_ts == min(starts)

    stream_lines = "\n".join(t for k, t in conversation.parts if k == "text")
    assert "[S1 00:00.000]" in stream_lines and "[S2 00:00.000]" in stream_lines, (
        "both slices open at the same window offset — concurrency is embodied in the numbers"
    )

    recordings = [
        t for k, t in conversation.parts if k == "text" and t.startswith("<SUMMARY>")
    ]
    assert len(recordings) == 2, "concurrent visitors stay delimited, attributed groups"


def test_a_multi_site_window_attributes_each_sites_frame(tmp_path):
    """A window spanning sites carries each site's own operator frame beside
    its slices — attributed WEBSITE blocks, and each SUMMARY naming its
    site — so no site's evidence reads under another site's frame."""
    conn, first, second = _overlap_db(tmp_path)
    conn.execute("UPDATE events SET snippet = 'site-a' WHERE slice_id = ?", (first,))
    conn.execute("UPDATE events SET snippet = 'site-b' WHERE slice_id = ?", (second,))
    conn.commit()

    conversation = FakeConversation()
    compose_window(
        conn,
        conversation,
        resolve_window(conn, [first, second]),
        {},
        site_contexts={"site-a": "Frame A.", "site-b": "Frame B."},
    )
    texts = [t for k, t in conversation.parts if k == "text"]
    websites = [t for t in texts if t.startswith("<WEBSITE")]
    assert websites == [
        '<WEBSITE site="site-a">\nFrame A.\n</WEBSITE>',
        '<WEBSITE site="site-b">\nFrame B.\n</WEBSITE>',
    ], "one attributed block per site, in the order the groups appear"
    recordings = [t for t in texts if t.startswith("<SUMMARY>")]
    assert "Site: site-a" in recordings[0] and "Site: site-b" in recordings[1]

    with pytest.raises(ValueError, match="site_contexts"):
        compose_window(
            conn,
            FakeConversation(),
            resolve_window(conn, [first, second]),
            {},
            site_contexts={"site-a": "Frame A.", "other": "x"},
        )


def test_a_group_window_attributes_each_visitors_slices(two_visitor_db):
    conn = connect(two_visitor_db)
    picks = conn.execute(
        "SELECT s.id, s.visitor_id, s.start_ts FROM slices s "
        "JOIN (SELECT visitor_id, MIN(start_ts) first_ts FROM slices "
        "      WHERE status='replayable' GROUP BY visitor_id) heads "
        "ON heads.visitor_id = s.visitor_id AND heads.first_ts = s.start_ts "
        "ORDER BY s.start_ts"
    ).fetchall()
    assert len(picks) == 2, "the fixture carries two visitors"

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [p["id"] for p in picks]),
        {},
        site_contexts={"site": "x"},
    )

    texts = [t for k, t in conversation.parts if k == "text"]
    recordings = [t for t in texts if t.startswith("<SUMMARY>")]
    assert len(recordings) == 2, (
        "each visitor's slices open with their own SUMMARY block"
    )
    for pick, block in zip(picks, recordings):
        assert f"Visitor: {pick['visitor_id']}" in block, (
            "the evidence itself attributes each slice to its visitor"
        )
    assert len([t for t in texts if t.startswith("<SESSION_CONTEXT>")]) == 2
    assert manifest.window_start_ts == min(p["start_ts"] for p in picks), (
        "one clock: offsets anchor at the earliest event across the group"
    )
    assert [(s.label, s.visitor) for s in manifest.slices] == [
        ("S1", picks[0]["visitor_id"]),
        ("S2", picks[1]["visitor_id"]),
    ], "the slice table names each label's visitor and slice"


def test_page_machinery_events_fold_into_the_invisible_trace():
    # A stylesheet adoption or a custom-element definition is the page's own machinery: nothing
    # the visitor did, nothing presented — counted, never printed as a line of its own.
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(10, "AdoptedStyleSheet", tag="shop-cart-sync"),
            row(20, "CustomElement"),
            row(30, "StyleSheetRule"),
            row(1000, "Scroll", x=0, y=10),
        ]
    )
    joined = "\n".join(lines)
    assert "AdoptedStyleSheet" not in joined and "CustomElement" not in joined
    assert [count_of(t) for t in traces(lines)] == [3]


def test_a_blank_removed_line_is_never_a_reference():
    # Inside a preformatted block a line of spaces is content the diff can remove; a reference
    # to nothing ("[0 chars: '' … '']") would say less than the line it replaces.
    long = "x" * 400
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "FullSnapshot", md=f"```\n{long}\n        \n```"),
            row(200, "Mutation", diff=f"- {long}\n-         \n+ done"),
        ]
    )
    joined = "\n".join(lines)
    assert "[0 chars" not in joined
    assert "[400 chars:" in joined


def test_a_field_value_the_reader_holds_is_referenced_wherever_it_recurs():
    # A pasted document is stated once — where it first appears — and every later statement of
    # the same value, on the page or on an Input, is its size and boundaries. The key is the
    # projection's written form, so a value with newlines matches its Input across the escape.
    doc = "\n".join(f"paragraph {i} of the pasted paper" for i in range(40))
    written = doc.replace("\n", "\\n")
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "FullSnapshot", md='[textarea placeholder="Write…"]'),
            row(200, "Input", tag="textarea", input=doc),
            row(
                300,
                "Mutation",
                diff=f'+ [textarea placeholder="Write…" value="{written}"]',
            ),
            row(400, "Input", tag="textarea", input="cleared"),
            row(500, "Input", tag="textarea", input=doc),
            row(
                600,
                "FullSnapshot",
                md=f'[textarea placeholder="Escribe…" value="{written}" label="Text"]',
            ),
        ]
    )
    joined = "\n".join(lines)
    assert joined.count("paragraph 7 of the pasted paper") == 1, (
        "the value is stated once"
    )
    n = len(written)
    assert lines[2].endswith(repr(doc)), "the first statement is whole"
    ref = f"[{n} chars: 'paragraph 0 of"
    assert f'+ [textarea placeholder="Write…" value={ref}' in joined
    assert f"Input <textarea> input={ref}" in joined
    assert f'[textarea placeholder="Escribe…" value={ref}' in joined
    assert '\'] label="Text"]' in joined, "what follows the value on its line survives"


def test_a_value_that_is_part_of_a_held_value_is_referenced():
    # A field trimmed at a cap holds the same text ending earlier; the reader already has it.
    doc = "\n".join(f"paragraph {i} of the pasted paper" for i in range(40))
    trimmed = doc[: len(doc) // 2]
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "FullSnapshot", md='[textarea placeholder="Write…"]'),
            row(200, "Input", tag="textarea", input=doc),
            row(300, "Click", tag="button", text="Trim"),
            row(400, "Input", tag="textarea", input=trimmed),
        ]
    )
    joined = "\n".join(lines)
    assert joined.count("paragraph 7 of the pasted paper") == 1
    n = len(trimmed.replace("\n", "\\n"))
    assert f"input=[{n} chars: 'paragraph 0 of" in lines[4]


def test_a_short_repeated_value_prints_whole():
    lines = stream_text(
        [
            row(0, "Meta", url="https://x.test/"),
            row(100, "FullSnapshot", md='[input type=text value="hello"]'),
            row(200, "Input", tag="input", input="hello"),
            row(300, "Mutation", diff='+ [input type=text value="hello"]'),
        ]
    )
    joined = "\n".join(lines)
    assert "chars:" not in joined
    assert joined.count('value="hello"') == 2
    assert "input='hello'" in joined
