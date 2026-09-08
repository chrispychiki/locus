"""The meta channel's one splitting rule."""

from locus.analysis.prompts import META_HEADER, split_meta


def test_split_meta_separates_channel_from_answer():
    answer, meta = split_meta(
        "The visitor arrived [00:01.000] and left.\n\n"
        f"{META_HEADER}\n- The exit reason is not knowable from a recording.\n"
    )
    assert answer == "The visitor arrived [00:01.000] and left."
    assert meta == "- The exit reason is not knowable from a recording."


def test_split_meta_without_channel_returns_whole_answer():
    text = "Just an answer [00:01.000]."
    assert split_meta(text) == (text, None)


def test_split_meta_ignores_a_drifted_header():
    """Text under a header the contract did not sanction stays in the answer —
    it gets judged with it rather than silently escaping measurement."""
    drifted = "Answer [00:01.000].\n\n## Ambiguities\n- something\n"
    answer, meta = split_meta(drifted)
    assert meta is None
    assert "## Ambiguities" in answer


def test_split_meta_channel_only_reply_has_empty_answer():
    answer, meta = split_meta(f"{META_HEADER}\n- only complaints\n")
    assert answer == ""
    assert meta == "- only complaints"


def test_split_meta_empty_channel_is_none():
    answer, meta = split_meta(f"Answer [00:01.000].\n{META_HEADER}\n   \n")
    assert answer == "Answer [00:01.000]."
    assert meta is None
