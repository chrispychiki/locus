"""The URL rule and the value-escape rule are each stated once on each side of the language boundary (distill/text.js for the projection, locus/text.py for the event stream and context blocks); this contract test runs both over the same cases so the twins cannot drift apart silently."""

import json
import subprocess
from pathlib import Path

from locus.evidence.text import URL_CAP, describe_url, escape_value

TEXT_JS = Path(__file__).parent.parent / "distill" / "text.js"

CASES = [
    "https://shop.test/cart",
    "https://shop.test/product?page=2",
    "https://shop.test/results?q=" + "flannel+overshirt+" * 20,
    "https://shop.test/product?" + "x" * 300,
    "https://shop.test/product?" + "x" * 3000,
    "https://school.test/?_gl=1*" + "x" * 3000 + "#/recording/42",
    "https://school.test/?_gl=1*" + "x" * 3000 + "#/" + "f" * 3000,
    "https://shop.test/" + "p/" * 3000,
    "https://school.test/課程/練習?_gl=" + "x" * 3000 + "#/編輯",
    # Astral characters: two UTF-16 code units in JS, one code point in Python — the case where
    # the twins' caps diverge unless both measure in the same unit.
    "https://shop.test/" + "\U0001f600" * 1500 + "/end",
    "https://shop.test/product?" + "\U0001f600" * 1500,
    "https://shop.test/" + "p" * (URL_CAP - 2) + "\U0001f600" * 4,
]


def test_the_python_twin_matches_the_js_rule_case_for_case():
    script = (
        f"import {{ describeUrl }} from {json.dumps(str(TEXT_JS))};"
        f"const cases = {json.dumps(CASES)};"
        f"console.log(JSON.stringify(cases.map(describeUrl)));"
    )
    out = subprocess.run(
        ["bun", "-e", script], check=True, capture_output=True, text=True
    ).stdout
    assert [describe_url(c) for c in CASES] == json.loads(out.strip())


def test_the_rule_itself():
    assert describe_url(None) is None
    searchy = "https://shop.test/results?q=" + "flannel+overshirt+" * 20
    assert describe_url(searchy) == searchy, (
        "a query under the ceiling is content — search terms, filter state"
    )
    long_query = "https://shop.test/product?" + "x" * 3000
    assert describe_url(long_query) == "https://shop.test/product?… (3kB query)"
    routed = "https://school.test/?_gl=1*" + "x" * 3000 + "#/recording/42"
    assert describe_url(routed) == "https://school.test/?… (3kB query)#/recording/42", (
        "a hash-routed SPA's route survives the cut"
    )
    pathy = "https://shop.test/" + "p/" * 3000
    assert len(describe_url(pathy)) == URL_CAP
    assert describe_url(pathy).endswith("…")


ESCAPE_CASES = [
    "plain",
    "two\nlines\r\n\tindented",
    "a\\b and a literal \\n",
    "quotes \"inside\" and 'apostrophes'",
    "\U0001f600\n",
]


def test_the_value_escape_twins_match_case_for_case():
    script = (
        f"import {{ escapeValue }} from {json.dumps(str(TEXT_JS))};"
        f"const cases = {json.dumps(ESCAPE_CASES)};"
        f"console.log(JSON.stringify(cases.map(escapeValue)));"
    )
    out = subprocess.run(
        ["bun", "-e", script], check=True, capture_output=True, text=True
    ).stdout
    assert [escape_value(c) for c in ESCAPE_CASES] == json.loads(out.strip())
