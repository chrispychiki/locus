"""The URL rule's Python side (locus/text.py), case by case. integration/test_text.py holds the JS twin to the same cases."""

from locus.evidence.text import URL_CAP, describe_url


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
