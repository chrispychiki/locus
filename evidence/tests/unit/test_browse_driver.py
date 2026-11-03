import pytest
from locus.evidence.browse_driver import chord, target_url


def test_chords_translate_to_playwright_names():
    assert chord("ctrl+Enter") == "Control+Enter"
    assert chord("cmd+a") == "Meta+a"
    assert chord("Enter") == "Enter"
    assert chord("shift+alt+ArrowDown") == "Shift+Alt+ArrowDown"


def test_target_url_passes_urls_and_resolves_files(tmp_path):
    assert target_url("https://example.test/a") == "https://example.test/a"
    page = tmp_path / "p.html"
    page.write_text("<html></html>")
    assert target_url(str(page)) == page.resolve().as_uri()
    with pytest.raises(FileNotFoundError):
        target_url("/no/such/page.html")


def test_target_url_keeps_the_fragment_a_citation_rides_in(tmp_path):
    # A replay page opens at a citation as page.html#t=<ms>; the fragment is not part of the
    # file path, so the existence check must strip it and the URI must keep it.
    page = tmp_path / "v.html"
    page.write_text("<html></html>")
    assert target_url(f"{page}#t=123-456") == page.resolve().as_uri() + "#t=123-456"
