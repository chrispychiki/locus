"""The asset resolver's contract: fail loud on ambiguity, so a version bump that swaps files in vendor/ can never silently resolve to the wrong one — and fail loud when the assets aren't reachable at all, rather than shelling out to a path that doesn't exist."""

import pytest
from locus.evidence import assets


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "distill").mkdir()
    monkeypatch.setattr(assets, "_evidence_root", lambda: tmp_path)
    return tmp_path


def test_resolves_a_unique_match(fake_root):
    (fake_root / "vendor" / "rrweb-replay-1.2.3.min.js").write_text("x")
    assert assets.vendored("rrweb-replay-*.min.js").name == "rrweb-replay-1.2.3.min.js"


def test_a_missing_asset_is_not_found(fake_root):
    with pytest.raises(FileNotFoundError, match="no 'rrweb-replay"):
        assets.vendored("rrweb-replay-*.min.js")


def test_an_ambiguous_vendor_dir_is_a_different_failure_and_names_the_rivals(
    fake_root,
):
    # Two versions vendored at once is not a missing file — the asset is there twice, and a
    # resolver that called it "not found" would send the reader hunting for what is in front
    # of them. Naming both is what turns it into one deletion.
    (fake_root / "vendor" / "rrweb-replay-1.2.3.min.js").write_text("x")
    (fake_root / "vendor" / "rrweb-replay-1.2.4.min.js").write_text("y")
    with pytest.raises(ValueError, match="1.2.3.*1.2.4"):
        assets.vendored("rrweb-replay-*.min.js")


def test_a_missing_script_is_named(fake_root):
    (fake_root / "distill" / "distill.js").write_text("x")
    assert assets.script("distill.js").exists()
    with pytest.raises(FileNotFoundError, match="rescue_gate.js"):
        assets.script("rescue_gate.js")


def test_an_install_that_cannot_see_the_assets_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "__file__", str(tmp_path / "a" / "b" / "assets.py"))
    with pytest.raises(FileNotFoundError, match="cannot see its non-Python assets"):
        assets.vendored("rrweb-replay-*.min.js")


def test_assets_beside_the_package_resolve(tmp_path, monkeypatch):
    # A wheel carries distill/ and vendor/ inside the package; the editable clone leaves them
    # at the evidence root. One search rule has to find both, or a shipped wheel is a lie.
    package = tmp_path / "site-packages" / "locus" / "evidence"
    (package / "distill").mkdir(parents=True)
    (package / "vendor").mkdir()
    (package / "distill" / "distill.js").write_text("x")
    monkeypatch.setattr(assets, "__file__", str(package / "assets.py"))
    assert assets.script("distill.js") == package / "distill" / "distill.js"


def test_the_real_clone_resolves():
    assert assets.script("distill.js").exists()
    assert assets.script("rescue_gate.js").exists()
    assert assets.vendored("rrweb-player-*.min.js").exists()
