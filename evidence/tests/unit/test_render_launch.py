"""The renderer's Chromium launches sandboxed or not at all."""

from pathlib import Path
from types import SimpleNamespace

import playwright
import pytest
from locus.evidence.render import SANDBOX_FAILED, launch_chromium
from playwright.sync_api import Error


def _playwright_failing_with(message):
    def launch(**options):
        assert options["chromium_sandbox"] is True
        raise Error(message)

    return SimpleNamespace(chromium=SimpleNamespace(launch=launch))


def test_a_machine_that_cannot_sandbox_chromium_is_told_what_the_sandbox_needs():
    with pytest.raises(RuntimeError, match="user namespaces") as refusal:
        launch_chromium(
            _playwright_failing_with(f"BrowserType.launch: {SANDBOX_FAILED}\n====")
        )
    assert isinstance(refusal.value.__cause__, Error)


def test_any_other_launch_failure_is_playwrights_own():
    with pytest.raises(Error, match="Executable doesn't exist"):
        launch_chromium(_playwright_failing_with("Executable doesn't exist at /x"))


def test_the_sandbox_failure_is_recognized_by_the_installed_playwrights_wording():
    bundle = Path(playwright.__file__).parent / "driver/package/lib/coreBundle.js"
    assert SANDBOX_FAILED in bundle.read_text()
