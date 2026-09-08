"""The evidence package's non-Python assets: the bun scripts under distill/ and the vendored rrweb player under vendor/.

One rule finds them either way the package is installed: the nearest directory at or above this module holding both. A wheel carries them inside the package (pyproject's wheel `only-include` and its `sources` remap); an editable install from the clone leaves them at the evidence root. An install that can see neither fails loud rather than shelling out to a path that does not exist.
"""

from pathlib import Path


def _evidence_root() -> Path:
    module = Path(__file__).resolve()
    for candidate in (module.parent, *module.parents):
        if (candidate / "distill").is_dir() and (candidate / "vendor").is_dir():
            return candidate
    raise FileNotFoundError(
        f"locus cannot see its non-Python assets (distill/, vendor/) anywhere "
        f"at or above {module.parent} — reinstall locus-evidence"
    )


def script(name: str) -> Path:
    """A bun script the package shells out to."""
    path = _evidence_root() / "distill" / name
    if not path.exists():
        raise FileNotFoundError(f"no distillation script {name} at {path}")
    return path


def vendored(pattern: str) -> Path:
    """A vendored asset by glob — version bumps swap files in vendor/ without touching module code. Zero matches is a missing asset; several is an ambiguous vendor directory, which is a different failure and says so."""
    vendor = _evidence_root() / "vendor"
    matches = sorted(vendor.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern!r} in {vendor}")
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} files match {pattern!r} in {vendor} "
            f"({', '.join(m.name for m in matches)}) — vendor one version"
        )
    return matches[0]


def rrweb_constants_js() -> str:
    """The generated rrweb constants (distill/rrweb_constants.js, the one numeric↔name truth) as a browser-global script: `window.LocusRrweb.EventType`, `.IncrementalSource`, and the rest, for the scripts that run inside a replay page and cannot import a module."""
    source = script("rrweb_constants.js").read_text()
    return "window.LocusRrweb = {};\n" + source.replace("export const ", "LocusRrweb.")
