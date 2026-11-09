"""The evidence package's non-Python assets: the bun scripts under distill/ and the vendored rrweb player under vendor/.

They sit at the evidence root of the clone the package is installed from in editable mode. An install that cannot see both there fails loud rather than shelling out to a path that does not exist.
"""

from pathlib import Path


def _evidence_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not ((root / "distill").is_dir() and (root / "vendor").is_dir()):
        raise FileNotFoundError(
            f"locus cannot see its non-Python assets (distill/, vendor/) at {root}: "
            "locus-evidence runs from its clone, installed by ./install.sh from the deployment root"
        )
    return root


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
