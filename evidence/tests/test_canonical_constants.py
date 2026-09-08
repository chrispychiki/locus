"""Structural guard: rrweb numeric↔name knowledge lives ONLY in the generated canonical mapping (distill/extract_rrweb_constants.js and the files it emits).

Hand-typed rrweb schema knowledge is always misremembered — a numeric literal compared against an event type, or a name↔number pair declared anywhere else, is a defect even when it happens to be correct, because the next one won't be. This test sweeps every source file, hand-authored JSON fixtures included, for the shapes that crime takes:

  1. comparisons of a type/source field against a numeric literal — "field" meaning both the bare rrweb keys and camelCase compounds like pointerType or nodeType
  2. type/source keys (same field set) assigned numeric literals in object/dict/JSON literals
  3. any canonical enum member name re-declared with a number, or any number mapped to a canonical name, outside the generated files
  4. a canonical enum value and its own name written beside each other — the shape of a positional event tuple or argument list carrying both the number and the label
  5. two numeric literals in a row capped by a quoted canonical name — a positional event row whose type numeral rides between a timestamp and its label

The name and value lists are read from the generated module itself, so extending the generator automatically extends the guard. Shapes carrying no canonical name and no type/source key — a bare (4, 1000) tuple — sit below the structural floor and are the reviewer's to catch; the fix that keeps them out is constructing test events from the constants, never from literals.
"""

import re
import subprocess
from pathlib import Path

from locus.evidence import rrweb_constants

EVIDENCE = Path(__file__).resolve().parents[1]
REPO = EVIDENCE.parent
# `data` is the deployment's accreted-state directory: replay payloads are serialized
# recordings, the component copy carries the vendored player, and analysis directories hold
# model transcripts — none of it authored source.
EXCLUDED_PARTS = {
    "node_modules",
    "vendor",
    "data",
    "__pycache__",
    ".venv",
    "dist",
    "public",
}
# demo_recording.json is a real recorder capture of the demo site — its numerics
# are the wire format itself, not hand-typed schema knowledge.
EXCLUDED_FILES = {
    "rrweb_constants.py",
    "rrweb_constants.js",
    "extract_rrweb_constants.js",
    "demo_recording.json",
    Path(__file__).name,
}

CANONICAL_PAIRS = sorted(
    {
        (num, name)
        for attr in dir(rrweb_constants)
        if attr.endswith("_NAMES")
        for num, name in getattr(rrweb_constants, attr).items()
    }
)
CANONICAL_NAMES = sorted({name for _, name in CANONICAL_PAIRS}, key=len, reverse=True)

ALTERNATION = "|".join(re.escape(name) for name in CANONICAL_NAMES)
PAIR_ALTERNATION = "|".join(
    rf"""\b{num}\s*,\s*["']{re.escape(name)}["']"""
    rf"""|["']{re.escape(name)}["']\s*,\s*{num}\b"""
    for num, name in CANONICAL_PAIRS
)
FIELD = r"(?:[a-z][a-zA-Z]*)?(?:[Tt]ype|[Ss]ource)"
PATTERNS = {
    "numeric comparison against a type/source field": re.compile(
        rf"""\b{FIELD}\b["'\]]{{0,2}}\s*===?\s*\d"""
    ),
    "reversed numeric comparison against a type/source field": re.compile(
        rf"""\d\s*===?\s*[\w$."'\[\]]*\b{FIELD}\b"""
    ),
    "numeric literal assigned to a type/source key": re.compile(
        rf"""["']?\b{FIELD}\b["']?\s*:\s*\d"""
    ),
    "canonical name re-declared with a number": re.compile(
        rf"""\b(?:{ALTERNATION})\s*[:=]\s*\d"""
    ),
    "number mapped to a canonical name": re.compile(
        rf"""\d\s*:\s*["'](?:{ALTERNATION})["']"""
    ),
    "canonical value beside its own name": re.compile(PAIR_ALTERNATION),
    "positional row carrying a type numeral before its label": re.compile(
        rf"""\b\d[\d_]*\s*,\s*\d[\d_]*\s*,\s*["'](?:{ALTERNATION})["']"""
    ),
}


def source_files():
    # git enumeration (tracked plus untracked-unignored) scopes the sweep to
    # authored files: generated local data — analysis output, browser profiles,
    # loaded recordings — is gitignored and carries rrweb numerics legitimately.
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
            "*.js",
            "*.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for name in listed.splitlines():
        path = REPO / name
        if not path.exists():
            continue
        if set(Path(name).parts) & EXCLUDED_PARTS:
            continue
        if path.name in EXCLUDED_FILES:
            continue
        yield path


def test_generated_constants_are_what_the_generator_emits():
    proc = subprocess.run(
        ["bun", str(EVIDENCE / "distill" / "extract_rrweb_constants.js"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_types_package_version_matches_the_vendored_replayer():
    header = (
        (EVIDENCE / "src" / "locus" / "evidence" / "rrweb_constants.py")
        .read_text()
        .splitlines()[0]
    )
    generated_from = re.search(r"@rrweb/types@([\w.-]+)", header).group(1)
    replayers = list((EVIDENCE / "vendor").glob("rrweb-replay-*.min.js"))
    assert len(replayers) == 1
    vendored = re.match(r"rrweb-replay-(.+)\.min\.js", replayers[0].name).group(1)
    assert generated_from == vendored


def test_rrweb_schema_knowledge_only_in_the_canonical_mapping():
    assert len(CANONICAL_NAMES) > 40
    violations = []
    for path in source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    violations.append(
                        f"{path.relative_to(REPO)}:{lineno} [{label}] {line.strip()}"
                    )
    assert not violations, (
        "hand-rolled rrweb schema knowledge found — import the generated constants instead:\n"
        + "\n".join(violations)
    )
