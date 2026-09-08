"""Analysis-side test scaffolding: the deployment's Gemini id, hand-built claims and verdicts, a scripted Conversation, and the offline tokenizer.
The event builders, db builders, and the fixtures both suites share live in evidence/tests/_support.py, which conftest.py puts on the import path."""

from pathlib import Path

import pytest
import tomllib
from locus.analysis import budget
from locus.analysis.ground.claims import Claim, ClaimType, TimestampComponents
from locus.analysis.model.cards import GEMINI as GEMINI_CLIENT
from locus.analysis.model.cards import models_speaking
from locus.analysis.model.protocol import Conversation

REPO = Path(__file__).resolve().parents[2]

# The deployment's Gemini model, from the card that owns the id.
GEMINI = models_speaking(GEMINI_CLIENT)[0]


def card_roster(
    monkeypatch, root: Path, roster: dict[str, str], default: str | None = None
) -> Path:
    """Point the card roster at a test-built one: one <name>.toml per entry under root/cards, the card named by `default` stamped `default = true`. A body that declares no `model` gets `model = "<name>"` filled in, so a test states the field only when the field is what it tests. Returns the roster directory."""
    from locus.analysis.model import cards as cards_module

    directory = root / "cards"
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in roster.items():
        if "model" not in tomllib.loads(body):
            body = f'model = "{name}"\n{body}'
        if name == default:
            body = f"default = true\n{body}"
        (directory / f"{name}.toml").write_text(body)
    monkeypatch.setattr(cards_module, "cards_dir", lambda: directory)
    return directory


def remark_default(
    monkeypatch, tmp_path: Path, name: str, dirname: str = "remarked-cards"
) -> Path:
    """A copy of the current roster with `default = true` moved to `name`, monkeypatched in — the switch the change-deployment skill describes, done to a copy so the shipped files stay untouched."""
    from locus.analysis.model import cards as cards_module

    directory = tmp_path / dirname
    directory.mkdir()
    for path in cards_module.cards_dir().glob("*.toml"):
        body = path.read_text().replace("default = true\n", "")
        if path.stem == name:
            body = f"default = true\n{body}"
        (directory / path.name).write_text(body)
    monkeypatch.setattr(cards_module, "cards_dir", lambda: directory)
    return directory


def claim(
    claim_id,
    kind=ClaimType.OBSERVATION,
    start=None,
    end=None,
    supports=None,
    label=None,
):
    ts = lambda s: TimestampComponents(minutes=0, seconds=s, milliseconds=0)
    return Claim(
        claim_id=claim_id,
        claim_text=f"claim {claim_id}",
        claim_type=kind,
        supporting_claim_ids=supports,
        label=label,
        start_timestamp_components=ts(start) if start is not None else None,
        end_timestamp_components=ts(end) if end is not None else None,
    )


def slice_rows(conn, slice_ids, window_start=None, window_end=None):
    """The window's slice table in window.json's own shape — what the oracle-side suites hand to evaluate/judge, built by the one real implementation."""
    from dataclasses import asdict

    from locus.analysis.window import slice_table

    return [asdict(s) for s in slice_table(conn, slice_ids, window_start, window_end)]


class Verdicts:
    """What the judge's sized batch schema parses to — the one shape both grounding suites script."""

    def __init__(self, evaluations):
        self.evaluations = evaluations


PREPROCESSOR_CONFIG = {
    "patch_size": 16,
    "merge_size": 2,
    "size": {"shortest_edge": 65536, "longest_edge": 16777216},
}


def install_offline_tokenizer(monkeypatch):
    """Price windows without the network or an HF cache. The real pinned tokenizer is what
    ships; here text is one token per word, so a suite's expected counts are its own arithmetic.
    The preprocessor config is the real one — the screenshot-token closed form is pinned against it."""

    class Encoding:
        def __init__(self, n):
            self.ids = list(range(n))

    class Tokenizer:
        def encode(self, text):
            return Encoding(len(text.split()))

    monkeypatch.setattr(budget, "_tokenizer", Tokenizer())
    monkeypatch.setattr(budget, "_preprocessor_config", PREPROCESSOR_CONFIG)


class ScriptedConversation(Conversation):
    """A Conversation whose replies are pre-scripted. Records what it was asked
    per call (tools offered, response schema), every image, and tool
    results — so a test can assert on the real wire shape. `parts` keeps
    everything in the order it was added, which is what a test of composition
    asserts on; `texts`/`images`/`tool_results` are views of it.

    Every image costs one token and the context holds `room` of them, so a test
    can drive the engine's serve-while-it-fits boundary with small numbers."""

    def __init__(self, responses=(), system_prompt="", room=1_000_000, record_dir=None):
        super().__init__(record_dir)
        self.system_prompt = system_prompt
        self.responses = list(responses)
        self.room = room
        self.parts = []
        self.texts = []
        self.images = []
        self.image_pushed = []
        self.tool_results = []
        self.calls = []

    def _add_user_text(self, text):
        self.parts.append(("text", text))
        self.texts.append(text)

    def _add_user_image(self, image_path, pushed):
        self.parts.append(("image", str(image_path)))
        self.images.append(str(image_path))
        self.image_pushed.append(pushed)
        return f"sent://{image_path}"

    def image_tokens(self, image_path):
        return 1

    def context_remaining(self):
        return self.room - len(self.images)

    def _add_tool_result(self, call_id, content):
        self.parts.append(("tool", call_id, content))
        self.tool_results.append((call_id, content))

    def _get_response(self, *, response_schema=None, tools=None):
        self.calls.append(
            {
                "tools": [t["name"] for t in tools] if tools else None,
                "schema": getattr(response_schema, "__name__", response_schema),
            }
        )
        return self.responses.pop(0)

    @property
    def execution_mode(self):
        return "scripted"


def horizon_at(root: Path) -> None:
    """A retention horizon declared under a test deployment root — every load and doctor holds the db to one. The suites' recordings are stamped at a fixed moment in the past, so the horizon declared here is the one that reaches them; a test of the horizon itself overwrites it."""
    from datetime import datetime, timezone

    from _support import T0
    from locus.evidence.retention import CONFIG_FILE

    recorded = datetime.fromtimestamp(T0 / 1000, timezone.utc).date()
    reach = (datetime.now(timezone.utc).date() - recorded).days + 1
    (root / "store").mkdir(exist_ok=True)
    (root / CONFIG_FILE).write_text(f"retention_days = {reach}\n")


@pytest.fixture(autouse=True)
def spend_isolation(tmp_path_factory, monkeypatch):
    """Every test runs against its own spend home: the declaration and ledger land in a throwaway root, the price-registry fetch is offline, and the once-per-process verification throttle is reset — so no test writes spend into the working clone or touches the network. The root starts as a fresh clone does — the repo's shipped spend.toml, byte-identical — and spend tests overwrite that declaration and seed the ledger as they need."""
    from locus.analysis import spend

    root = tmp_path_factory.mktemp("spend_root")
    (root / "config").mkdir()
    (root / "config" / "spend.toml").write_text(
        (REPO / "config" / "spend.toml").read_text()
    )
    monkeypatch.setattr(spend, "_root", lambda: root)
    monkeypatch.setattr(spend, "_memo_path", lambda: root / "price_registry_memo.json")

    def offline():
        raise OSError("network disabled under test")

    monkeypatch.setattr(spend, "_fetch_registry", offline)
    monkeypatch.setattr(spend, "_VERIFIED", set())
    return root
