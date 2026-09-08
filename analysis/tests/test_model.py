import json
import time
from pathlib import Path

import pytest
from _analysis_support import (
    GEMINI,
    install_offline_tokenizer,
)
from locus.analysis.model import Conversation
from locus.analysis.model.cards import SUPPORTED_SAMPLING, load_card, thinking_levels
from locus.analysis.model.factory import make_conversation
from locus.analysis.model.gemini import (
    CACHE_READ_TTL_S,
    GEMINI_FILE_TTL_S,
    PROTECTED_AGE_S,
    UploadCache,
)
from locus.analysis.model.openai_compat import OpenAICompatConversation
from locus.analysis.model.protocol import DryRun, Response


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-not-used")


@pytest.fixture(autouse=True)
def offline_pricing(monkeypatch):
    install_offline_tokenizer(monkeypatch)


def test_factory_dispatch(tmp_path):
    local = make_conversation(
        "qwen3.6-35b-a3b-6bit", "sys", record_dir=tmp_path / "record"
    )
    gemini = make_conversation(
        GEMINI, "sys", cache_path=tmp_path / "cache.db", record_dir=tmp_path / "record"
    )
    assert isinstance(local, OpenAICompatConversation)
    for conversation in (local, gemini):
        assert isinstance(conversation, Conversation)
    with pytest.raises(ValueError, match="no card declares model"):
        make_conversation("qwen/qwen3.6-flash", "sys", record_dir=tmp_path / "record")

    # An unknown model must not just refuse — it must name the cards that do exist, or the agent has
    # nowhere to go from the error. The check is that every local card is listed, not which ones
    # they happen to be today: a test that pins the roster fails when a deployment adds a model,
    # which is the one thing adding a model must not do.
    from locus.analysis.model.cards import declared_models, load_card

    local = []
    for name in declared_models():
        try:
            load_card(name)
        except ValueError:
            continue
        local.append(name)

    with pytest.raises(ValueError) as refused:
        make_conversation("gpt-nonsense", "sys", record_dir=tmp_path / "record")
    assert all(name in str(refused.value) for name in local)


def test_the_card_names_its_client(tmp_path, monkeypatch):
    """A card's `conversation` declaration is the only thing that routes: a card named gemini-anything that names openai-compatible reaches that client at its base_url, a card named qwen-anything that names gemini is Gemini's, a card naming a client the registry lacks refuses naming the registry, and a card naming none refuses naming the field."""
    from _analysis_support import card_roster
    from locus.analysis.model import cards
    from locus.analysis.model.factory import CONVERSATIONS

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "gemini-styled-local": (
                'conversation = "openai-compatible"\n'
                'model = "someone/model"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                'family = "qwen3"\n'
                "max_image_tokens = { activity_screenshots = 600, pulled_screenshots = 1200 }\n"
                "thinking_budget = 1\n"
                "max_tokens = 1\n"
                "context_tokens = 1000\n"
                "headroom = 0.8\n"
                "[sampling]\n"
                "temperature = 1.0\n"
            ),
            "qwen-styled-cloud": (
                'conversation = "gemini"\n'
                'thinking_levels = ["low"]\n'
                "context_tokens = 1000\n"
                "headroom = 0.5\n"
                "max_output_tokens = 64\n"
                "[pricing]\n"
                "input_per_mtok = 1.0\ncached_input_per_mtok = 0.1\noutput_per_mtok = 2.0\n"
            ),
            "names-a-stranger": (
                'conversation = "anthropic"\ncontext_tokens = 1000\nheadroom = 0.5\n'
            ),
            "names-nothing": "context_tokens = 1000\nheadroom = 0.5\n",
        },
    )

    assert cards.conversation("gemini-styled-local") == cards.OPENAI_COMPATIBLE
    assert cards.conversation("qwen-styled-cloud") == cards.GEMINI
    assert cards.models_speaking(cards.OPENAI_COMPATIBLE) == ["gemini-styled-local"]
    conversation = make_conversation(
        "gemini-styled-local", "sys", record_dir=tmp_path / "record"
    )
    assert isinstance(conversation, OpenAICompatConversation)
    assert conversation.base_url == "http://127.0.0.1:9/v1"
    from locus.analysis.model.gemini import GeminiConversation

    assert isinstance(
        make_conversation(
            "qwen-styled-cloud",
            "sys",
            cache_path=tmp_path / "cache.db",
            record_dir=tmp_path / "record",
        ),
        GeminiConversation,
    )
    with pytest.raises(ValueError) as stranger:
        make_conversation("names-a-stranger", "sys", record_dir=tmp_path / "record")
    assert "anthropic" in str(stranger.value)
    assert all(name in str(stranger.value) for name in CONVERSATIONS)
    with pytest.raises(ValueError, match="declares no `conversation`"):
        make_conversation("names-nothing", "sys", record_dir=tmp_path / "record")
    with pytest.raises(ValueError, match="not an openai-compatible card"):
        load_card("qwen-styled-cloud")


def test_default_card_is_the_marked_card_and_nothing_else(tmp_path, monkeypatch):
    """The default card is the roster card marked `default = true` and nothing else: a key in the environment never picks it, declaration order never picks it, and no mark refuses only when the default is actually asked for. The roster's cross-card invariants refuse every read: two marks, a non-bool mark, a missing `model`, two cards declaring one model."""
    from _analysis_support import card_roster
    from locus.analysis.model import cards

    roster = {
        "gemini-a": "context_tokens = 1\n",
        "local-a": "[sampling]\ntemperature = 1.0\n",
        "local-b": "[sampling]\ntemperature = 1.0\n",
    }
    card_roster(monkeypatch, tmp_path, roster, default="local-b")

    monkeypatch.setenv("GOOGLE_API_KEY", "banked")
    assert cards.default_card() == "local-b", "a banked key does not pick the card"
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert cards.default_card() == "local-b", "declaration order does not pick it"

    unmarked = tmp_path / "unmarked"
    card_roster(monkeypatch, unmarked, roster)
    assert cards.declared_models() == sorted(roster), "an unmarked roster reads fine"
    with pytest.raises(ValueError, match="no card declares `default = true`.*--model"):
        cards.default_card()

    twice = tmp_path / "twice"
    card_roster(
        monkeypatch,
        twice,
        {**roster, "local-a": "default = true\n" + roster["local-a"]},
        default="local-b",
    )
    with pytest.raises(ValueError, match="2 cards declare `default = true`"):
        cards.default_card()

    typoed = tmp_path / "typoed"
    card_roster(
        monkeypatch,
        typoed,
        {**roster, "local-b": 'default = "yes"\n' + roster["local-b"]},
    )
    with pytest.raises(ValueError, match="`default` is a bool"):
        cards.default_card()

    twins = tmp_path / "twins"
    card_roster(
        monkeypatch,
        twins,
        {**roster, "local-b": 'model = "local-a"\n' + roster["local-b"]},
        default="local-a",
    )
    with pytest.raises(ValueError, match="1:1"):
        cards.default_card()

    modelless = tmp_path / "modelless"
    directory = card_roster(monkeypatch, modelless, roster, default="local-b")
    (directory / "local-a.toml").write_text("[sampling]\ntemperature = 1.0\n")
    with pytest.raises(ValueError, match="declares no `model`"):
        cards.default_card()


def test_openai_compat_merges_user_turns(tmp_path):
    from PIL import Image

    image = tmp_path / "f.png"
    Image.new("RGB", (8, 8)).save(image)
    conversation = OpenAICompatConversation(
        "sys", card=load_card("qwen3.6-35b-a3b-6bit"), record_dir=tmp_path / "record"
    )
    conversation.add_user_text("events:")
    conversation.add_user_image(str(image))
    conversation.add_user_text("guidelines")

    assert len(conversation.contents) == 1
    parts = conversation.contents[0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text"]
    assert parts[1]["image_url"]["url"] == str(image.resolve())

    wire = conversation._wire_messages()
    assert wire[0] == {"role": "system", "content": "sys"}

    with pytest.raises(FileNotFoundError):
        conversation.add_user_image(str(tmp_path / "missing.png"))


class RecordedConversation(Conversation):
    def __init__(self, responses=(), record_dir=None):
        super().__init__(record_dir)
        self.system_prompt = "sys prompt"
        self.responses = list(responses)

    def _add_user_text(self, text):
        pass

    def _add_user_image(self, image_path, pushed):
        return f"remote://{Path(image_path).name}"

    def image_tokens(self, image_path):
        return 1

    def context_remaining(self):
        return 1_000

    def _add_tool_result(self, call_id, content):
        pass

    def _get_response(self, *, response_schema=None, tools=None):
        if not self.responses:
            raise RuntimeError("backend boom")
        return self.responses.pop(0)

    @property
    def execution_mode(self):
        return "test"


def test_persisted_calls_capture_the_full_payload_as_sent(tmp_path):
    image = tmp_path / "image_1.png"
    image.write_bytes(b"\x89PNG fake")
    payloads = tmp_path / "payloads" / "analysis"
    conversation = RecordedConversation(
        responses=[
            Response(
                text="answer one",
                thoughts_text="thinking…",
                usage_metadata={"total_tokens": 9},
            ),
            Response(text="answer two"),
        ],
        record_dir=payloads,
    )
    conversation.add_user_text("events stream")
    conversation.add_user_image(str(image))
    conversation.add_user_text("guidelines")
    conversation.get_response(label="answer")
    conversation.add_user_text("revise please")
    conversation.get_response(label="revision")

    first = (payloads / "1_answer_input.txt").read_text()
    assert first.startswith("=== system ===\n\nsys prompt")
    assert first.count("=== user ===") == 1
    assert "events stream" in first
    assert (
        '![image_1.png](../../image_1.png "sent as remote://image_1.png")' in first
    ), "the pointer resolves from the payload file that carries it"
    assert "revise please" not in first
    assert (payloads / "1_answer_response.txt").read_text() == (
        "=== thoughts ===\nthinking…\n\n=== response ===\nanswer one\n"
    )
    meta = json.loads((payloads / "1_answer_meta.json").read_text())
    assert meta["config"]["execution_mode"] == "test"
    assert meta["usage"] == {"total_tokens": 9}
    assert meta["elapsed_s"] >= 0
    assert meta["error"] is None

    second = (payloads / "2_revision_input.txt").read_text()
    assert "=== assistant ===\n\nanswer one" in second
    assert "thinking…" not in second
    assert second.rstrip().endswith("revise please")
    assert (payloads / "2_revision_response.txt").read_text() == "answer two\n"


def test_record_collision_refuses_before_the_wire(tmp_path):
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    (payloads / "1_answer_input.txt").write_text("a prior attempt's record\n")
    conversation = RecordedConversation(
        responses=[Response(text="never sent")], record_dir=payloads
    )
    conversation.add_user_text("events stream")
    with pytest.raises(FileExistsError):
        conversation.get_response(label="answer")
    assert conversation.responses, "the wire was reached despite the collision"
    assert (payloads / "1_answer_input.txt").read_text() == (
        "a prior attempt's record\n"
    )


def test_input_record_lands_before_the_wire(tmp_path):
    payloads = tmp_path / "payloads"
    seen = {}

    class WireProbe(RecordedConversation):
        def _get_response(self, **kwargs):
            seen["input_on_disk"] = (payloads / "1_answer_input.txt").exists()
            return super()._get_response(**kwargs)

    conversation = WireProbe(responses=[Response(text="ok")], record_dir=payloads)
    conversation.add_user_text("events stream")
    conversation.get_response(label="answer")
    assert seen["input_on_disk"], (
        "the input record must land before the request is sent"
    )


def test_failed_calls_persist_with_their_error(tmp_path):
    conversation = RecordedConversation(responses=[], record_dir=tmp_path / "payloads")
    conversation.add_user_text("hello")
    with pytest.raises(RuntimeError, match="backend boom"):
        conversation.get_response()
    assert (tmp_path / "payloads" / "1_input.txt").exists()
    assert not (tmp_path / "payloads" / "1_response.txt").exists()
    meta = json.loads((tmp_path / "payloads" / "1_meta.json").read_text())
    assert "backend boom" in meta["error"]
    assert meta["usage"] is None
    assert meta["elapsed_s"] >= 0


def test_dry_run_refuses_at_the_wire_after_persisting(tmp_path):
    from PIL import Image

    image = tmp_path / "screenshot.png"
    Image.new("RGB", (390, 663)).save(image)

    local = OpenAICompatConversation(
        "sys",
        card=load_card("qwen3.6-35b-a3b-6bit"),
        dry_run=True,
        record_dir=tmp_path / "local_payloads",
    )
    local.add_user_text("events stream")
    local.add_user_image(str(image))
    with pytest.raises(DryRun, match="not sent"):
        local.get_response(label="answer")
    record = (tmp_path / "local_payloads" / "1_answer_input.txt").read_text()
    assert "events stream" in record
    assert "screenshot.png" in record
    assert not (tmp_path / "local_payloads" / "1_answer_response.txt").exists()
    meta = json.loads((tmp_path / "local_payloads" / "1_answer_meta.json").read_text())
    assert "DryRun" in meta["error"]

    from locus.analysis.model.gemini import GeminiConversation

    gemini = GeminiConversation(
        GEMINI,
        "sys",
        dry_run=True,
        cache_path=tmp_path / "cache.db",
        record_dir=tmp_path / "gemini_payloads",
    )
    gemini.add_user_text("events stream")
    with pytest.raises(DryRun, match="not sent"):
        gemini.get_response(label="answer")
    assert (tmp_path / "gemini_payloads" / "1_answer_input.txt").exists()


def test_cards_carry_only_supported_sampling_fields():
    from locus.analysis.model.cards import OPENAI_COMPATIBLE, models_speaking

    for name in models_speaking(OPENAI_COMPATIBLE):
        card = load_card(name)
        assert card["model"]
        assert card["thinking_budget"] > 0
        assert card["context_tokens"] > card["max_tokens"]
        assert set(card["sampling"]) <= SUPPORTED_SAMPLING


def _tiny_card(context_tokens):
    card = json.loads(json.dumps(load_card("qwen3.6-35b-a3b-6bit")))
    card["max_tokens"] = 64
    card["context_tokens"] = context_tokens
    return card


def test_oversized_request_refused_before_send(tmp_path):
    conversation = OpenAICompatConversation(
        "sys", card=_tiny_card(100), record_dir=tmp_path / "record"
    )
    conversation.add_user_text(" ".join(["word"] * 50))
    with pytest.raises(ValueError, match="refused before send") as excinfo:
        conversation.get_response()
    message = str(excinfo.value)
    assert "51 prompt tokens" in message
    assert "max_tokens 64" in message
    assert "context 100" in message
    assert "smaller window" in message


def test_wire_boundary_counts_images_in_tokens(tmp_path):
    from PIL import Image

    image = tmp_path / "screenshot.png"
    Image.new("RGB", (390, 663)).save(image)

    conversation = OpenAICompatConversation(
        "sys", card=_tiny_card(300), record_dir=tmp_path / "record"
    )
    conversation.add_user_text("a b c")
    conversation.add_user_image(str(image))
    with pytest.raises(ValueError, match="refused before send"):
        conversation.get_response()


def test_the_card_ceiling_shrinks_an_oversized_screenshot_to_budget(tmp_path):
    from locus.analysis.budget import qwen_image_tokens
    from PIL import Image

    image = tmp_path / "screenshot.png"
    Image.new("RGB", (2560, 1323)).save(image)

    conversation = OpenAICompatConversation(
        "sys", card=_tiny_card(100_000), record_dir=tmp_path / "record"
    )
    ceiling = conversation.max_image_tokens["activity_screenshots"]
    conversation.add_user_image(str(image), pushed=True)

    sent = conversation.contents[0]["content"][0]["image_url"]["url"]
    assert sent != str(image.resolve()), "the server is handed a smaller file"
    with Image.open(sent) as shrunk:
        cost = qwen_image_tokens(*shrunk.size)
    assert cost <= ceiling, "the card's ceiling is a hard per-image bound"
    assert cost > ceiling * 0.9, "the resize lands near the ceiling, not far under"


def test_a_pull_rides_under_its_own_ceiling_and_is_priced_so(tmp_path):
    from locus.analysis.budget import qwen_image_tokens
    from PIL import Image

    image = tmp_path / "pull.png"
    Image.new("RGB", (2560, 1323)).save(image)

    conversation = OpenAICompatConversation(
        "sys", card=_tiny_card(100_000), record_dir=tmp_path / "record"
    )
    activity = conversation.max_image_tokens["activity_screenshots"]
    pulled = conversation.max_image_tokens["pulled_screenshots"]
    cost = conversation.image_tokens(str(image))
    conversation.add_user_image(str(image))

    sent = conversation.contents[0]["content"][0]["image_url"]["url"]
    assert sent != str(image.resolve()), "over the pull ceiling, a smaller file"
    with Image.open(sent) as shrunk:
        assert qwen_image_tokens(*shrunk.size) == cost, "priced as sent"
    assert activity < cost <= pulled, "a pull rides under its own, higher ceiling"


def test_a_local_card_with_a_bare_image_ceiling_is_refused(tmp_path, monkeypatch):
    from _analysis_support import card_roster
    from locus.analysis.model.cards import max_image_tokens

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "qwen-bare": (
                'conversation = "openai-compatible"\n'
                'model = "mlx-community/x"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                'family = "qwen3"\n'
                "max_image_tokens = 600\n"
                "thinking_budget = 1\n"
                "max_tokens = 1\n"
                "context_tokens = 1000\n"
                "headroom = 0.8\n"
                "[sampling]\n"
                "temperature = 1.0\n"
            )
        },
    )
    with pytest.raises(ValueError, match="activity_screenshots"):
        max_image_tokens("qwen-bare")


def test_a_screenshot_under_the_ceiling_passes_untouched(tmp_path):
    from PIL import Image

    image = tmp_path / "small.png"
    Image.new("RGB", (320, 200)).save(image)

    conversation = OpenAICompatConversation(
        "sys", card=_tiny_card(100_000), record_dir=tmp_path / "record"
    )
    conversation.add_user_image(str(image), pushed=True)

    sent = conversation.contents[0]["content"][0]["image_url"]["url"]
    assert sent == str(image.resolve()), (
        "under the ceiling nothing moves — a screenshot is never upscaled"
    )


def test_local_cards_declare_family_image_ceiling_and_endpoint():
    # The counting/resize arithmetic (pinned tokenizer, smart_resize closed
    # form) is one family's; a card states its membership and its per-image
    # ceiling so neither is ever an unstated assumption. The endpoint is the
    # card's too — the one home analysis dials and the server binds.
    from locus.analysis.model.cards import SUPPORTED_FAMILIES

    card = load_card("qwen3.6-35b-a3b-6bit")
    assert card["family"] in SUPPORTED_FAMILIES
    assert set(card["max_image_tokens"]) == {
        "activity_screenshots",
        "pulled_screenshots",
    }
    assert (
        card["max_image_tokens"]["pulled_screenshots"]
        > card["max_image_tokens"]["activity_screenshots"]
        > 0
    )
    assert card["base_url"].startswith("http")
    from urllib.parse import urlparse

    parsed = urlparse(card["base_url"])
    assert parsed.hostname and parsed.port


def test_a_card_outside_the_arithmetic_family_is_refused(tmp_path, monkeypatch):

    from _analysis_support import card_roster

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "llama-x": (
                'conversation = "openai-compatible"\n'
                'model = "someone/llama-x"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                'family = "llama"\n'
                "max_image_tokens = { activity_screenshots = 600, pulled_screenshots = 1200 }\n"
                "thinking_budget = 1\n"
                "max_tokens = 1\n"
                "context_tokens = 1000\n"
                "headroom = 0.8\n"
                "[sampling]\n"
                "temperature = 1.0\n"
            )
        },
    )
    with pytest.raises(ValueError, match="family"):
        load_card("llama-x")


def test_a_local_card_without_an_image_ceiling_is_refused(tmp_path, monkeypatch):

    from _analysis_support import card_roster

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "qwen-bare": (
                'conversation = "openai-compatible"\n'
                'model = "someone/qwen-bare"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                'family = "qwen3"\n'
                "thinking_budget = 1\n"
                "max_tokens = 1\n"
                "context_tokens = 1000\n"
                "headroom = 0.8\n"
                "[sampling]\n"
                "temperature = 1.0\n"
            )
        },
    )
    with pytest.raises(ValueError, match="max_image_tokens"):
        load_card("qwen-bare")


def test_a_local_card_without_base_url_is_refused(tmp_path, monkeypatch):

    from _analysis_support import card_roster

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "qwen-bare": (
                'conversation = "openai-compatible"\n'
                'model = "someone/qwen-bare"\n'
                'family = "qwen3"\n'
                "max_image_tokens = { activity_screenshots = 600, pulled_screenshots = 1200 }\n"
                "thinking_budget = 1\n"
                "max_tokens = 1\n"
                "context_tokens = 1000\n"
                "headroom = 0.8\n"
                "[sampling]\n"
                "temperature = 1.0\n"
            )
        },
    )
    with pytest.raises(ValueError, match="base_url"):
        load_card("qwen-bare")


def test_a_local_card_base_url_without_port_is_refused(tmp_path, monkeypatch):

    from _analysis_support import card_roster

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "qwen-bare": (
                'conversation = "openai-compatible"\n'
                'model = "someone/qwen-bare"\n'
                'base_url = "http://127.0.0.1/v1"\n'
                'family = "qwen3"\n'
                "max_image_tokens = { activity_screenshots = 600, pulled_screenshots = 1200 }\n"
                "thinking_budget = 1\n"
                "max_tokens = 1\n"
                "context_tokens = 1000\n"
                "headroom = 0.8\n"
                "[sampling]\n"
                "temperature = 1.0\n"
            )
        },
    )
    with pytest.raises(ValueError, match="explicit host and port"):
        load_card("qwen-bare")


def test_request_within_context_passes_the_wire_boundary(tmp_path):
    conversation = OpenAICompatConversation(
        "sys", card=_tiny_card(300), dry_run=True, record_dir=tmp_path / "record"
    )
    conversation.add_user_text("a b c")
    with pytest.raises(DryRun, match="not sent"):
        conversation.get_response()


def test_card_claiming_unsupported_field_fails_loud(tmp_path, monkeypatch):
    from _analysis_support import card_roster
    from locus.analysis.model import cards

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "some-model": (
                'conversation = "openai-compatible"\nmodel = "m"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                'family = "qwen3"\n'
                "max_image_tokens = { activity_screenshots = 600, pulled_screenshots = 1200 }\nthinking_budget = 1\n"
                "[sampling]\n"
                "temperature = 1.0\ntop_k = 20\n"
            )
        },
    )
    with pytest.raises(ValueError, match=r"top_k.*SUPPORTED_SAMPLING"):
        cards.load_card("some-model")


# The shipped card roster is the deployment's declaration, and pricing reads it before any model is called.
# A card missing a composition fact would refuse every analysis priced against it, so the file is checked as a whole rather than one entry at a time.


def test_every_declared_model_declares_what_composition_needs():
    from locus.analysis.model.cards import context_tokens, declared_models, headroom

    models = declared_models()
    assert models, "a deployment with no declared model can analyze nothing"
    for name in models:
        assert context_tokens(name) > 0
        assert 0 < headroom(name) <= 1, (
            f"{name}'s headroom is a fraction of its context, not a token count"
        )


def test_a_model_with_no_card_is_named_loudly_not_guessed_at():
    from locus.analysis.model.cards import context_tokens, declared_models

    with pytest.raises(ValueError, match="context_tokens"):
        context_tokens("a-model-nobody-declared")
    with pytest.raises(ValueError, match=declared_models()[0]):
        context_tokens("a-model-nobody-declared")


def test_every_shipped_local_card_states_only_what_the_engine_honors():
    from locus.analysis.model.cards import (
        SUPPORTED_SAMPLING,
        declared_models,
        load_card,
    )

    local = []
    for name in declared_models():
        try:
            card = load_card(name)  # raises if it claims a field the sampler drops
        except ValueError:
            continue  # a cloud entry: composition facts only, no sampling
        local.append(name)
        assert card["sampling"], f"{name} declares an empty sampling block"
        assert set(card["sampling"]) <= SUPPORTED_SAMPLING
        assert card["model"], f"{name} names no repo to request"
        assert card["base_url"], f"{name} names no endpoint to dial"
    assert local, "the bundled local backend needs at least one card"


def test_every_shipped_gemini_card_declares_a_valid_pricing_block():
    """Every Gemini card bills, so every shipped one must carry the full validated price block — a card without it would refuse every conversation, and a card whose block slipped past validation would ledger wrong dollars."""
    from locus.analysis.model.cards import (
        GEMINI,
        PRICING_FIELDS,
        models_speaking,
        pricing,
    )

    models = models_speaking(GEMINI)
    assert models, "the shipped roster declares at least one Gemini card"
    for name in models:
        assert set(pricing(name)) == set(PRICING_FIELDS)


def test_a_gemini_card_without_prices_refuses_at_construction(tmp_path, monkeypatch):
    """The wire always bills, so a card declaring no prices fails before anything is spent — never a conversation that prices every call at zero dollars past the walls and the ledger."""

    from _analysis_support import card_roster

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "gemini-unpriced": (
                'conversation = "gemini"\n'
                "context_tokens = 1000\n"
                "headroom = 0.5\n"
                "max_output_tokens = 64\n"
            )
        },
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "banked")

    with pytest.raises(ValueError, match="pricing"):
        make_conversation("gemini-unpriced", "sys", record_dir=tmp_path / "record")


def test_request_states_full_config_explicitly(tmp_path):
    card = load_card("qwen3.6-35b-a3b-6bit")
    thinking = OpenAICompatConversation(
        "sys", card=card, record_dir=tmp_path / "record"
    )
    assert thinking.base_url == card["base_url"], (
        "the conversation dials the card's declared endpoint"
    )
    thinking.add_user_text("q")
    params = thinking._request_params()
    assert params["model"] == card["model"]
    for key, value in card["sampling"].items():
        sent = params.get(key, params["extra_body"].get(key))
        assert sent == value, f"{key} not stated on the wire"
    assert params["extra_body"]["enable_thinking"] is True
    assert params["extra_body"]["thinking_budget"] == card["thinking_budget"]
    assert params["max_tokens"] == card["max_tokens"]


class _FakeStream:
    """The openai Stream surface the backend consumes: a context manager
    yielding ChatCompletionChunk objects."""

    def __init__(self, chunks):
        from openai.types.chat import ChatCompletionChunk

        self.chunks = [
            ChatCompletionChunk.model_validate(
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "m",
                    "choices": [],
                    **c,
                }
            )
            for c in chunks
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.chunks)


def _streaming_conversation(chunks, record_dir, captured=None, card=None):
    from types import SimpleNamespace

    def create(**params):
        if captured is not None:
            captured.update(params)
        return SimpleNamespace(
            headers={"x-locus-request": "30fd3652c349"},
            parse=lambda: _FakeStream(chunks),
        )

    conversation = OpenAICompatConversation(
        "sys", card=card or load_card("qwen3.6-35b-a3b-6bit"), record_dir=record_dir
    )
    conversation.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=create)
            )
        )
    )
    return conversation


def test_local_stream_reassembles_the_non_streamed_record(tmp_path):
    captured = {}
    conversation = _streaming_conversation(
        record_dir=tmp_path / "record",
        captured=captured,
        chunks=[
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning": "hmm "}}
                ]
            },
            {"choices": [{"index": 0, "delta": {"reasoning": "ok"}}]},
            {"choices": [{"index": 0, "delta": {"content": "\n\nhi "}}]},
            {
                "choices": [
                    {"index": 0, "delta": {"content": "there"}, "finish_reason": "stop"}
                ]
            },
            {
                "usage": {
                    "prompt_tokens": 13419,
                    "completion_tokens": 4647,
                    "total_tokens": 18066,
                    "prompt_tokens_details": {"cached_tokens": 6017},
                },
                "timings": {
                    "prompt_n": 7402,
                    "cache_n": 6017,
                    "predicted_n": 4647,
                    "prompt_ms": 9120.0,
                    "prompt_per_token_ms": 1.23,
                    "prompt_per_second": 811.6,
                    "predicted_ms": 88100.0,
                    "predicted_per_token_ms": 18.96,
                    "predicted_per_second": 52.7,
                    "peak_memory": 24.8,
                },
            },
        ],
    )
    conversation.add_user_text("q")

    response = conversation.get_response()
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert response.text == "hi there"
    assert response.thoughts_text == "hmm ok"
    assert conversation.contents[-1] == {
        "role": "assistant",
        "content": "hi there",
        "reasoning_content": "hmm ok",
        "reasoning": "hmm ok",
    }
    usage = response.usage_metadata
    assert usage["prompt_tokens"] == 13419
    assert usage["prompt_tokens_details"] == {"cached_tokens": 6017}
    assert usage["timings"]["predicted_per_second"] == 52.7
    assert usage["timings"]["cache_n"] == 6017
    assert usage["timings"]["peak_memory"] == 24.8
    assert "completion_tokens_details" not in usage
    assert usage["server_request"] == "30fd3652c349", (
        "the server's telemetry id joins this call to its memory.jsonl rows"
    )


def test_local_call_reports_its_phase_as_the_wire_reveals_it(tmp_path):
    """A caller's pulse reads the conversation's progress while the call blocks: waiting until the server's headers arrive (the server admitted the request), prefilling until the first delta, then generating — with the server's request id on every line after admission, so the run's log joins to the server's."""
    seen = []
    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        chunks=[
            {"choices": [{"index": 0, "delta": {"content": "a"}}]},
            {
                "choices": [
                    {"index": 0, "delta": {"content": "b"}, "finish_reason": "stop"}
                ]
            },
            {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        ],
    )
    original_create = conversation.client.chat.completions.with_raw_response.create

    def create(**params):
        seen.append(conversation.progress_line())
        return original_create(**params)

    conversation.client.chat.completions.with_raw_response.create = create
    # The waiting line also reads the server's /health; none is in reach here.
    conversation._server_admission = lambda: None
    conversation.add_user_text("q")
    assert conversation.progress_line() == ""

    conversation.get_response()

    assert seen == ["waiting for the server's admission"]
    assert conversation.progress["phase"] == "generating"
    assert conversation.progress["server_request"] == "30fd3652c349"
    assert conversation.progress_line() == (
        "server request 30fd3652c349 generating, 2 delta(s) received"
    )


def test_a_waiting_call_reads_the_servers_admission_state(tmp_path, monkeypatch):
    """While the call waits for admission, the line says what the server holds — read from /health — so a wait has a length instead of looking like a hang; a server that cannot be read leaves the line as it was."""
    conversation = _streaming_conversation(record_dir=tmp_path / "r", chunks=[])
    conversation.progress = {
        "phase": "waiting",
        "line": "waiting for the server's admission",
    }
    monkeypatch.setattr(
        conversation,
        "_server_admission",
        lambda: {"running": 1, "waiting": 3},
    )
    assert conversation.progress_line() == (
        "waiting for the server's admission: it is serving 1 request with 3 waiting, one at a time in arrival order"
    )
    monkeypatch.setattr(conversation, "_server_admission", lambda: None)
    assert conversation.progress_line() == "waiting for the server's admission", (
        "a server that cannot be read leaves the line as it was"
    )


def test_local_stream_merges_tool_call_deltas_by_index(tmp_path):
    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        chunks=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "request_screenshots",
                                        "arguments": '{"timesta',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'mps": [1200]}'},
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {"name": "other", "arguments": "{}"},
                                },
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ],
    )
    conversation.add_user_text("q")

    response = conversation.get_response()
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("call_a", "request_screenshots", {"timestamps": [1200]}),
        ("call_b", "other", {}),
    ]
    turn = conversation.contents[-1]
    assert turn["tool_calls"][0]["function"]["arguments"] == ('{"timestamps": [1200]}')


def test_local_tool_call_arguments_that_never_parse_are_marked_not_raised(tmp_path):
    """The arguments are the model's own streamed writing: text that is not a JSON object must not kill the run at normalization — the call arrives with arguments None, exists to be answered, and the record keeps the raw text the model actually wrote."""
    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        chunks=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "request_screenshots",
                                        "arguments": '{"timestamps": [12',
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {
                                        "name": "request_screenshots",
                                        "arguments": '"just a string"',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ],
    )
    conversation.add_user_text("q")

    response = conversation.get_response()
    assert [(c.id, c.arguments) for c in response.tool_calls] == [
        ("call_a", None),
        ("call_b", None),
    ], "truncated JSON and a non-object both mark the call, never raise"
    turn = conversation.contents[-1]
    assert turn["tool_calls"][0]["function"]["arguments"] == '{"timestamps": [12', (
        "the record keeps the text the model actually wrote"
    )


def test_local_tool_result_must_answer_a_call_the_model_made(tmp_path):
    """A result addressed to a call the model never made answers into the void, and the call it did make
    goes unanswered — the model then waits on a turn that never comes. Refuse it at the boundary."""
    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        chunks=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "request_screenshots",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ],
    )
    conversation.add_user_text("q")

    with pytest.raises(ValueError, match="never made"):
        conversation.add_tool_result("call_a", "rendered 2 screenshots")

    conversation.get_response()
    conversation.add_tool_result("call_a", "rendered 2 screenshots")
    assert conversation.contents[-1] == {
        "role": "tool",
        "tool_call_id": "call_a",
        "content": "rendered 2 screenshots",
    }

    with pytest.raises(ValueError, match="never made"):
        conversation.add_tool_result("call_ghost", "for nobody")


def test_a_tool_call_is_answered_exactly_once(tmp_path):
    from locus.analysis.model.protocol import ToolCall

    conversation = RecordedConversation(
        responses=[
            Response(
                text="",
                tool_calls=[
                    ToolCall(id="call_a", name="request_screenshots", arguments={})
                ],
            )
        ],
        record_dir=tmp_path,
    )
    conversation.add_user_text("q")
    conversation.get_response()
    conversation.add_tool_result("call_a", "rendered 1 screenshot")
    with pytest.raises(ValueError, match="already answered"):
        conversation.add_tool_result("call_a", "rendered again")


def test_the_context_left_anchors_on_the_last_responses_usage(tmp_path):
    from PIL import Image

    image = tmp_path / "screenshot.png"
    Image.new("RGB", (320, 200)).save(image)

    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        chunks=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "seen"},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "usage": {
                    "prompt_tokens": 9000,
                    "completion_tokens": 1000,
                    "total_tokens": 10000,
                }
            },
        ],
    )
    conversation.add_user_text("q")
    with pytest.raises(RuntimeError, match="no response has been received"):
        conversation.context_remaining()

    conversation.get_response()
    base = conversation.context_tokens - conversation.max_tokens - 10000
    assert conversation.context_remaining() == base, (
        "the anchor is the wire's persisted usage total, not a client-side walk"
    )

    conversation.add_user_text("three more words")
    assert conversation.context_remaining() == base - 3, (
        "text appended since the response is priced by the pinned tokenizer"
    )

    cost = conversation.image_tokens(str(image))
    conversation.add_user_image(str(image))
    assert conversation.context_remaining() == base - 3 - cost, (
        "what a screenshot costs is exactly what adding it takes away"
    )


def test_appended_text_is_counted_once_in_the_anchored_refusal(tmp_path):
    """Text appended after a response is priced exactly once by the anchored
    delta. The numbers bracket the boundary: counted once, the 10-token task
    fits exactly under the ceiling a double count would breach, and one word
    past the true boundary refuses."""
    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        card=_tiny_card(180),
        chunks=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "usage": {
                    "prompt_tokens": 90,
                    "completion_tokens": 10,
                    "total_tokens": 100,
                }
            },
        ],
    )
    conversation.add_user_text("q")
    conversation.get_response()

    # anchored 100 + task 10 + max_tokens 64 = 174 <= 180; double-counted the
    # task it would be 184 > 180 and refuse.
    conversation.add_user_text("a b c d e f g h i j")
    conversation.get_response()

    # anchor re-set to 100 by that response; 100 + 17 + 64 = 181 > 180.
    conversation.add_user_text(" ".join(["w"] * 17))
    with pytest.raises(ValueError, match="refused before send"):
        conversation.get_response()


def _flattened(messages):
    """Every request part in wire order, so one request's parts can be compared against another's as sequences."""
    parts = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append((message["role"], content))
        else:
            parts.extend((message["role"], json.dumps(part)) for part in content)
    return parts


def test_each_local_request_extends_the_last(tmp_path):
    """The local server reuses a prefix only as a whole exact-prompt snapshot, so a later request pays prefill only for what it appended and nothing else. The conversation only ever grows — every part appended ahead of the call that sends it stays — so each request's token sequence is a strict extension of the last one's."""
    from types import SimpleNamespace

    sent = []

    def create(**params):
        sent.append(_flattened(params["messages"]))
        return SimpleNamespace(
            headers={"x-locus-request": "30fd3652c349"},
            parse=lambda: _FakeStream(
                [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "drafted"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                    {
                        "usage": {
                            "prompt_tokens": 90,
                            "completion_tokens": 10,
                            "total_tokens": 100,
                        }
                    },
                ]
            ),
        )

    conversation = OpenAICompatConversation(
        "sys", card=_tiny_card(4000), record_dir=tmp_path / "record"
    )
    conversation.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=create)
            )
        )
    )

    conversation.add_user_text("the evidence")
    conversation.add_user_text("the task, riding last")
    conversation.get_response()
    conversation.add_user_text("now repair the citations")
    conversation.get_response()

    assert sent[1][: len(sent[0])] == sent[0], (
        "the later request is the earlier one plus what was appended, part for part"
    )
    assert ("user", json.dumps({"type": "text", "text": "the task, riding last"})) in (
        sent[1]
    ), "delivered task text is history the later request carries"


def test_wire_anchored_used_prices_the_delta_since_the_response(tmp_path):
    from locus.analysis.model.protocol import ToolCall

    class Anchored(RecordedConversation):
        def _usage_total(self, usage_metadata):
            return usage_metadata.get("total_tokens")

        def _delta_text_tokens(self, texts):
            return sum(len(text.split()) for text in texts)

    conversation = Anchored(
        responses=[
            Response(
                text="",
                usage_metadata={"total_tokens": 500},
                tool_calls=[
                    ToolCall(id="c1", name="request_screenshots", arguments={})
                ],
            )
        ],
        record_dir=tmp_path,
    )
    conversation.add_user_text("q")
    conversation.get_response()
    assert conversation._wire_anchored_used() == 500
    conversation.add_tool_result("c1", "two words")
    conversation.add_user_text("screenshot at 00:01:")
    conversation.add_user_image("/nowhere/f.png")
    assert conversation._wire_anchored_used() == 500 + 2 + 3 + 1, (
        "tool-result text and served screenshots are the priced delta"
    )


def test_local_stream_error_event_fails_loud(tmp_path):
    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        chunks=[
            {"choices": [{"index": 0, "delta": {"content": "partial"}}]},
            {"error": "Metal OOM"},
        ],
    )
    conversation.add_user_text("q")
    with pytest.raises(RuntimeError, match="server error mid-stream.*Metal OOM"):
        conversation.get_response()


def test_local_stream_parses_structured_output_client_side(tmp_path):
    from pydantic import BaseModel

    class Verdict(BaseModel):
        ok: bool

    captured = {}
    conversation = _streaming_conversation(
        record_dir=tmp_path / "r",
        captured=captured,
        chunks=[
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": '{"ok": '}}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": "true}"}, "finish_reason": "stop"}
                ]
            },
        ],
    )
    conversation.add_user_text("q")

    response = conversation.get_response(response_schema=Verdict)
    assert response.parsed == Verdict(ok=True)
    assert captured["response_format"]["type"] == "json_schema"


def test_a_reply_failing_schema_validation_still_stands_in_the_record(tmp_path):
    from pydantic import BaseModel

    class Verdict(BaseModel):
        ok: bool

    conversation = _streaming_conversation(
        record_dir=tmp_path,
        chunks=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": '{"ok": "not json'},
                        "finish_reason": "stop",
                    }
                ]
            },
        ],
    )
    conversation.add_user_text("q")
    with pytest.raises(ValueError, match="no parseable structured output"):
        conversation.get_response(response_schema=Verdict, label="verdict")
    persisted = (tmp_path / "1_verdict_response.txt").read_text()
    assert '{"ok": "not json' in persisted, (
        "the wire call succeeded, so the reply belongs in the record"
    )


def test_persisted_meta_is_the_complete_config_record(tmp_path):
    card = load_card("qwen3.6-35b-a3b-6bit")
    conversation = OpenAICompatConversation(
        "sys", card=card, dry_run=True, record_dir=tmp_path
    )
    conversation.add_user_text("q")
    with pytest.raises(DryRun):
        conversation.get_response(label="probe")
    config = json.loads((tmp_path / "1_probe_meta.json").read_text())["config"]
    for key, value in card["sampling"].items():
        assert config[key] == value, f"{key} absent from the artifact"
    assert config["model"] == card["model"]
    assert config["base_url"] == card["base_url"]
    assert config["enable_thinking"] is True
    assert config["thinking_budget"] == card["thinking_budget"]
    assert config["max_tokens"] == card["max_tokens"]


def _cache_with(tmp_path, rows, key_fingerprint="fp-test"):
    cache = UploadCache(tmp_path / "cache.db", key_fingerprint)
    with cache._connect() as conn:
        for name, ts_s in rows:
            conn.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cache.key_fp,
                    f"hash-{name}",
                    f"uri-{name}",
                    "image/png",
                    f"files/{name}",
                    1000,
                    int(ts_s * 1000),
                ),
            )
    return cache


def test_upload_cache_ttl_and_protections(tmp_path):
    now = time.time()
    cache = _cache_with(
        tmp_path,
        [
            ("fresh", now - 60),
            ("aging", now - PROTECTED_AGE_S - 60),
            ("stale_read", now - CACHE_READ_TTL_S - 60),
            ("expired", now - GEMINI_FILE_TTL_S - 60),
        ],
    )

    assert cache.get("hash-fresh")["uri"] == "uri-fresh"
    assert cache.get("hash-stale_read") is None

    deletable = [r["file_name"] for r in cache.deletable()]
    assert deletable == ["files/stale_read", "files/aging"], (
        "a file under the protected age is never a reclaim candidate"
    )

    cache.prune_expired()
    assert cache.live_bytes() == 3000

    cache.forget(["files/aging"])
    assert [r["file_name"] for r in cache.deletable()] == ["files/stale_read"]


def test_upload_cache_is_scoped_to_the_api_key(tmp_path):
    """A Files API file is readable only through the key that uploaded it, so a rotated key must never be served another key's URIs — its reads miss, its storage accounting starts at zero, and reclaim never offers it files its credential cannot delete."""
    now = time.time()
    original = _cache_with(
        tmp_path, [("fresh", now - 60), ("aging", now - PROTECTED_AGE_S - 60)]
    )
    rotated = UploadCache(tmp_path / "cache.db", "fp-rotated")

    assert original.get("hash-fresh")["uri"] == "uri-fresh"
    assert rotated.get("hash-fresh") is None
    assert rotated.live_bytes() == 0
    assert rotated.deletable() == []
    assert original.live_bytes() == 2000

    rotated.put("hash-fresh", "uri-fresh-2", "image/png", "files/fresh-2", 500)
    assert rotated.get("hash-fresh")["uri"] == "uri-fresh-2"
    assert original.get("hash-fresh")["uri"] == "uri-fresh", (
        "each key keeps its own row for the same bytes"
    )


def test_upload_cache_rebuilds_a_pre_key_scoped_file_loudly(tmp_path, capsys):
    """Rows written before key scoping name no owner, so they cannot be claimed by any key — the cache rebuilds empty and says so."""
    import sqlite3 as _sqlite3

    path = tmp_path / "cache.db"
    conn = _sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE uploads (content_hash TEXT PRIMARY KEY, uri TEXT NOT NULL, "
        "mime_type TEXT NOT NULL, file_name TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
        "uploaded_at REAL NOT NULL)"
    )
    conn.execute(
        "INSERT INTO uploads VALUES ('h', 'u', 'image/png', 'files/h', 1, ?)",
        (int(time.time() * 1000),),
    )
    conn.commit()
    conn.close()

    cache = UploadCache(path, "fp-test")
    assert cache.get("h") is None
    assert "rebuilt empty" in capsys.readouterr().err


def test_gemini_rejects_bad_resolution(tmp_path):
    from locus.analysis.model.gemini import GeminiConversation

    with pytest.raises(ValueError):
        GeminiConversation(
            GEMINI,
            "sys",
            media_resolution={"pulled_screenshots": "ultra"},
            cache_path=tmp_path / "cache.db",
            record_dir=tmp_path / "record",
        )
    with pytest.raises(ValueError):
        GeminiConversation(
            GEMINI,
            "sys",
            media_resolution={"default": "high"},
            cache_path=tmp_path / "cache.db",
            record_dir=tmp_path / "record",
        )


def test_remote_storage_is_bounded_on_the_path_that_grows_it(tmp_path, monkeypatch):
    # Nothing else deletes a remote file, so if an upload does not reclaim, the ~19GiB bound is a
    # sentence in a docstring and the Files API quota is what actually stops the next pass.
    from types import SimpleNamespace

    from locus.analysis.model import gemini
    from locus.analysis.model.gemini import GeminiConversation

    now = time.time()
    monkeypatch.setattr(gemini, "MAX_STORAGE_BYTES", 1500)
    conversation = GeminiConversation(
        GEMINI, "sys", cache_path=tmp_path / "cache.db", record_dir=tmp_path / "record"
    )
    conversation.cache = _cache_with(
        tmp_path,
        [
            ("old", now - PROTECTED_AGE_S - 60),
            ("recent", now - 60),
        ],
    )

    deleted, uploaded = [], tmp_path / "screenshot.png"
    uploaded.write_bytes(b"\x89PNG")
    conversation.client = SimpleNamespace(
        files=SimpleNamespace(
            delete=lambda name: deleted.append(name),
            upload=lambda file: SimpleNamespace(
                state="ACTIVE", uri="uri-new", mime_type="image/png", name="files/new"
            ),
        )
    )

    conversation.add_user_image(str(uploaded))

    assert deleted == ["files/old"], "the reclaim runs, oldest first, sparing the young"
    assert conversation.cache.get("hash-old") is None
    assert (
        conversation.cache.get(GeminiConversation._content_hash(str(uploaded)))["uri"]
        == "uri-new"
    )


def _gemini_with_uploads(tmp_path, upload):
    from types import SimpleNamespace

    from locus.analysis.model.gemini import GeminiConversation

    conversation = GeminiConversation(
        GEMINI, "sys", cache_path=tmp_path / "cache.db", record_dir=tmp_path / "record"
    )
    conversation.client = SimpleNamespace(files=SimpleNamespace(upload=upload))
    return conversation


def _active_file(file):
    from types import SimpleNamespace

    return SimpleNamespace(
        state="ACTIVE",
        uri=f"uri-{Path(file).name}",
        mime_type="image/png",
        name=f"files/{Path(file).name}",
    )


def _interaction(
    text="",
    usage=None,
    thoughts=None,
    tool_calls=(),
    status=None,
    interaction_id="v1_test",
):
    """A reply as the real SDK model parses it — the fake returns actual
    Interaction objects, so the backend's step-walking is pinned against the
    installed SDK's shapes, not a hand-rolled double."""
    from google.genai import interactions as gi

    steps = []
    if thoughts:
        steps.append(
            {
                "type": "thought",
                "signature": "c2ln",
                "summary": [{"type": "text", "text": thoughts}],
            }
        )
    for call in tool_calls:
        steps.append(
            {
                "type": "function_call",
                "id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
            }
        )
    if text:
        steps.append(
            {"type": "model_output", "content": [{"type": "text", "text": text}]}
        )
    if status is None:
        status = "requires_action" if tool_calls else "completed"
    return gi.Interaction.model_validate(
        {"status": status, "id": interaction_id, "usage": usage or {}, "steps": steps}
    )


def _interactions_conversation(tmp_path, replies, wire_log=None, **kwargs):
    from types import SimpleNamespace

    from locus.analysis.model.gemini import GeminiConversation

    conversation = GeminiConversation(
        GEMINI,
        "sys",
        cache_path=tmp_path / "cache.db",
        record_dir=tmp_path / "payloads",
        **kwargs,
    )

    def create(**request):
        if wire_log is not None:
            wire_log.append(request)
        return replies.pop(0)

    conversation.client = SimpleNamespace(
        interactions=SimpleNamespace(create=create),
        files=SimpleNamespace(upload=lambda file: _active_file(file)),
    )
    return conversation


def test_exhausted_thinking_retry_is_on_the_record(tmp_path):
    """The retry is a second paid call: the record must carry both attempts'
    spend, and the recorded config must be the configuration that actually
    produced the reply — minimal thinking, not the declared level it fell
    back from. The burned interaction is never chained from. The burned
    attempt's usage is the live-measured overrun shape: the status is the
    signal, and output tokens are nonzero — the model leaks a fragment before
    the cap lands."""
    wire = []
    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction(
                usage={
                    "total_input_tokens": 53,
                    "total_thought_tokens": 192,
                    "total_output_tokens": 4,
                    "total_tokens": 249,
                },
                status="incomplete",
                interaction_id="v1_burned",
            ),
            _interaction(
                "the answer",
                usage={
                    "total_input_tokens": 100,
                    "total_output_tokens": 40,
                    "total_tokens": 140,
                },
                interaction_id="v1_good",
            ),
        ],
        wire_log=wire,
        thinking_level="high",
    )
    conversation.add_user_text("q")
    response = conversation.get_response(label="answer")

    assert response.text == "the answer"
    assert wire[0]["generation_config"]["thinking_level"] == "high"
    assert wire[1]["generation_config"]["thinking_level"] == thinking_levels(GEMINI)[0]
    assert wire[1]["generation_config"]["thinking_summaries"] == "none"
    assert "previous_interaction_id" not in wire[1], (
        "the retry re-runs against the same previous state, never the burned attempt"
    )
    assert conversation._interaction_id == "v1_good"
    first = response.usage_metadata["exhausted_thinking_retry"]
    assert first["thinking_level"] == "high"
    assert first["usage"]["total_thought_tokens"] == 192, (
        "the burned first attempt's spend is on the record"
    )

    meta = json.loads((tmp_path / "payloads" / "1_answer_meta.json").read_text())
    assert meta["config"]["thinking_level"] == thinking_levels(GEMINI)[0], (
        "the recorded config is the one that produced the reply"
    )
    assert meta["config"]["interaction_id"] == "v1_good"
    assert (
        meta["usage"]["exhausted_thinking_retry"]["usage"]["total_thought_tokens"]
        == 192
    )


def test_unusable_reply_lands_its_reported_usage(tmp_path, spend_isolation):
    """A call that fails after the wire reported usage still spent: the raise
    carries the usage out, the meta records it, and the ledger gets the entry
    — an error must never read as less spend. A 400/500 reports nothing and
    records nothing; that path stays entryless."""
    from locus.analysis.model.protocol import UnusableReply
    from locus.analysis.spend import read_ledger

    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction(
                usage={
                    "total_input_tokens": 53,
                    "total_thought_tokens": 192,
                    "total_output_tokens": 4,
                    "total_tokens": 249,
                },
                status="incomplete",
                interaction_id="v1_cut",
            ),
        ],
        thinking_level=thinking_levels(GEMINI)[0],
    )
    conversation.add_user_text("q")
    with pytest.raises(UnusableReply, match="incomplete"):
        conversation.get_response(label="answer")

    (entry,) = read_ledger(spend_isolation)
    assert (
        entry["input_tokens"],
        entry["output_tokens"],
        entry["thoughts_tokens"],
    ) == (53, 4, 192)
    assert entry["usd"] > 0
    meta = json.loads((tmp_path / "payloads" / "1_answer_meta.json").read_text())
    assert meta["usage"]["total_thought_tokens"] == 192
    assert "incomplete" in meta["error"]


def test_unusable_retry_folds_the_burned_attempt_into_the_ledger(
    tmp_path, spend_isolation
):
    """When the exhausted-thinking retry itself ends unusable, both attempts'
    reported usage rides the raise: one ledger entry carries the whole spend."""
    from locus.analysis.model.protocol import UnusableReply
    from locus.analysis.spend import read_ledger

    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction(
                usage={
                    "total_input_tokens": 53,
                    "total_thought_tokens": 95,
                    "total_output_tokens": 1,
                    "total_tokens": 149,
                },
                status="incomplete",
                interaction_id="v1_burned",
            ),
            _interaction(
                usage={
                    "total_input_tokens": 53,
                    "total_thought_tokens": 0,
                    "total_output_tokens": 90,
                    "total_tokens": 143,
                },
                status="incomplete",
                interaction_id="v1_also_cut",
            ),
        ],
        thinking_level="high",
    )
    conversation.add_user_text("q")
    with pytest.raises(UnusableReply, match="v1_also_cut"):
        conversation.get_response(label="answer")

    (entry,) = read_ledger(spend_isolation)
    assert (
        entry["input_tokens"],
        entry["output_tokens"],
        entry["thoughts_tokens"],
    ) == (106, 91, 95)


def test_gemini_context_left_anchors_on_the_wire_usage(tmp_path, monkeypatch):
    from locus.analysis.model import gemini
    from locus.analysis.model.cards import context_tokens
    from locus.analysis.model.gemini import image_tier_tokens

    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction(
                "seen",
                usage={
                    "total_input_tokens": 4000,
                    "total_output_tokens": 1000,
                    "total_tokens": 5000,
                },
            )
        ],
    )

    conversation.add_user_text("q")
    with pytest.raises(RuntimeError, match="no response has been received"):
        conversation.context_remaining()

    conversation.get_response()
    base = context_tokens(GEMINI) - 5000
    assert conversation.context_remaining() == base, (
        "the anchor is the wire's total_tokens, not a part walk"
    )

    monkeypatch.setattr(
        gemini,
        "count_text_tokens",
        lambda model, texts: sum(len(t.split()) for t in texts),
    )
    conversation.add_user_text("two words")
    assert conversation.context_remaining() == base - 2

    screenshot = tmp_path / "screenshot.png"
    screenshot.write_bytes(b"\x89PNG")
    conversation.add_user_image(str(screenshot))
    assert conversation.context_remaining() == base - 2 - image_tier_tokens(
        conversation.media_resolution.get("pulled_screenshots")
    ), (
        "a served screenshot is priced at the pulled_screenshots tier, flat whatever its pixels"
    )


def test_gemini_turns_chain_statefully_and_send_only_the_delta(tmp_path):
    """store rides every create; the first turn opens the chain, each later
    turn names the previous interaction and carries only what it adds — the
    server holds the rest."""
    wire = []
    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction("first", usage={"total_tokens": 10}, interaction_id="v1_a"),
            _interaction("second", usage={"total_tokens": 20}, interaction_id="v1_b"),
        ],
        wire_log=wire,
    )
    conversation.add_user_text("opening evidence")
    conversation.get_response(label="answer")
    conversation.add_user_text("follow-up")
    conversation.get_response(label="answer")

    assert wire[0]["store"] is True and wire[1]["store"] is True
    assert "previous_interaction_id" not in wire[0]
    assert wire[1]["previous_interaction_id"] == "v1_a"
    assert conversation._interaction_id == "v1_b"
    texts = [item["text"] for step in wire[1]["input"] for item in step["content"]]
    assert texts == ["follow-up"], "a later turn's wire input is the delta alone"
    assert wire[1]["system_instruction"] == "sys"


def test_delivered_content_is_history_on_the_stateful_backend(tmp_path):
    """Content a successful call carried is history — the server holds it, and the next turn's persisted input shows it: the record is what the model sees."""
    wire = []
    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction("first", usage={"total_tokens": 10}),
            _interaction("second", usage={"total_tokens": 20}),
        ],
        wire_log=wire,
    )
    conversation.add_user_text("evidence")
    conversation.add_user_text("the task, riding last")
    conversation.get_response(label="answer")
    conversation.add_user_text("follow-up")
    conversation.get_response(label="answer")

    texts = [item["text"] for step in wire[0]["input"] for item in step["content"]]
    assert texts == ["evidence", "the task, riding last"]
    second_input = (tmp_path / "payloads" / "2_answer_input.txt").read_text()
    assert "the task, riding last" in second_input, (
        "delivered content stays in the record because it stays in the state"
    )


def test_a_failed_call_leaves_its_content_pending(tmp_path):
    """A failed call advanced no server state, so the content it would have carried stays pending and the next call sends it — nothing is silently lost, nothing double-sent."""
    from types import SimpleNamespace

    from locus.analysis.model.gemini import GeminiConversation

    conversation = GeminiConversation(
        GEMINI,
        "sys",
        cache_path=tmp_path / "cache.db",
        record_dir=tmp_path / "payloads",
    )

    def explode(**request):
        raise RuntimeError("wire down")

    conversation.client = SimpleNamespace(interactions=SimpleNamespace(create=explode))
    conversation.add_user_text("evidence")
    with pytest.raises(RuntimeError, match="wire down"):
        conversation.get_response(label="answer")
    assert [
        item["text"] for step in conversation._pending for item in step["content"]
    ] == ["evidence"], "the failed call's content is still pending for the next attempt"


def test_gemini_images_ride_their_declared_tiers(tmp_path):
    """A pushed screenshot rides the card's activity_screenshots tier, a pull the
    pulled_screenshots tier, each named per part on the wire — and a card
    declaring no activity_screenshots tier sends pushed screenshots at the pulled_screenshots
    tier."""
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_bytes(b"\x89PNG")
    wire = []
    conversation = _interactions_conversation(
        tmp_path,
        [_interaction("ok", usage={"total_tokens": 5})],
        wire_log=wire,
        media_resolution={
            "activity_screenshots": "medium",
            "pulled_screenshots": "high",
        },
    )
    conversation.add_user_text("evidence")
    conversation.add_user_image(str(screenshot), pushed=True)
    conversation.add_user_image(str(screenshot))
    conversation.get_response(label="answer")

    items = wire[0]["input"][0]["content"]
    assert [item.get("resolution") for item in items] == [None, "medium", "high"]

    bare = _interactions_conversation(
        tmp_path / "bare",
        [_interaction("ok", usage={"total_tokens": 5})],
        wire_log=(bare_wire := []),
        media_resolution={"pulled_screenshots": "high"},
    )
    bare.add_user_image(str(screenshot), pushed=True)
    bare.get_response(label="answer")
    assert bare_wire[0]["input"][0]["content"][0]["resolution"] == "high"


def test_gemini_request_carries_the_full_declared_config(tmp_path):
    """Tools as function declarations, structured output as a schema'd text
    response_format with propertyOrdering stamped."""
    from pydantic import BaseModel

    class Verdict(BaseModel):
        ok: bool
        why: str

    wire = []
    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction(
                tool_calls=[
                    {
                        "id": "fc1",
                        "name": "request_screenshots",
                        "arguments": {"timestamps": [5]},
                    }
                ],
                usage={"total_tokens": 5},
            ),
            _interaction('{"ok": true, "why": "fine"}', usage={"total_tokens": 6}),
        ],
        wire_log=wire,
        max_output_tokens=32768,
    )
    conversation.add_user_text("q")
    tool = {
        "name": "request_screenshots",
        "description": "d",
        "parameters": {"type": "object"},
    }
    response = conversation.get_response(label="answer", tools=[tool])
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("fc1", "request_screenshots", {"timestamps": [5]})
    ]

    assert wire[0]["tools"] == [
        {
            "type": "function",
            "name": "request_screenshots",
            "description": "d",
            "parameters": {"type": "object"},
        }
    ]
    assert wire[0]["generation_config"]["max_output_tokens"] == 32768

    conversation.add_tool_result("fc1", "rendered 1 screenshot")
    conversation.get_response(label="verdict", response_schema=Verdict)
    result_step = wire[1]["input"][0]
    assert result_step == {
        "type": "function_result",
        "call_id": "fc1",
        "name": "request_screenshots",
        "result": "rendered 1 screenshot",
    }
    response_format = wire[1]["response_format"]
    assert response_format["type"] == "text"
    assert response_format["mime_type"] == "application/json"
    assert response_format["schema"]["propertyOrdering"] == ["ok", "why"]


def test_gemini_requests_parse_as_the_sdk_request_model(tmp_path):
    """Every request the backend builds must be a valid CreateModelInteraction
    by the installed SDK's own pydantic model — the offline pin that the shapes
    the backend emits are the shapes the wire accepts."""
    from google.genai._gaos.types.interactions.createmodelinteraction import (
        CreateModelInteraction,
    )

    screenshot = tmp_path / "screenshot.png"
    screenshot.write_bytes(b"\x89PNG")
    wire = []
    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction(
                tool_calls=[
                    {
                        "id": "fc1",
                        "name": "request_screenshots",
                        "arguments": {"timestamps": [1]},
                    }
                ],
                usage={"total_tokens": 5},
                interaction_id="v1_a",
            ),
            _interaction("done", usage={"total_tokens": 6}),
        ],
        wire_log=wire,
        media_resolution={
            "activity_screenshots": "medium",
            "pulled_screenshots": "high",
        },
        thinking_level="medium",
        max_output_tokens=32768,
    )
    conversation.add_user_text("evidence")
    conversation.add_user_image(str(screenshot), pushed=True)
    tool = {
        "name": "request_screenshots",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    }
    conversation.add_user_text("the task")
    conversation.get_response(label="answer", tools=[tool])
    conversation.add_tool_result("fc1", "rendered")
    conversation.add_user_image(str(screenshot))
    conversation.add_user_text("the task")
    conversation.get_response(label="answer", tools=[tool])

    for request in wire:
        parsed = CreateModelInteraction.model_validate(request)
        assert parsed.store is True


def test_gemini_thoughts_are_read_from_thought_steps(tmp_path):
    conversation = _interactions_conversation(
        tmp_path,
        [_interaction("the answer", thoughts="let me look", usage={"total_tokens": 9})],
    )
    conversation.add_user_text("q")
    response = conversation.get_response(label="answer")
    assert response.thoughts_text == "let me look"
    persisted = (tmp_path / "payloads" / "1_answer_response.txt").read_text()
    assert "=== thoughts ===\nlet me look" in persisted


def test_a_failed_interaction_status_is_loud(tmp_path):
    conversation = _interactions_conversation(
        tmp_path,
        [
            _interaction(
                "partial",
                usage={"total_output_tokens": 3, "total_tokens": 5},
                status="failed",
            )
        ],
    )
    conversation.add_user_text("q")
    with pytest.raises(RuntimeError, match="failed"):
        conversation.get_response(label="answer")


def test_prefetch_is_a_noop_where_images_ride_as_paths(tmp_path):
    conversation = OpenAICompatConversation(
        "sys", card=load_card("qwen3.6-35b-a3b-6bit"), record_dir=tmp_path / "record"
    )
    # A path backend has nothing to move; the path is not even touched.
    conversation.prefetch_images(["/nowhere/screenshot.png"])


def test_prefetch_uploads_misses_concurrently_and_adds_hit_cache(tmp_path):
    import threading

    screenshots = []
    for i in range(3):
        screenshot = tmp_path / f"image_{i}.png"
        screenshot.write_bytes(b"\x89PNG" + bytes([i]))
        screenshots.append(str(screenshot))

    in_flight_together = threading.Barrier(3, timeout=10)
    uploaded = []

    def upload(file):
        in_flight_together.wait()  # serial uploads would hang here and break the barrier
        uploaded.append(file)
        return _active_file(file)

    conversation = _gemini_with_uploads(tmp_path, upload)
    conversation.prefetch_images(screenshots)
    assert sorted(Path(f).name for f in uploaded) == [
        "image_0.png",
        "image_1.png",
        "image_2.png",
    ]

    def refuse(file):
        raise AssertionError(f"prefetched screenshot re-uploaded: {file}")

    conversation.client.files.upload = refuse
    for screenshot in screenshots:
        conversation.add_user_image(screenshot)
    record = [entry for entry in conversation._record if entry[0] == "image"]
    assert [entry[2] for entry in record] == [
        "uri-image_0.png",
        "uri-image_1.png",
        "uri-image_2.png",
    ], "the payload record carries the remote reference actually sent"


def test_prefetch_skips_cached_images_and_identical_bytes(tmp_path):
    already = tmp_path / "already.png"
    already.write_bytes(b"seen-before")
    twin_a = tmp_path / "twin_a.png"
    twin_b = tmp_path / "twin_b.png"
    twin_a.write_bytes(b"same-bytes")
    twin_b.write_bytes(b"same-bytes")

    uploaded = []

    def upload(file):
        uploaded.append(file)
        return _active_file(file)

    conversation = _gemini_with_uploads(tmp_path, upload)
    conversation.add_user_image(str(already))
    assert len(uploaded) == 1

    conversation.prefetch_images([str(already), str(twin_a), str(twin_b)])
    assert len(uploaded) == 2, (
        "one upload for the twins, none for the cached screenshot"
    )


def test_prefetch_failure_is_loud_and_successes_stay_cached(tmp_path):
    healthy_a = tmp_path / "healthy_a.png"
    healthy_b = tmp_path / "healthy_b.png"
    poisoned = tmp_path / "poisoned.png"
    for i, screenshot in enumerate((healthy_a, healthy_b, poisoned)):
        screenshot.write_bytes(b"\x89PNG" + bytes([i]))

    def upload(file):
        if "poisoned" in file:
            raise RuntimeError("wire failure on poisoned.png")
        return _active_file(file)

    conversation = _gemini_with_uploads(tmp_path, upload)
    with pytest.raises(RuntimeError, match="poisoned"):
        conversation.prefetch_images([str(healthy_a), str(healthy_b), str(poisoned)])

    def refuse(file):
        raise AssertionError(f"healthy screenshot re-uploaded after failure: {file}")

    conversation.client.files.upload = refuse
    conversation.add_user_image(str(healthy_a))
    conversation.add_user_image(str(healthy_b))


def test_prefetch_reclaims_against_the_batch_it_is_about_to_land(tmp_path, monkeypatch):
    # live_bytes alone sits under the ceiling; only counting the batch's own
    # pending bytes pushes past it, so this fails if prefetch reclaims blind.
    from types import SimpleNamespace

    from locus.analysis.model import gemini

    now = time.time()
    monkeypatch.setattr(gemini, "MAX_STORAGE_BYTES", 1500)
    screenshot = tmp_path / "big.png"
    screenshot.write_bytes(b"x" * 600)

    deleted = []
    conversation = _gemini_with_uploads(tmp_path / "conv", _active_file)
    conversation.cache = _cache_with(
        tmp_path,
        [
            ("old", now - PROTECTED_AGE_S - 60),
        ],
    )
    conversation.client.files = SimpleNamespace(
        upload=_active_file, delete=lambda name: deleted.append(name)
    )

    conversation.prefetch_images([str(screenshot)])
    assert deleted == ["files/old"]


def test_schema_with_unparsed_response_raises_after_persisting(tmp_path):
    conversation = RecordedConversation(
        [Response(text="155KB of runaway thinking, no JSON", parsed=None)],
        record_dir=tmp_path,
    )
    conversation.add_user_text("judge these claims")

    with pytest.raises(ValueError, match="no parseable structured output"):
        conversation.get_response(response_schema=dict, label="verdicts")

    persisted = (tmp_path / "1_verdicts_response.txt").read_text()
    assert "runaway thinking" in persisted


def test_a_model_bound_backend_cannot_be_built_without_a_record_home(tmp_path):
    """The record is a bundle with the output: the motion that creates a
    conversation states where its transcript lands, and no shape exists where a
    model-bound conversation runs unrecorded."""
    with pytest.raises(TypeError, match="record_dir"):
        OpenAICompatConversation(
            "sys", card=load_card("qwen3.6-35b-a3b-6bit"), dry_run=True
        )
    from locus.analysis.model.gemini import GeminiConversation

    with pytest.raises(TypeError, match="record_dir"):
        GeminiConversation(
            GEMINI, "sys", dry_run=True, cache_path=tmp_path / "cache.db"
        )


def test_response_text_inverts_the_persisted_response_format():
    from locus.analysis.model.protocol import response_text

    assert response_text("just the answer\n") == "just the answer"
    recorded = "=== thoughts ===\nhmm let me think\n\n=== response ===\nthe actual answer\nsecond line\n"
    assert response_text(recorded) == "the actual answer\nsecond line"


def test_sdk_retry_waits_narrate_to_stderr_once(capsys):
    # The SDK announces each backoff wait only on its own logger; unhandled,
    # a minutes-long 429 backoff reads as a hang to the agent watching the
    # call. Narration routes those records to stderr, and asserting it twice
    # proves idempotence — one handler however many conversations construct.
    import logging

    from locus.analysis.model.gemini import _narrate_retries

    logger = logging.getLogger("google_genai._api_client")
    for handler in list(logger.handlers):
        if getattr(handler, "_locus_retry_narration", False):
            logger.removeHandler(handler)
    before = list(logger.handlers)
    try:
        _narrate_retries()
        _narrate_retries()
        added = [h for h in logger.handlers if h not in before]
        assert len(added) == 1, "idempotent — one narration handler"
        logger.info("Retrying in 32.0 seconds as it raised ...")
        assert "Retrying in 32.0 seconds" in capsys.readouterr().err
    finally:
        for handler in list(logger.handlers):
            if handler not in before:
                logger.removeHandler(handler)


def test_measured_serving_reads_the_medians_of_this_models_recent_calls(tmp_path):
    """The pace a local price quotes is measurement: the medians of the last N completed calls' server timings for this model under analyses/ — another model's calls, a failed call, and a call with no timings are not this model's pace — and None until a call has completed here."""
    import json

    from locus.analysis.model.openai_compat import measured_serving

    def meta(
        dirname, call, *, model, started, pps, n, served, error=None, timings=True
    ):
        d = tmp_path / dirname
        d.mkdir(exist_ok=True)
        usage = {"prompt_tokens": 1}
        if timings:
            usage["timings"] = {
                "prompt_per_second": pps,
                "predicted_per_second": 30,
                "predicted_n": n,
                "prompt_ms": served * 1000 / 2,
                "predicted_ms": served * 1000 / 2,
            }
        (d / f"{call}_answer_meta.json").write_text(
            json.dumps(
                {
                    "call": call,
                    "config": {"model": model},
                    "usage": usage,
                    "started": started,
                    "elapsed_s": served * 6,
                    "error": error,
                }
            )
        )

    assert measured_serving(tmp_path, "m") is None
    meta("a", 1, model="m", started="2026-09-04T10:00:00Z", pps=300, n=4000, served=900)
    meta(
        "a", 2, model="m", started="2026-09-04T10:20:00Z", pps=400, n=6000, served=1500
    )
    meta("b", 1, model="m", started="2026-09-04T09:00:00Z", pps=100, n=100, served=60)
    meta("c", 1, model="other", started="2026-09-04T11:00:00Z", pps=9, n=9, served=9)
    meta(
        "d",
        1,
        model="m",
        started="2026-09-04T11:00:00Z",
        pps=9,
        n=9,
        served=9,
        error="boom",
    )
    meta(
        "e",
        1,
        model="m",
        started="2026-09-04T11:00:00Z",
        pps=9,
        n=9,
        served=9,
        timings=False,
    )

    served = measured_serving(tmp_path, "m", last=2)
    assert served == {
        "calls": 2,
        "prompt_tokens_per_s": 350,
        "generated_tokens_per_s": 30,
        "generated_tokens_median": 5000,
        "served_s_median": 1200,
        "since": "2026-09-04T10:00:00Z",
    }, (
        "the two most recent of this model's completed calls, by their start — served time is the server's, never the client's wall-clock"
    )
