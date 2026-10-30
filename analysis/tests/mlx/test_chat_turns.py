"""The chat-turns patch (mlx/chat_turns.py) over the model's real processor and template, with neither the server nor the weights: a tool message's images, an assistant answer's reasoning and a request's chat_template_kwargs reach the rendering, the count reads them as the server will, and drift in any source the patch rests on refuses before anything is swapped.

The count runs the patched functions in this process (count.py), so its rendering is the server's; test_integration.py proves the handler's half against the live server.
"""

import pytest
from locus.analysis.mlx import chat_turns, count
from locus.analysis.model.cards import load_card
from mlx_vlm.server import openai
from PIL import Image

MODEL = load_card("qwen3.6-35b-a3b-6bit")["model"]
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
REASONING = "the checkout button never received the click"


def _codes():
    return {what: chat_turns._function(what)[2].__code__ for what in chat_turns.PINNED}


def _image(tmp_path, name="pull.png", size=(480, 925)):
    path = tmp_path / name
    Image.new("RGB", size, (40, 70, 220)).save(path)
    return path


def _call():
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "request_screenshots", "arguments": "{}"},
    }


def _pull(tool_content):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what happened?"},
            {"role": "assistant", "content": "", "tool_calls": [_call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        ],
    }


def _answered(**extra):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what happened?"},
            {"role": "assistant", "content": "nothing", "reasoning_content": REASONING},
            {"role": "user", "content": "fix the citation"},
        ],
        **extra,
    }


def _render(request):
    with count._serving(request["model"]):
        _, prompt, sent, _ = count._render(request)
    return prompt, sent


def test_a_tool_replys_image_renders_inside_the_reply(tmp_path):
    path = _image(tmp_path)
    prompt, sent = _render(
        _pull(
            [
                {"type": "text", "text": "screenshot at 00:01.000:"},
                {"type": "image_url", "image_url": {"url": str(path)}},
                {"type": "text", "text": "now answer"},
            ]
        )
    )
    assert sent == [str(path)]
    reply = prompt.split("<tool_response>\n", 1)[1].split("\n</tool_response>", 1)[0]
    assert reply == f"screenshot at 00:01.000:{VISION_BLOCK}now answer"


def test_an_answers_reasoning_renders_under_preserve_thinking_and_only_then():
    preserved, _ = _render(_answered(chat_template_kwargs={"preserve_thinking": True}))
    default, _ = _render(_answered())
    assert f"<think>\n{REASONING}\n</think>" in preserved
    assert REASONING not in default
    assert count.prompt_tokens(
        _answered(chat_template_kwargs={"preserve_thinking": True})
    ) > count.prompt_tokens(_answered())


def test_a_tool_call_turns_reasoning_renders_under_preserve_thinking():
    request = _pull("rendered 0 screenshots")
    request["messages"][2]["reasoning_content"] = REASONING
    request["messages"].append({"role": "user", "content": "answer now"})
    preserved, _ = _render(
        {**request, "chat_template_kwargs": {"preserve_thinking": True}}
    )
    default, _ = _render(request)
    assert f"<think>\n{REASONING}\n</think>" in preserved
    assert REASONING not in default


def test_template_options_arrive_through_extra_body_as_the_client_sends_them():
    via_extra_body, _ = _render(
        _answered(extra_body={"chat_template_kwargs": {"preserve_thinking": True}})
    )
    assert REASONING in via_extra_body


@pytest.mark.parametrize(
    "options,refusal",
    [
        ({"enable_thinking": False}, "fields of the request itself"),
        ({"max_pixels": 1}, "fields of the request itself"),
        ({"add_generation_prompt": False}, "already gives"),
        ({"video": "clip.mp4"}, "already gives"),
        ({"documents": []}, "already gives"),
        ({"preserve_thinkng": True}, "never reads"),
        ({"num_images": 3}, "never reads"),
        (["preserve_thinking"], "must be an object"),
    ],
)
def test_an_option_is_refused_unless_a_variable_the_template_reads_that_nothing_else_sets(
    options, refusal
):
    with pytest.raises(chat_turns.TemplateOptionRefused, match=refusal):
        count.prompt_tokens(_answered(chat_template_kwargs=options))


def test_a_refused_option_is_the_clients_error():
    from fastapi import HTTPException

    def refusing():
        raise chat_turns.TemplateOptionRefused("no")

    with pytest.raises(HTTPException) as raised:
        chat_turns.refused_as_400(refusing)()
    assert raised.value.status_code == 400


def test_installing_again_changes_nothing():
    chat_turns.install_chat_turns()
    before = _codes()
    chat_turns.install_chat_turns()
    assert _codes() == before


def test_the_route_runs_the_rewritten_handler():
    """The route was bound to the handler's function object when mlx_vlm.server was imported; the in-place swap is what makes it run the rewrite."""
    from mlx_vlm.server.app import inference_router

    chat_turns.install_chat_turns()
    endpoints = {
        route.endpoint
        for route in inference_router.routes
        if getattr(route, "path", None) in ("/chat/completions", "/v1/chat/completions")
    }
    assert endpoints == {openai.chat_completions_endpoint}
    assert "refused_as_400" in openai.chat_completions_endpoint.__code__.co_names


@pytest.fixture
def uninstalled(monkeypatch):
    chat_turns.install_chat_turns()
    monkeypatch.setattr(chat_turns, "_installed", False)
    return _codes()


@pytest.mark.parametrize("what", sorted(chat_turns.EXPECTED_SHA256))
def test_refuses_on_upstream_source_drift_and_swaps_nothing(
    monkeypatch, uninstalled, what
):
    monkeypatch.setitem(chat_turns.EXPECTED_SHA256, what, "0" * 64)
    with pytest.raises(SystemExit, match="upstream drift"):
        chat_turns.install_chat_turns()
    assert _codes() == uninstalled
    assert chat_turns._installed is False


@pytest.mark.parametrize("what", sorted(chat_turns.REWRITES))
def test_refuses_when_a_rewritten_line_is_not_there_exactly_once_and_swaps_nothing(
    monkeypatch, uninstalled, what
):
    monkeypatch.setitem(
        chat_turns.REWRITES, what, [("a line upstream never wrote", "anything")]
    )
    with pytest.raises(SystemExit, match="holds 0 copies"):
        chat_turns.install_chat_turns()
    assert _codes() == uninstalled
