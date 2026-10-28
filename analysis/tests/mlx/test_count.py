"""The server's own count run in this process (mlx/count.py): mlx-vlm's chat steps over the model's processor and config, with neither the server nor the weights.

The image pins were measured on the live server, each the prompt_tokens one image of that size added to a request, markers included; test_integration.py proves whole requests against the live server. The count is upstream's own code, so a failing pin means upstream's arithmetic moved under the card: re-measure on the server before re-pinning.
"""

import pytest
from locus.analysis import budget
from locus.analysis.mlx import count
from locus.analysis.model.cards import load_card
from PIL import Image

MODEL = load_card("qwen3.6-35b-a3b-6bit")["model"]

SERVER_MEASURED = [
    ((1, 1), 66),
    ((31, 33), 74),
    ((50, 2000), 126),
    ((480, 925), 437),
    ((1045, 587), 596),
    ((1814, 624), 1142),
    ((803, 1529), 1202),
]


def _request(parts, **extra):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": parts},
        ],
        **extra,
    }


@pytest.mark.parametrize("size,expected", SERVER_MEASURED)
def test_an_image_costs_what_the_server_counted(size, expected):
    assert count.image_tokens(MODEL, *size) == expected


def test_an_image_counts_by_its_size_alone(tmp_path):
    """The count reads an image's size, never its pixels, so a blank image of the size a screenshot will have counts as the screenshot."""
    noise, blank = tmp_path / "noise.png", tmp_path / "blank.png"
    Image.effect_noise((733, 411), 64).convert("RGB").save(noise)
    Image.new("RGB", (733, 411)).save(blank)

    def parts(path):
        return [
            {"type": "text", "text": "what is shown?"},
            {"type": "image_url", "image_url": {"url": str(path)}},
        ]

    assert count.prompt_tokens(_request(parts(noise))) == count.prompt_tokens(
        _request(parts(blank))
    )


def test_the_tool_declarations_and_the_thinking_switch_are_counted():
    parts = [{"type": "text", "text": "hello"}]
    tool = {
        "type": "function",
        "function": {
            "name": "look",
            "description": "look again",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    bare = count.prompt_tokens(_request(parts))
    assert count.prompt_tokens(_request(parts, tools=[tool])) > bare
    assert count.prompt_tokens(
        _request(parts, extra_body={"enable_thinking": False})
    ) != count.prompt_tokens(_request(parts, extra_body={"enable_thinking": True}))


def test_image_geometry_realizes_the_per_image_token_ceiling():
    assert budget.image_geometry(MODEL, 390, 663, 600) == (390, 663), (
        "a screenshot under the ceiling passes untouched — never upscaled"
    )
    width, height = budget.image_geometry(MODEL, 1920, 921, 600)
    cost = count.image_tokens(MODEL, width, height)
    assert cost <= 600, "the ceiling bounds everything the image adds"
    assert cost > 540, "the resize lands near the ceiling, not far under it"
    assert abs(width / height - 1920 / 921) < 0.05, "aspect preserved"


def test_a_ceiling_below_the_smallest_image_fails_loud():
    with pytest.raises(ValueError, match="smallest image"):
        budget.image_geometry(MODEL, 1920, 921, 10)


def test_an_unjudged_upstream_refuses_to_count(monkeypatch):
    monkeypatch.setattr(count, "EXPECTED_CHAT_HANDLER_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="upstream drift"):
        count._verify_upstream()


def _all_at_once(request):
    """The server's own preprocessing of the request, every image loaded and processed together — what the per-size count must equal."""
    from mlx_vlm.utils import load_image

    with count._serving(request["model"]):
        model, prompt, sent, options = count._render(request)
        return count._tokens(model, prompt, [load_image(url) for url in sent], options)


def test_mixed_sizes_count_as_the_server_processes_them_together(tmp_path):
    sizes = [(1280, 800), (390, 663), (1280, 800), (733, 2100), (31, 33)]
    parts = [{"type": "text", "text": "look:"}]
    for i, size in enumerate(sizes):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", size).save(path)
        parts += [
            {"type": "text", "text": f"image {i}"},
            {"type": "image_url", "image_url": {"url": str(path)}},
        ]
    request = _request(parts)
    assert count.prompt_tokens(request) == _all_at_once(request)


def test_the_count_runs_under_the_servers_declared_env(tmp_path, monkeypatch):
    """What models.toml declares for the server is what the count runs under — here a pixel cap the server would read from its env — and this process's own env is untouched after."""
    import os

    from locus.analysis.mlx import serve

    monkeypatch.delenv("MLX_VLM_MAX_PIXELS", raising=False)
    path = tmp_path / "big.png"
    Image.new("RGB", (1920, 1080)).save(path)
    request = _request(
        [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": str(path)}},
        ]
    )
    uncapped = count.prompt_tokens(request)

    declared = serve.serving_env
    monkeypatch.setattr(
        serve,
        "serving_env",
        lambda slug: {**declared(slug), "MLX_VLM_MAX_PIXELS": str(256 * 256)},
    )
    capped = count.prompt_tokens(request)
    assert capped < uncapped, "the declared cap shrank the image"
    assert capped == _all_at_once(request)
    assert "MLX_VLM_MAX_PIXELS" not in os.environ


def test_a_model_not_on_this_machine_fails_loud_without_fetching():
    with pytest.raises(RuntimeError, match="not in this machine's Hugging Face cache"):
        count._local_files("nobody-here/not-a-cached-model")


def test_the_models_files_are_found_without_the_network(monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the count reached for the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert (count._local_files(MODEL) / "config.json").exists()


def test_a_count_setting_only_this_shell_holds_is_not_read(tmp_path, monkeypatch):
    """The server will not boot with a count-affecting variable its declaration leaves out, so the count does not read one either: an inherited pixel cap changes nothing, and is still set after."""
    import os

    path = tmp_path / "big.png"
    Image.new("RGB", (1920, 1080)).save(path)
    request = _request(
        [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": str(path)}},
        ]
    )
    monkeypatch.delenv("MLX_VLM_MAX_PIXELS", raising=False)
    declared = count.prompt_tokens(request)
    monkeypatch.setenv("MLX_VLM_MAX_PIXELS", str(256 * 256))
    assert count.prompt_tokens(request) == declared
    assert os.environ["MLX_VLM_MAX_PIXELS"] == str(256 * 256)
