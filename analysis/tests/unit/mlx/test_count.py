"""The server's own count run in this process (mlx/count.py): mlx-vlm's chat steps over the model's processor and config, with neither the server nor the weights.

The image pins were measured on the live server, each the prompt_tokens one image of that size added to a request, markers included; model_server_tests/test_integration.py proves whole requests against the live server. The count is upstream's own code, so a failing pin means upstream's arithmetic moved under the card: re-measure on the server before re-pinning.
"""

import pytest
from locus.analysis import budget
from locus.analysis.mlx import count
from locus.analysis.model.cards import load_card
from PIL import Image

CARD = load_card("qwen3.6-35b-a3b-6bit")
MODEL = CARD["model"]

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


def test_image_geometry_sizes_any_viewport_to_the_per_image_token_value():
    cell = count.grid(MODEL)
    for size, value in [
        ((390, 663), 600),
        ((390, 844), 1200),
        ((1920, 921), 600),
        ((2560, 1323), 1200),
        ((390, 12000), 600),
    ]:
        width, height = budget.image_geometry(MODEL, *size, value)
        cost = count.image_tokens(MODEL, width, height)
        assert cost <= value, "the image's full cost fits the value"
        assert cost > value * 0.9, (
            "the image costs within 10% of the value at any viewport size"
        )
        assert width % cell == 0 and height % cell == 0, "the size lands on the grid"
        assert count.processed_size(MODEL, width, height) == (width, height), (
            "the processor encodes the screenshot at its own size"
        )


def test_the_least_value_gets_the_viewports_smallest_size_the_processor_encodes():
    size = budget.image_geometry(MODEL, 1920, 921, 66)
    assert size == (384, 192)
    assert count.processed_size(MODEL, *size) == size, "drawn as encoded"
    assert count.image_tokens(MODEL, *size) == 74, (
        "the processor enlarges any smaller size at this shape"
    )


def test_the_processor_sizes_by_its_own_rule():
    assert count.grid(MODEL) == 32
    assert count.processed_size(MODEL, 1120, 544) == (1120, 544), "on the grid, as is"
    assert count.processed_size(MODEL, 1000, 500) == (992, 512), (
        "off it, rounded to cells"
    )


# Every cache a count under another pixel limit fills.
SIZING_CACHES = (
    "pixel_limit",
    "_cell_budget",
    "_largest_image",
    "_check_image_tokens",
    "_processed_size",
    "_image_tokens",
)


def _pixel_limit_under(monkeypatch, max_pixels):
    from locus.analysis.mlx import serve

    declared = serve.serving_env
    monkeypatch.setattr(
        serve,
        "serving_env",
        lambda slug: {**declared(slug), "MLX_VLM_MAX_PIXELS": str(max_pixels)},
    )
    for name in SIZING_CACHES:
        getattr(count, name).cache_clear()
    try:
        return count.pixel_limit(MODEL)
    finally:
        for name in SIZING_CACHES:
            getattr(count, name).cache_clear()


def test_the_pixel_limit_is_the_processors_floored_to_whole_cells(monkeypatch):
    assert count.pixel_limit(MODEL) == 4096 * 4096
    assert _pixel_limit_under(monkeypatch, 1300 * 1024 + 500) == 1300 * 1024


def test_a_pixel_limit_under_one_cell_fails_loud(monkeypatch):
    with pytest.raises(ValueError, match="under one 32x32 cell"):
        _pixel_limit_under(monkeypatch, 100)


def test_the_processor_encodes_from_its_smallest_image_to_its_budget():
    assert count._image_token_range(MODEL) == (66, 16386), (
        "a one-cell image enlarged to the processor's least pixels, and 4096x4096 at its most"
    )


def test_gemma4_encodes_up_to_its_largest_trained_budget():
    gemma = load_card("gemma-4-26b-a4b-6bit")["model"]
    assert count.pixel_limit(gemma) == 1120 * 48 * 48
    assert count._image_token_range(gemma) == (3, 1122)


def test_a_card_value_outside_what_the_processor_encodes_is_refused(monkeypatch):
    from locus.analysis.model import cards

    for value in (65, 16387):
        outside = {**CARD["image_tokens"], "pulled_screenshots": value}
        monkeypatch.setattr(
            cards,
            "load_card",
            lambda name, table=outside: {**CARD, "image_tokens": table},
        )
        count._check_image_tokens.cache_clear()
        with pytest.raises(
            ValueError, match="smallest image costs 66 tokens and its largest 16386"
        ):
            count._check_image_tokens(MODEL)
    monkeypatch.setattr(cards, "load_card", lambda name: CARD)
    count._check_image_tokens.cache_clear()
    count._check_image_tokens(MODEL)


def test_an_unjudged_upstream_refuses_to_count(monkeypatch):
    monkeypatch.setattr(count, "EXPECTED_CHAT_HANDLER_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="upstream drift"):
        count._verify_upstream()


def test_a_processor_of_another_familys_module_refuses_to_count():
    with count._serving(MODEL):
        processor, _ = count._resources(MODEL)
    with pytest.raises(SystemExit, match="judged on for family gemma4"):
        count._verify_processor(processor, "gemma4")


def test_an_image_processor_outside_the_judged_module_refuses_to_count():
    """grid and processed_size read the image processor's sizing, so the image processor is judged by its own module, never through the processor holding it."""

    class Unjudged:
        patch_size, merge_size = 16, 2

    with count._serving(MODEL):
        processor, _ = count._resources(MODEL)
    held = processor.image_processor
    processor.image_processor = Unjudged()
    try:
        with pytest.raises(SystemExit, match="Unjudged"):
            count._verify_processor(processor, "qwen3")
    finally:
        processor.image_processor = held


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
    """The count runs under the serving env (serve.serving_env) — here a pixel cap the server would read from its env — and leaves this process's own env untouched."""
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
    assert capped < uncapped, "the serving env's cap shrank the image"
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
    """The server will not boot with a count-affecting variable its serving env leaves out, so the count does not read one either: an inherited pixel cap changes nothing, and is still set after."""
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
