"""The bundled server's own prompt count, run in this process without the server.

A chat-completions request's prompt_tokens is decided on the CPU before the GPU sees anything: upstream renders the model's chat template over the request and runs the model's processor over it and its images (server/openai.py chat_completions_endpoint into server/generation.py _cpu_preprocess). prompt_tokens() runs those same steps through upstream's own functions, on the model's processor and config loaded without its weights, under the environment the server serves the model under (serve.serving_env), so a local card's count is the server's count by construction — offline, nothing booted, no copy of the template, the tokenizer or the image arithmetic kept here. Where the server preprocesses a request's images all at once, the count takes each distinct image size once (prompt_tokens says why that is the same number), so what it holds does not grow with the number of images.

The steps follow upstream's chat handler, load and preprocessing, and the shortcut rests on the model's processor, so all are pinned to their source like every upstream-keyed patch (patching.py): a changed source refuses until the steps are re-checked against it.
"""

import functools
import os
from contextlib import contextmanager

from . import require_extra
from .patching import expect_source, refuse

# utils.get_model_path's own file patterns for a model, without the weights: everything its
# processor and config are read from. A copy, so get_model_path is pinned below.
MODEL_FILES = ["*.json", "*.jsonl", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jinja"]

EXPECTED_CHAT_HANDLER_SHA256 = (
    "74c05a1d1cf9b115f6ae23a997feb3202c2853ee972620eb58c32424a7e7dc03"
)
EXPECTED_CPU_PREPROCESS_SHA256 = (
    "987ac63286683d76cab4e84a32cf74a8da572ac2dbe63674092fbe7780e8781b"
)
EXPECTED_LOAD_SHA256 = (
    "de9dfe44dbc77f4ca3086433614a99a9936f2985eea5e26de0360e28f7762509"
)
EXPECTED_LOAD_MODEL_SHA256 = (
    "27e77d26d0b5254ba7ddc8edf5eb6312ad61e46d93b138d04c09db0f9649f6a4"
)
EXPECTED_LOAD_MODEL_RESOURCES_SHA256 = (
    "7474f492b3efce35614b8caf5b3a4980f92ef6df78148128614bdc4feb39934f"
)
EXPECTED_GET_MODEL_PATH_SHA256 = (
    "dd19819ebf50c4484f09d4bc1f8f1e390d72a84cd7bbe0e47cd95e0d75dbc499"
)
EXPECTED_PREPARE_INPUTS_SHA256 = (
    "51cdb24efbcfb28c690f0566f7859d8f7a4c9351b50dc85fec85976e29de5fe7"
)
EXPECTED_GEN_ARGS_SHA256 = (
    "abd877df1489b9295d0efc47ca433672796dc35f637f7cebc78f3220f0c53d5a"
)
EXPECTED_APP_GEN_ARGS_SHA256 = (
    "7860eff939ba38b903bf9573d6d5c8d6d9d62bb59f141a6934648d52680c20b9"
)
EXPECTED_LOAD_IMAGE_SHA256 = (
    "96b3ed78540f0e2bf7719f27166d1ef4cfd25ba6e9d2b3a909a718454e1c9bc9"
)
# Every environment variable the count's upstream steps read that can change a count, by the
# reader it reaches upstream through: the image pixel limits (also in the runtime config), the
# thinking defaults the chat template receives, and the processor-load arguments. A server and a
# count must read the same values for these, so both take them from the declared serving env alone
# (serve.serving_env): the count clears any it inherits, and the server refuses to boot on one.
COUNT_ENV = {
    "MLX_VLM_MIN_PIXELS": "get_server_min_pixels",
    "MLX_VLM_MAX_PIXELS": "get_server_max_pixels",
    "MLX_VLM_ENABLE_THINKING": "get_server_enable_thinking",
    "MLX_VLM_THINKING_BUDGET": "get_server_thinking_budget",
    "MLX_VLM_THINKING_START_TOKEN": "get_server_thinking_start_token",
    "MLX_VLM_THINKING_END_TOKEN": "get_server_thinking_end_token",
    "EXPERT_CACHE_GB": "get_expert_cache_gb",
    "MAX_KV_SIZE": "get_configured_context_limit",
    "MLX_TRUST_REMOTE_CODE": "load_model_resources",
}
EXPECTED_ENV_READER_SHA256 = {
    "get_server_min_pixels": "c28b0479fc1a61b031c6162563e9134adceee7ea370cd477ca69ba8fd8e47b30",
    "get_server_max_pixels": "bbaa31f9af2a4a95ca488ee6ba9017c2742e822721daa8088cf48e40f237ee89",
    "get_server_enable_thinking": "a8519eba3dbb93551c647fd17f5777893cf4c076f4fd9d68e8dc56dc6eb53bfe",
    "get_server_thinking_budget": "8fa8946bed520457a842ce253591e2b7eb3a6f3b1b15e7a2846ce7b16ccac299",
    "get_server_thinking_start_token": "2e57cc45a0dbcd11388545bc24644463a83650ce48d7a5608b1072e977bfd2a3",
    "get_server_thinking_end_token": "c0cda98a9cc96003477ab2331f56a8f4ab8fcf1f70d2e9575fcf6d68899745c2",
    "get_expert_cache_gb": "c59a23d84daef3de94821f2889a29dc921d23d636667948710a8d5e3277562ab",
    "get_configured_context_limit": "e57421c0797ca38297d2687426afd8c0b19f9fc60ca912d058fe02b0b3951cf3",
}
EXPECTED_RUNTIME_FROM_ENV_SHA256 = (
    "0fc298c17e9000877f49886dafb69b2181e44b0918d0463fe3c560bf129e5db7"
)

# The processors the per-image shortcut has been judged on, by module: each expands an image's
# placeholder from that image's own grid alone.
JUDGED_PROCESSORS = {
    "mlx_vlm.models.qwen3_vl.processing_qwen3_vl": (
        "9c6bc521ea6b45c3346967502f2d0b69697371327d7b340c23650e76de91f5d2"
    ),
}

OWES = "Re-check prompt_tokens' steps against the new source and re-pin (analysis/src/locus/analysis/mlx/count.py)."


def _verify_upstream() -> None:
    import importlib

    from mlx_vlm import utils
    from mlx_vlm.server import generation, openai, request_normalization, runtime_config

    # The package re-exports the app object under the module's own name.
    app = importlib.import_module("mlx_vlm.server.app")
    for what, fn, sha in (
        (
            "server/app.py _build_gen_args",
            app._build_gen_args,
            EXPECTED_APP_GEN_ARGS_SHA256,
        ),
        (
            "server/request_normalization.py _build_gen_args",
            request_normalization._build_gen_args,
            EXPECTED_GEN_ARGS_SHA256,
        ),
        (
            "server/openai.py chat_completions_endpoint",
            openai.chat_completions_endpoint,
            EXPECTED_CHAT_HANDLER_SHA256,
        ),
        (
            "server/generation.py ResponseGenerator._cpu_preprocess",
            generation.ResponseGenerator._cpu_preprocess,
            EXPECTED_CPU_PREPROCESS_SHA256,
        ),
        (
            "server/generation.py load_model_resources",
            generation.load_model_resources,
            EXPECTED_LOAD_MODEL_RESOURCES_SHA256,
        ),
        ("utils.py load", utils.load, EXPECTED_LOAD_SHA256),
        ("utils.py load_model", utils.load_model, EXPECTED_LOAD_MODEL_SHA256),
        (
            "utils.py get_model_path",
            utils.get_model_path,
            EXPECTED_GET_MODEL_PATH_SHA256,
        ),
        (
            "utils.py prepare_inputs",
            utils.prepare_inputs,
            EXPECTED_PREPARE_INPUTS_SHA256,
        ),
        ("utils.py load_image", utils.load_image, EXPECTED_LOAD_IMAGE_SHA256),
        (
            "server/runtime_config.py RuntimeConfig.from_env",
            runtime_config.RuntimeConfig.from_env,
            EXPECTED_RUNTIME_FROM_ENV_SHA256,
        ),
        *(
            (f"server/generation.py {name}", getattr(generation, name), sha)
            for name, sha in EXPECTED_ENV_READER_SHA256.items()
        ),
    ):
        expect_source("count", what, fn, sha, OWES)


def _verify_processor(processor) -> None:
    import importlib

    module = type(processor).__module__
    if module not in JUDGED_PROCESSORS:
        refuse(
            "count",
            f"the processor {type(processor).__name__}",
            f"(module {module}) is not one the per-image shortcut was judged on",
            OWES,
        )
    expect_source(
        "count",
        module,
        importlib.import_module(module),
        JUDGED_PROCESSORS[module],
        OWES,
    )


def _slug(model: str) -> str:
    from ..model.cards import OPENAI_COMPATIBLE, model_id, models_speaking

    for name in models_speaking(OPENAI_COMPATIBLE):
        if model_id(name) == model:
            return name
    raise ValueError(
        f"no openai-compatible card declares model {model!r} — the count runs under "
        f"the serving env of the card that names it"
    )


@contextmanager
def _serving(model: str):
    """Upstream run under the environment the bundled server serves `model` under (serve.serving_env): every setting upstream reads from the environment, its runtime config included, is the server's for the duration — a count-affecting variable the declaration leaves out is cleared, as the server will not boot with one (COUNT_ENV) — and this process's own is back after."""
    from mlx_vlm.server import runtime_config
    from mlx_vlm.server.runtime import runtime

    from .serve import serving_env

    env = serving_env(_slug(model))
    cleared = [key for key in COUNT_ENV if key not in env]
    before = {key: os.environ.get(key) for key in [*env, *cleared]}
    config = runtime.config
    for key in cleared:
        os.environ.pop(key, None)
    os.environ.update(env)
    runtime.config = runtime_config.RuntimeConfig.from_env()
    try:
        yield
    finally:
        runtime.config = config
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _local_files(model: str):
    """Where the model's processor and config files are, read from this machine alone — counting makes no network call. A local directory is itself; a Hub repo is the snapshot the server loads, from the Hugging Face cache its boot fills (upstream's get_model_path, without the fetch)."""
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    path = Path(model)
    if path.exists():
        return path
    try:
        return Path(
            snapshot_download(
                repo_id=model, allow_patterns=MODEL_FILES, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        raise RuntimeError(
            f"{model} is not in this machine's Hugging Face cache, and counting "
            f"reads only what is already here — boot the bundled server on its card "
            f"once (analysis/scripts/start-server.sh <card>), which downloads it, "
            f"then count again"
        ) from None


@functools.cache
def _resources(model: str):
    """The processor and config the server would hold for `model`, built by upstream's load steps minus the weights, with the arguments load_model_resources passes: the model's module is resolved first (it registers the model's own processor classes), then the config the way load_model builds it, then the processor with the model's own image processor in place. Called under _serving."""
    from pathlib import Path

    from mlx_vlm.server.generation import (
        get_configured_context_limit,
        get_expert_cache_gb,
    )
    from mlx_vlm.utils import (
        apply_generation_config_defaults,
        get_model_and_args,
        load_config,
        load_image_processor,
        load_processor,
        update_module_configs,
    )

    from .serve import install_heif_decode

    _verify_upstream()
    # The server decodes HEIF images too, so the count reads what it reads.
    install_heif_decode()
    kwargs = {
        "trust_remote_code": os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower()
        == "true",
        "expert_cache_gb": get_expert_cache_gb(),
        "max_kv_size": get_configured_context_limit(),
    }
    path = _local_files(model)
    config = load_config(path, **kwargs)
    model_class, _ = get_model_and_args(config=config, model_path=path)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})
    ple_storage = config["text_config"].get("ple_storage")
    manifest = ple_storage.get("manifest") if ple_storage else None
    if manifest and not Path(manifest).is_absolute():
        ple_storage["manifest"] = str(path / manifest)
    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config,
        model_class,
        config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    model_config = apply_generation_config_defaults(model_config, config)
    if hasattr(model_config, "model_path"):
        model_config.model_path = str(path)
    image_processor = load_image_processor(path, **kwargs)
    processor = load_processor(
        path,
        True,
        eos_token_ids=getattr(model_config, "eos_token_id", None),
        **kwargs,
    )
    if image_processor is not None:
        processor.image_processor = image_processor
    _verify_processor(processor)
    return processor, model_config


def _render(request: dict):
    """The request as the chat handler turns it into a prompt: (model, rendered prompt, the image urls in order, the request's preprocessing options as a hashable key). Called under _serving."""
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.server.openai import _ensure_effective_input
    from mlx_vlm.server.request_normalization import (
        _build_gen_args,
        _chat_message_to_prompt,
        _normalize_instruction_messages,
    )
    from mlx_vlm.server.schemas import ChatRequest
    from mlx_vlm.tools.policy import _prepare_chat_tool_choice

    body = {key: value for key, value in request.items() if key != "extra_body"}
    body.update(request.get("extra_body", {}))
    chat = ChatRequest(**body)
    processor, config = _resources(chat.model)

    sent, messages = [], []
    for message in chat.messages:
        if isinstance(message.content, list) and message.role == "user":
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    sent.append(item["image_url"]["url"])
        messages.append(_chat_message_to_prompt(message.model_dump(exclude_none=True)))
    _normalize_instruction_messages(messages)
    _ensure_effective_input(messages, images=sent, audio=[])
    messages, tools, tool_choice = _prepare_chat_tool_choice(
        messages, chat.tools, chat.tool_choice
    )
    # The structured-output grammar a request's response_format builds only steers sampling,
    # so the count skips building it.
    gen_args = _build_gen_args(
        chat,
        processor,
        structured_logits_processor_builder=lambda request, processor: None,
    )
    template_kwargs = gen_args.to_template_kwargs()
    if tool_choice is not None:
        template_kwargs["tool_choice"] = tool_choice
    prompt = apply_chat_template(
        processor,
        config,
        messages,
        num_images=len(sent),
        num_audios=0,
        video=None,
        tools=tools,
        **template_kwargs,
    )
    options = tuple(
        sorted(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in gen_args.preprocessing_kwargs().items()
        )
    )
    return chat.model, prompt, sent, options


def _tokens(model: str, prompt: str, images: list | None, options: tuple) -> int:
    """Upstream's own preprocessing of the prompt (ResponseGenerator._cpu_preprocess, run on the processor and config it would hold), counted. On the CPU: it is CPU work on the server too, and here nothing contends with a resident model for the GPU."""
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_vlm.server.generation import ResponseGenerator

    processor, config = _resources(model)
    holder = SimpleNamespace(processor=processor, model=SimpleNamespace(config=config))
    with mx.stream(mx.cpu):
        ids = ResponseGenerator._cpu_preprocess(
            holder, prompt, images, **dict(options)
        )["input_ids"]
        return ids.size if hasattr(ids, "size") else len(ids)


@functools.cache
def _expansion(model: str, width: int, height: int, options: tuple) -> int:
    """What the processor turns one image's placeholder into, less the placeholder: a one-image prompt preprocessed with an image of this size, less the same prompt tokenized without it. The image is blank — the expansion reads its size, never its pixels — and is gone when this returns."""
    from PIL import Image

    with _serving(model):
        _, prompt, _, _ = _render(_one_image_request(model))
        with_image = _tokens(
            model, prompt, [Image.new("RGB", (width, height))], options
        )
        return with_image - _tokens(model, prompt, None, options)


def _one_image_request(model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x"},
                    {"type": "image_url", "image_url": {"url": "image"}},
                ],
            }
        ],
    }


def _count(model: str, prompt: str, sizes: list, options: tuple) -> int:
    return _tokens(model, prompt, None, options) + sum(
        _expansion(model, width, height, options) for width, height in sizes
    )


def prompt_tokens(request: dict) -> int:
    """The prompt_tokens the server would report for this chat-completions request body (the client's own request: model, messages, tools, and the extra_body fields merged in).

    The server preprocesses every image of a request together; this count gets the same number while holding at most one image at a time. The processor expands each image's placeholder by that image alone — from the grid its size and the request's preprocessing options decide, nothing else in the request (the judged processors, JUDGED_PROCESSORS) — so the prompt is tokenized once with its placeholders unexpanded, and each image adds its expansion, taken once per distinct size and kept. Each image is read by upstream's own loader, as the server reads it, for its size alone."""
    require_extra("mlx.core", "mlx_vlm")
    from mlx_vlm.utils import load_image

    with _serving(request["model"]):
        model, prompt, sent, options = _render(request)
        sizes = [load_image(url).size for url in sent]
        return _count(model, prompt, sizes, options)


@functools.cache
def image_tokens(model: str, width: int, height: int) -> int:
    """What one image of this size costs in the server's count, everything the template wraps around it included: the count of a one-image request less the same request without it."""
    require_extra("mlx.core", "mlx_vlm")
    with_image = _one_image_request(model)
    without = {
        "model": model,
        "messages": [
            {"role": "user", "content": with_image["messages"][0]["content"][:1]}
        ],
    }
    with _serving(model):
        _, prompt, _, options = _render(with_image)
        _, bare, _, _ = _render(without)
        return _count(model, prompt, [(width, height)], options) - _count(
            model, bare, [], options
        )
