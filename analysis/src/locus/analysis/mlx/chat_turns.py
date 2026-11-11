"""Every turn of a conversation reaches the model's chat template as it was sent: no turn is compacted away, a tool's reply keeps its images, and a request's chat_template_kwargs reach the template.

Unless a request sends `context_management: []`, upstream's chat handler compacts a conversation: it clamps max_tokens to the context left and, past the context limit, swaps the conversation's head, the evidence an answer cites, for a model-written summary. Patched: the handler never compacts, and a request that asks it to is refused with a 400 (compacts).

The chat template each local model is served with renders a tool message's images in place (Qwen's own; Gemma 4's as gemma4.py changes it), and reads options a request sets: under preserve_thinking, which both cards set, Qwen's renders every earlier turn's reasoning, as its model card recommends for agent use, and Gemma 4's every earlier tool-call turn's, which its model card says is kept. Through upstream alone, neither reaches the template:
- a tool message's images: the chat handler and the request normalizer take images from user messages only. Patched: they take a tool message's images the same way.
- chat_template_kwargs, where vLLM and SGLang take template options: upstream ignores it. Patched: the options reach get_chat_template whole. An option is refused with a 400 unless the template reads it and nothing else sets it: a field of the request belongs in the request, and a name the server already passes cannot be overridden.

An assistant turn's reasoning reaches the template through upstream alone: apply_chat_template copies reasoning_content onto the message it rebuilds, which is why it is pinned.

Each patch rewrites the named upstream function's code in place, so every caller runs the rewritten code, the count (count.py) included. Each is pinned to the source it rewrites, as are the unrewritten functions it depends on (patching.py).
"""

import inspect

from .patching import expect_source, refuse

# The roles whose list content carries images to the template.
IMAGE_ROLES = ("user", "tool")

# The keyword the request's options travel under from the handler to get_chat_template.
CARRIER = "chat_template_kwargs"

# Every name the render hands the template beyond its keyword parameters: the conversation.
RENDER_SETS = ("messages",)

EXPECTED_SHA256 = {
    "server/openai.py chat_completions_endpoint": "f94a064abb6e43d73f57021068b9d6ce413646f716a89f57a99622be4e7debc8",
    "server/request_normalization.py _chat_message_to_prompt": "f43965254200dda17f3f6d34079efe846ebee3c836d76ac7d1e4d639862cf207",
    "prompt_utils.py apply_chat_template": "376ce52e1b34b3aee9242e97e21b46d1f3818532c6ff27a4c31d4447b0ffcf8d",
    "prompt_utils.py get_chat_template": "6cbda46759fa8de2c9e580b26210642aa0b93902be2fd63eaa4c14d612809d45",
    "prompt_utils.py normalize_image_content": "b92c704d42e5953c346a2d40b96a50fdde6c8056ea455ff4887ef19d6effc8cf",
    "prompt_utils.py _normalize_tool_message": "cc377cf6dfedc6186daab2f849a0edbe276002aee8917dd819e02bf04d1f6775",
    "prompt_utils.py _content_media_count": "202e184739ee1e415ad03f465033ce9fa3b5f6766ca5fd725d90c8a8bd7cecc4",
}

REWRITES = {
    "server/openai.py chat_completions_endpoint": [
        (
            "        if request.context_management != []:\n",
            (
                "        from locus.analysis.mlx.chat_turns import compacts\n"
                "\n"
                "        if compacts(request):\n"
            ),
        ),
        (
            'if message.role == "user":',
            f"if message.role in {IMAGE_ROLES!r}:",
        ),
        (
            "        template_kwargs = gen_args.to_template_kwargs()\n",
            (
                "        template_kwargs = gen_args.to_template_kwargs()\n"
                "        from locus.analysis.mlx.chat_turns import refused_as_400, request_options\n"
                "\n"
                f"        template_kwargs[{CARRIER!r}] = refused_as_400(request_options)(request)\n"
            ),
        ),
        (
            "        formatted_prompt = continuation.render_prompt(\n            apply_chat_template,\n",
            "        formatted_prompt = continuation.render_prompt(\n            refused_as_400(apply_chat_template),\n",
        ),
    ],
    "server/request_normalization.py _chat_message_to_prompt": [
        (
            'if message["role"] == "user"',
            f'if message["role"] in {IMAGE_ROLES!r}',
        ),
    ],
    "prompt_utils.py get_chat_template": [
        (
            '    chat_template_override = kwargs.get("chat_template", None)\n',
            (
                "    from locus.analysis.mlx.chat_turns import template_options\n"
                "\n"
                f"    options = kwargs.pop({CARRIER!r}, None)\n"
                '    chat_template_override = kwargs.get("chat_template", None)\n'
            ),
        ),
        (
            "        if template_processor is None:\n",
            (
                "        if template_processor is None:\n"
                "            template_options(options, kwargs, None, chat_template_override)\n"
            ),
        ),
        (
            "        try:\n            return template_processor.apply_chat_template(\n",
            (
                "        template_kwargs.update(\n"
                "            template_options(\n"
                "                options, template_kwargs, template_processor, chat_template_override\n"
                "            )\n"
                "        )\n"
                "        try:\n            return template_processor.apply_chat_template(\n"
            ),
        ),
    ],
}

PINNED = {
    "server/openai.py chat_completions_endpoint": (
        "mlx_vlm.server.openai",
        "chat_completions_endpoint",
    ),
    "server/request_normalization.py _chat_message_to_prompt": (
        "mlx_vlm.server.request_normalization",
        "_chat_message_to_prompt",
    ),
    "prompt_utils.py apply_chat_template": (
        "mlx_vlm.prompt_utils",
        "apply_chat_template",
    ),
    "prompt_utils.py get_chat_template": ("mlx_vlm.prompt_utils", "get_chat_template"),
    "prompt_utils.py normalize_image_content": (
        "mlx_vlm.prompt_utils",
        "normalize_image_content",
    ),
    "prompt_utils.py _normalize_tool_message": (
        "mlx_vlm.prompt_utils",
        "_normalize_tool_message",
    ),
    "prompt_utils.py _content_media_count": (
        "mlx_vlm.prompt_utils",
        "_content_media_count",
    ),
}

OWES = (
    "Each rewrite rests on its lines doing in the new source what they did in the pinned one — "
    "the handler compacting only inside the block its context_management test gates, "
    "collecting a message's images, building the template's keyword arguments and "
    "rendering with them, the normalizer choosing which roles keep list content, "
    "apply_chat_template carrying an assistant message's reasoning_content to the template and "
    "counting a tool message's image markers, get_chat_template building what the template renders with — "
    "re-verify those, then re-pin (analysis/src/locus/analysis/mlx/chat_turns.py)."
)

_installed = False


class TemplateOptionRefused(ValueError):
    """A request's chat_template_kwargs carry an option the template cannot take as sent."""


def refused_as_400(step):
    """`step`, with a refused template option answered as the client's error (400) rather than the server's."""
    from fastapi import HTTPException

    def refusing(*args, **kwargs):
        try:
            return step(*args, **kwargs)
        except TemplateOptionRefused as refusal:
            raise HTTPException(status_code=400, detail=str(refusal)) from refusal

    return refusing


def compacts(request) -> bool:
    """The handler's compaction gate, always False here. A request that asks for compaction is refused as the client's error (400); every other is served as composed, whether it sends `context_management` or not."""
    if request.context_management:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=(
                "this server never compacts a conversation: it serves every "
                "conversation as composed, and refuses one over the context limit; "
                "send no context_management"
            ),
        )
    return False


def request_options(request) -> dict | None:
    """The request's chat_template_kwargs, None when it sends none: refused unless an object, and refused where an option is a field of the request itself."""
    options = (request.model_extra or {}).get(CARRIER)
    if options is None:
        return None
    if not isinstance(options, dict):
        raise TemplateOptionRefused(
            f"chat_template_kwargs must be an object of template options; got {options!r}"
        )
    fields = sorted(set(options) & set(type(request).model_fields))
    if fields:
        raise TemplateOptionRefused(
            f"chat_template_kwargs sets {fields}, fields of the request itself — send them there"
        )
    return options


def _rendered_template(template_processor, override) -> str | None:
    """The template the render will use, chosen as transformers chooses it: the named template or the given one when the call passes one, else the processor's own, its "default" when it holds several."""
    held = getattr(template_processor, "chat_template", None)
    if isinstance(held, dict):
        return held.get("default" if override is None else override, override)
    return held if override is None else override


def _template_variables(template: str) -> set[str]:
    """The variables the template reads from whoever renders it: transformers' own reading (the one its processor sorts keyword arguments by), less the names its environment supplies."""
    from transformers.utils.chat_template_utils import (
        _compile_jinja_template,
        _get_template_variables,
    )

    environment = _compile_jinja_template(template).environment
    return set(_get_template_variables(template)) - set(environment.globals)


def template_options(options, given, template_processor, override) -> dict:
    """The request's options, as they join what the template renders with: {} when it sent none; refused when the model has no template, when an option is a name the template is already given (`given`, the render's own parameters, the conversation), and when the template never reads it."""
    if not options:
        return {}
    template = (
        None
        if template_processor is None
        else _rendered_template(template_processor, override)
    )
    if not isinstance(template, str):
        raise TemplateOptionRefused(
            "chat_template_kwargs sent, but the model has no chat template to read them"
        )
    parameters = [
        name
        for name, parameter in inspect.signature(
            template_processor.apply_chat_template
        ).parameters.items()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    taken = sorted(set(options) & {*given, *parameters, *RENDER_SETS})
    if taken:
        raise TemplateOptionRefused(
            f"chat_template_kwargs sets {taken}, which the server already gives "
            f"the chat template"
        )
    unread = sorted(set(options) - _template_variables(template))
    if unread:
        raise TemplateOptionRefused(
            f"chat_template_kwargs {unread}: the model's chat template never reads "
            f"them, so they would change nothing"
        )
    return dict(options)


def _function(what: str):
    import importlib

    module_name, name = PINNED[what]
    module = importlib.import_module(module_name)
    return module, name, getattr(module, name)


def _rewritten(what: str):
    """The patched function's code, compiled from its pinned source with every anchor rewritten."""
    module, name, function = _function(what)
    source = inspect.getsource(function)
    for anchor, replacement in REWRITES[what]:
        if source.count(anchor) != 1:
            refuse(
                "chat turns",
                what,
                f"holds {source.count(anchor)} copies of {anchor!r} where one was rewritten",
                OWES,
            )
        source = source.replace(anchor, replacement)
    # Padded so the rewritten code keeps the original's line numbers in tracebacks.
    padded = "\n" * (function.__code__.co_firstlineno - 1) + source
    namespace: dict = {}
    exec(compile(padded, module.__file__, "exec"), module.__dict__, namespace)  # noqa: S102
    return function, namespace[name].__code__


def install_chat_turns() -> None:
    """Verify every pinned function and rewrite every anchor, then swap the patched functions' code in place, all of it or none; once per process."""
    global _installed
    if _installed:
        return
    for what in PINNED:
        expect_source(
            "chat turns", what, _function(what)[2], EXPECTED_SHA256[what], OWES
        )
    for function, code in [_rewritten(what) for what in REWRITES]:
        function.__code__ = code
    _installed = True
