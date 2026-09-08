"""Deliver a conversation's images to the model where the conversation put them.

The client composes each message as ordered parts — text and screenshots interleaved at the moments they belong to — and the model's own chat template renders a message's parts in the positions it is handed. Between the two, upstream destroys the order twice.

Across messages: the chat-completions handler strips every image part out of the message it arrived in and collects the images into one flat list, so the messages it templates carry text only, and the template layer, left without markers, hangs every image on the *last* user message (`prompt_utils.apply_chat_template`: "Any legacy side-channel media without markers remains attached to the last user message for backward compatibility"). A conversation's evidence jumps forward to its newest turn — a later turn's task is handed the screenshots it reasons over after its own instruction — and the request stops being an extension of the one before it at the first relocated image token, the shape any server-side prefix cache would need to reuse.

Within a message: the prompt builder joins the message's text parts into one string and re-attaches its images as a count, images first, so the message's own interleaving collapses to an image block ahead of its text.

Three symbol swaps restore what those layers discard, none changing which images are sent — the handler has collected them before any of this runs — only where the template places them:

- the handler's `extract_text_from_content` keeps a content list that carries media, so each message's markers survive and its images are counted against the message that carried them (upstream's own responses endpoint keeps markers the same way);
- `prompt_utils`' `extract_text_from_content` keeps the same lists on the template-side pass;
- `get_message_json` builds a media-carrying list into template parts in the list's own order — text where text sat, an image marker where each image sat, honoring the image count upstream's allocation assigned to the message. String content, media-free lists, and any message holding a part type this patch does not place take upstream's own path unchanged.

Both halves are properties of the exact upstream source — that the handler collects a request's images before it flattens each message's content, and that the template layer places media by the markers left in it — so install verifies both functions' sources against the pinned hashes and refuses to boot on drift (patching.py owns that policy).
"""

from .patching import expect_source

MEDIA_PART_TYPES = ("image", "image_url", "input_image")
TEXT_PART_TYPES = ("text", "input_text")

EXPECTED_ENDPOINT_SHA256 = (
    "e34a19d3bda48caf202de7c604bca78fec811283b0ce0aa1d82b3326b39132be"
)
EXPECTED_TEMPLATE_SHA256 = (
    "2c4393fc3e5c06a667486234f2fedcea851af8020af8f5f07904d2a9bf64efba"
)

_OWES = (
    "Re-read both — the handler must collect a request's images before it flattens each "
    "message's content, and the template layer must place media by the markers left in it — "
    "then re-validate the swaps and re-pin."
)


def carries_media(content) -> bool:
    """Whether this message content holds media parts the template must place by marker."""
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") in MEDIA_PART_TYPES
        for part in content
    )


def ordered_parts(content, skip_image_token: bool, num_images: int):
    """Template parts for a media-carrying content list, in the list's own order — or None when the list holds a part this patch does not place, so upstream's own path takes the message.

    Image markers are emitted up to the count upstream's allocation assigned this message, exactly as upstream's own builder honors it; text parts are emitted verbatim where they sat."""
    from mlx_vlm.prompt_utils import MessageBuilder

    parts = []
    remaining = num_images
    for item in content:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind in TEXT_PART_TYPES:
            text = item.get("text", "") or item.get("content", "")
            if text:
                parts.append(MessageBuilder.text_message(text))
        elif kind in MEDIA_PART_TYPES:
            if not skip_image_token and remaining > 0:
                parts.append(MessageBuilder.image_message())
                remaining -= 1
        else:
            return None
    return parts


def install_media_placement() -> None:
    """Verify the pinned upstream sources and swap the three symbols; on drift, refuse to boot (patching.py)."""
    from mlx_vlm import prompt_utils
    from mlx_vlm.server import openai

    expect_source(
        "media placement",
        "chat_completions_endpoint",
        openai.chat_completions_endpoint,
        EXPECTED_ENDPOINT_SHA256,
        _OWES,
    )
    expect_source(
        "media placement",
        "apply_chat_template",
        prompt_utils.apply_chat_template,
        EXPECTED_TEMPLATE_SHA256,
        _OWES,
    )

    handler_extract = openai.extract_text_from_content

    def keeping_media_at_handler(content):
        return content if carries_media(content) else handler_extract(content)

    template_extract = prompt_utils.extract_text_from_content

    def keeping_media_at_template(content):
        return content if carries_media(content) else template_extract(content)

    build_message = prompt_utils.get_message_json

    def placing_in_order(
        model_name,
        prompt,
        role="user",
        skip_image_token=False,
        skip_audio_token=False,
        num_images=0,
        num_audios=0,
        **kwargs,
    ):
        if isinstance(prompt, list):
            parts = (
                ordered_parts(prompt, skip_image_token, num_images)
                if carries_media(prompt)
                else None
            )
            if parts is not None:
                return {"role": role, "content": parts}
            prompt = template_extract(prompt)
        return build_message(
            model_name,
            prompt,
            role,
            skip_image_token,
            skip_audio_token,
            num_images,
            num_audios,
            **kwargs,
        )

    openai.extract_text_from_content = keeping_media_at_handler
    prompt_utils.extract_text_from_content = keeping_media_at_template
    prompt_utils.get_message_json = placing_in_order
