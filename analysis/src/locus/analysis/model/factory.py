"""The only place a model name becomes a client.

Every client has the same property: the model's context is the actual limit. Which client speaks to a card is the card's own `conversation` declaration, looked up in the registry here. A new client is a Conversation subclass (protocol.py) and one registry line. The analysis README says why the field is this narrow.
"""

from .cards import (
    GEMINI,
    OPENAI_COMPATIBLE,
    conversation,
    load_card,
    max_output_tokens,
    media_resolution,
    thinking_level,
)
from .gemini import GeminiConversation
from .openai_compat import OpenAICompatConversation
from .protocol import Conversation

CONVERSATIONS = {
    GEMINI: GeminiConversation,
    OPENAI_COMPATIBLE: OpenAICompatConversation,
}


def make_conversation(
    model_name: str, system_prompt: str, *, record_dir, **kwargs
) -> Conversation:
    """record_dir is where the conversation's transcript lands — stated by the motion that creates the conversation, beside the output it produces. There is no unrecorded model-bound conversation."""
    kwargs["record_dir"] = record_dir
    client = conversation(model_name)
    if client not in CONVERSATIONS:
        raise ValueError(
            f"card {model_name!r} names conversation {client!r}, which no client "
            f"here speaks — the registry holds {sorted(CONVERSATIONS)}"
        )
    if client == GEMINI:
        kwargs.setdefault("thinking_level", thinking_level(model_name))
        kwargs.setdefault("media_resolution", media_resolution(model_name))
        kwargs.setdefault("max_output_tokens", max_output_tokens(model_name))
        return GeminiConversation(model_name, system_prompt, **kwargs)
    card = load_card(model_name)
    return OpenAICompatConversation(system_prompt, card=card, **kwargs)
