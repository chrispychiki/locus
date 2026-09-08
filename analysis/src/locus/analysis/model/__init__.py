"""Provider-agnostic conversation layer.

Two backends, one shared property: the model's context is the actual limit. Gemini-direct rides the Files API (images upload free, ride as URIs); the local OpenAI-compatible server reads images straight off the shared filesystem. `make_conversation` in factory.py is the only place a model name is interpreted into a backend choice.
"""

from .factory import make_conversation
from .protocol import Conversation, Response

__all__ = ["Conversation", "Response", "make_conversation"]
