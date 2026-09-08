"""The conversation contract every model backend implements.

A capability a backend cannot realize is degraded openly, with the limit documented where it happens, so a caller always knows what its argument actually did. Backend-specific knobs (thinking levels, media resolution, reasoning effort) belong to the concrete backend's constructor, visible only to callers holding the concrete class.

How an image rides the wire is the backend's own card's business, never the caller's: the local backend realizes its card's per-image token ceilings by handing the server a physically smaller file, and Gemini sends each image at a card-declared resolution tier for that tier's flat price. The one thing a caller states is what the image is — add_user_image's pushed flag marks an activity-screenshot, whose job is sampling the session — and each backend realizes it as its card allows: Gemini rides activity-screenshots at the card's activity_screenshots tier, the local backend carries each under its card's ceiling for that kind. Either way the record names what was actually sent.

A caller that wants to know what an image will cost, or how much context is left to spend, asks the backend — image_tokens and context_remaining are the wire's own arithmetic, so an agentic caller serves what fits and refuses the rest against a number that is true at the moment it acts.

The public surface is a template method over backend primitives (_add_user_text / _add_user_image / _get_response), so per-call payload persistence is inherited identically by every backend, and it is never optional: a model-bound conversation states its record home at construction (record_dir — the same motion that places the output it produces), and each get_response() writes three files there, numbered by call — N_input.txt (the entire conversation the model sees on this call, whatever the transport's wire shape: system prompt and every part verbatim in order, roles delimited, images as inline pointer lines carrying the image's path and the remote reference actually sent), N_response.txt (the reply, with thoughts inline and delimited when present), N_meta.json (generation config, usage, the call's UTC start stamp, wall-clock elapsed, error). The input record lands before the request goes to the wire, the response and meta on completion — so a collision with an existing record refuses before anything is spent, a call that dies mid-flight still leaves its input behind, and a failed call leaves its input and meta with the error. Existing files are never overwritten. response_text() inverts the response file's format, so a reader of the record recovers exactly the reply text.

Image pointers are written relative to the payload file that carries them, so they resolve from where they sit and the record directory holding them stays portable.

Spend governance rides the same template method: a backend that declares its per-token prices (_billing) has every generation call checked against the deployment's declared spend walls before the request goes to the wire and appended to the spend ledger from its persisted usage after — so a billed backend cannot run outside the walls or off the ledger, the same way no model-bound conversation can run unrecorded. A refused call still leaves its input and its meta with the refusal; a call that fails after the wire reported usage (UnusableReply) lands that usage in its meta and on the ledger, because a failure must never read as less spend. The walls, the ledger, and the arithmetic are spend.py's.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DryRun(Exception):
    """Raised by a dry-run backend at the wire boundary instead of sending.

    The intervention point is the lowest level, latest moment: the entire real path runs — content accumulation, uploads, config assembly, payload persistence — and only the generation request itself is refused, so the persisted input record is the payload exactly as it would have been sent.
    """


class UnusableReply(RuntimeError):
    """The wire returned a reply, but its terminal status carries no usable output — the caller must not mistake it for an answer. It still carries the usage the wire reported: generation happened and its tokens are spend, so the protocol layer's error path lands that usage on the ledger and in the call's meta even though the call raises."""

    def __init__(self, message: str, usage_metadata: dict | None = None):
        super().__init__(message)
        self.usage_metadata = usage_metadata or {}


@dataclass
class ToolCall:
    """A model-emitted tool call, normalized across backends. `arguments` is the parsed argument object — or None where the wire carried argument text that is not a JSON object (the arguments are the model's own writing, and a backend without generation-level grammar can emit anything): the call still exists and must be answered, so the caller answers it with the defect named rather than crashing on it."""

    id: str
    name: str
    arguments: dict | None


@dataclass
class Response:
    """Normalized response shape, identical across backends. parsed is filled by the protocol layer (get_response), never by a backend: the reply text is validated into the caller's requested type after the reply is persisted, so the caller always receives instances of what it asked for and a reply that fails validation still stands complete in the record."""

    text: str
    thoughts_text: str = ""
    parsed: Any = None
    usage_metadata: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)


# The backend schema contract.
# A schema keyword means something at this seam only if both backends enforce
# it AT GENERATION — the ground layer's structured-output guarantees hang on
# exactly these (ground/claims.py says why each exists):
#   - a root anyOf union (ClaimsFound | NoClaims) — both branches objects, so
#     the branch decision is a named key, never the opening token
#     (ground/claims.py says why that placement is load-bearing);
#   - minItems == maxItems pinned equal (a verdict batch's count pinned to its
#     claim count);
#   - numeric minimum/maximum from ge/le (timestamp components, pct scores);
#   - minItems 1 from min_length=1 (a claims list is non-empty or is not the
#     list branch at all);
#   - a required string constant (NoClaims' asserted branch) — a string and
#     never the bool it reads as, because Gemini expresses constants only as
#     string enum (below).
#
# Per backend, with the evidence:
#
# Gemini-direct — the API's Schema object documents anyOf, minItems, maxItems,
# minimum, maximum (plus minLength/maxLength/pattern for strings); enum is
# string-only and no const field exists (https://ai.google.dev/api/caching#Schema,
# checked 2026-07-17). A non-string Literal therefore has nowhere to go and the
# google-genai SDK rejects it; only a string constant survives the trip.
# propertyOrdering is Gemini's own extra (gemini.py stamps it).
#
# Local (bundled mlx server) — upstream mlx-vlm compiles the
# response_format json_schema through llguidance (grammar_from("json_schema"),
# LLMatcher token masks at the draw; mlx_vlm/structured.py), so enforcement is
# grammar-level and thinking-aware. Probed live 2026-07-17 against the served
# qwen3.6-35b-a3b-6bit, adversarially: a prompt naming two claims against a
# batch pinned to three returned exactly three verdicts; demanded scores of 250
# and -50 came back inside [0, 100]; a demanded bare [] against the union could
# not be emitted — the reply was a conforming non-empty claims list. Observed
# limits, probed against llguidance 1.8.0 directly (LLMatcher over the served
# model's tokenizer): additionalProperties is not closed, so extra keys can
# appear in objects (pydantic validation ignores them); and an object's declared
# properties are admitted only in declaration order — a declared key offered out
# of order is masked at generation, and the open additionalProperties then
# admits a near-miss mangled key in its place, which pydantic silently drops.
# So a model's field declaration order IS wire grammar on this backend: it must
# match the emission order the prompt teaches, or the taught field is lost.
#
# A keyword outside this set is not load-bearing: get_response re-validates
# every reply client-side, so an unenforced keyword degrades to a loud
# post-hoc failure, never silent acceptance.
def json_schema_of(response_schema: Any) -> dict:
    """The caller's response schema as the JSON schema a backend puts on its wire — refused when empty, because an empty schema (the Any case) constrains nothing while looking like structured output. The comment block above is the contract for which schema keywords the backends enforce at generation."""
    from pydantic import TypeAdapter

    schema = TypeAdapter(response_schema).json_schema()
    if not schema:
        raise ValueError(
            "response_schema produced an empty JSON schema — pass a concrete type, never Any"
        )
    return schema


def reply_record(response: "Response") -> str:
    """A response rendered in the record's reply format — thoughts delimited when the backend exposed them, otherwise the text alone. This is the one written form of a model's utterance: _persist_result writes every N_response.txt with it, and response_text() inverts it."""
    if response.thoughts_text:
        return f"=== thoughts ===\n{response.thoughts_text}\n\n=== response ===\n{response.text}"
    return response.text


def response_text(recorded: str) -> str:
    """The reply text back out of the record's reply format — the inverse of reply_record, kept beside the writer so the two halves cannot drift. A response with thoughts is delimited; without, the file is the text plus the trailing newline the persistence layer guarantees."""
    body = recorded.removesuffix("\n")
    if body.startswith("=== thoughts ===\n"):
        _, _, body = body.partition("\n\n=== response ===\n")
    return body


def latest_answer_response(record_dir: Path) -> Path:
    """The analysis's answer transcript: the highest-numbered answer-labeled response in a conversation's record directory. The reader twin of _persist_input's `{n}_{label}` stems, kept beside the writer so the naming grammar has one home; both the engine (handing the path back) and the oracle (measuring it) read through here. A directory holding none fails loud — there is no answer."""
    responses = sorted(
        record_dir.glob("*_answer_response.txt"),
        key=lambda p: int(p.name.split("_", 1)[0]),
    )
    if not responses:
        raise ValueError(
            f"{record_dir} holds no answer response transcript — nothing to read"
        )
    return responses[-1]


class Conversation(ABC):
    """A running exchange with one vision-capable model.

    Usage: construct with a system prompt, add user text/images in order, call get_response(); the assistant turn is recorded automatically so the next get_response() continues the same conversation.
    """

    def __init__(self, record_dir: str | Path | None = None, system_prompt: str = ""):
        """record_dir is where this conversation's transcript lands — assigned by the motion that creates the conversation, alongside the output it exists to produce. The model-bound backends require it at construction; only a conversation that never reaches a model (a counting tally, a test double that fakes the wire) may run without one. system_prompt heads every persisted input record; each backend carries it to the wire in its own system channel."""
        self.system_prompt = system_prompt
        self._record: list[tuple] = []
        self._persist_dir: Path | None = (
            Path(record_dir) if record_dir is not None else None
        )
        self._persisted_calls = 0
        self._open_calls: set[str] = set()
        self._last_usage_total: int | None = None
        self._record_at_response = 0
        # Where the call in flight stands, as the backend learns it — read from another thread by
        # a caller's pulse while this one blocks on the wire, so it is only ever replaced whole.
        self.progress: dict = {}

    def progress_line(self) -> str:
        """The call in flight, in one line for a caller's pulse: empty until a backend has something to say, and after the call whatever it said last."""
        return self.progress.get("line", "")

    def prefetch_images(self, image_paths) -> None:
        """Latency hint for a batch of images the conversation is about to carry. A backend that must move image bytes somewhere before they can ride a request (Gemini's Files API) satisfies the batch concurrently here so the add_user_image calls that follow hit its cache; a backend whose images ride as local paths has nothing to move and inherits this no-op. Semantics never change: a caller that skips this sends identical requests, just slower."""

    def add_user_text(self, text: str) -> None:
        self._add_user_text(text)
        self._record.append(("text", text))

    def add_user_image(self, image_path: str, *, pushed: bool = False) -> None:
        """pushed marks an activity-screenshot — one of the composed set sampling the session; a backend whose card prices per-kind resolution tiers rides it at its activity_screenshots tier, and a backend with one image economics carries it like any other image."""
        sent_as = self._add_user_image(image_path, pushed)
        self._record.append(("image", str(image_path), sent_as))

    def add_tool_result(self, call_id: str, content: str) -> None:
        """Answer a model tool call — exactly once, and only a call this
        conversation's model actually made: a result addressed anywhere else is
        answered into the void while the call the model did make goes unanswered,
        so it is refused here, where the mistake is still legible. On every
        backend the rendered screenshot that a request produces is fed back as a
        separate user turn (add_user_image), not inside the tool result — Gemini
        can carry images in a functionResponse but mlx collects pixels only from
        user-role messages, so the user-turn shape is the one both honor. This
        carries the textual acknowledgement the tool-calling protocol requires
        before the next user turn."""
        if call_id not in self._open_calls:
            raise ValueError(
                f"tool result for call {call_id!r}, which this conversation "
                f"never made or already answered; open calls: "
                f"{sorted(self._open_calls) or 'none'}"
            )
        self._open_calls.discard(call_id)
        self._add_tool_result(call_id, content)
        self._record.append(("tool", call_id, content))

    def get_response(
        self,
        *,
        response_schema: Any = None,
        label: str | None = None,
        tools: list | None = None,
    ) -> Response:
        # The config manifest is read after the call, not before it: what
        # meta.json records must be the configuration that actually produced
        # (or failed to produce) this reply, and a backend may adjust its own
        # knobs mid-call (Gemini's exhausted-thinking retry re-runs at minimal
        # thinking, and the record states that level, not the declared one).
        def config() -> dict:
            return {
                "execution_mode": self.execution_mode,
                "response_schema": (
                    None
                    if response_schema is None
                    else getattr(response_schema, "__name__", repr(response_schema))
                ),
                "tools": [t.get("name") for t in tools] if tools else None,
                **self._config_manifest(),
            }

        billing = self._billing()
        stem = self._persist_input(label)
        from locus.evidence.clock import utc_stamp

        started_at = utc_stamp()
        started = time.monotonic()
        try:
            if billing is not None:
                from ..spend import guard_paid_call

                guard_paid_call(billing)
            response = self._get_response(
                response_schema=response_schema,
                tools=tools,
            )
        except Exception as error:
            # A call that failed after the wire reported usage still spent:
            # the reported usage lands in the meta and, on a billed
            # backend, on the ledger — an error must never read as less
            # spend. An error carrying no usage (a 400/500, a refused
            # call) reported nothing and records nothing.
            usage = getattr(error, "usage_metadata", None)
            self._persist_result(
                stem,
                config(),
                label,
                error=repr(error),
                usage=usage,
                started_at=started_at,
                elapsed_s=time.monotonic() - started,
            )
            if billing is not None and usage:
                from ..spend import record_paid_call

                record_paid_call(
                    billing,
                    label=label,
                    usage=self._spend_usage(usage),
                    backend=type(self).__name__,
                    record=(
                        None
                        if stem is None
                        else str(self._persist_dir / f"{stem}_meta.json")
                    ),
                )
            raise
        self._persist_result(
            stem,
            config(),
            label,
            response=response,
            started_at=started_at,
            elapsed_s=time.monotonic() - started,
        )
        if billing is not None:
            from ..spend import record_paid_call

            record_paid_call(
                billing,
                label=label,
                usage=self._spend_usage(response.usage_metadata),
                backend=type(self).__name__,
                record=(
                    None
                    if stem is None
                    else str(self._persist_dir / f"{stem}_meta.json")
                ),
            )
        if (
            response_schema is not None
            and response.parsed is None
            and not response.tool_calls
        ):
            if not response.text:
                raise ValueError(
                    f"model returned no structured output (label={label!r}; "
                    f"content was empty — typically the model burned "
                    f"max_tokens inside its thinking block); the full "
                    f"response is in the persisted payloads"
                )
            from pydantic import TypeAdapter

            try:
                response.parsed = TypeAdapter(response_schema).validate_json(
                    response.text
                )
            except Exception as error:
                raise ValueError(
                    f"model returned no parseable structured output "
                    f"(label={label!r}; {error!r}); the full response is in "
                    f"the persisted payloads"
                ) from error
        if response.tool_calls:
            self._record.append(
                (
                    "assistant_tool_calls",
                    response.tool_calls,
                    response.text,
                    response.thoughts_text,
                )
            )
            self._open_calls.update(call.id for call in response.tool_calls)
        elif response.text:
            self._record.append(("assistant", response.text, response.thoughts_text))
        self._last_usage_total = self._usage_total(response.usage_metadata)
        self._record_at_response = len(self._record)
        return response

    @abstractmethod
    def _add_user_text(self, text: str) -> None: ...

    @abstractmethod
    def _add_user_image(self, image_path: str, pushed: bool) -> str:
        """Attach the image as this backend's card carries images — honoring
        the pushed intent where the card prices tiers — and return the
        remote reference (or on-disk path) actually sent."""

    @abstractmethod
    def image_tokens(self, image_path: str) -> int:
        """What this image costs on this backend's wire as a pull — the only kind appended after turn 1, which is the only time anything asks."""

    @abstractmethod
    def context_remaining(self) -> int:
        """Tokens the model's context still holds beyond what this conversation
        already carries and what its generation will claim. Anchored on the
        wire's own count — the last response's persisted usage total — plus the
        priced delta of parts appended since it (_wire_anchored_used), so it is
        measurable only once a response exists; turn-1 fit is the composition
        plane's question (budget.price_payload), asked before anything is
        sent."""

    def _billing(self) -> dict | None:
        """{"model", "prices"} when this conversation's generation calls bill — the card's name and its declared per-token prices — or None when they are free, which is what every free path inherits. The card decides: a pricing block has its calls governed at those rates, and non-None here engages the spend layer in the template method, so a billed conversation is inside the operator's walls and on the ledger by construction (spend.py owns both)."""
        return None

    def _spend_usage(self, usage_metadata: dict) -> dict:
        """A response's usage record read into the ledger's token vocabulary — input_tokens (everything sent, cached included), cached_tokens (the cache-discounted subset), output_tokens, thoughts_tokens — every billed token counted, burned retry attempts included. Each billed backend states how its own usage shape reads; a free backend has no reading to give."""
        raise NotImplementedError(
            "a backend whose calls bill must state how its usage record reads into spend tokens"
        )

    def _usage_total(self, usage_metadata: dict) -> int | None:
        """This backend's total-token figure out of a response's usage record —
        the wire's authoritative count of everything the conversation carried
        through that reply. None where the record carries none (a conversation
        that never reaches a model)."""
        return None

    def _delta_text_tokens(self, texts: list[str]) -> int:
        """What the given text parts cost on this backend's wire — the pricing
        half of _wire_anchored_used, implemented by the model-bound backends."""
        raise NotImplementedError

    def _wire_anchored_used(self) -> int:
        """The conversation's size by the wire's own record: the last response's
        usage total — already persisted to its meta.json — plus the priced delta
        of parts appended since it (tool-result text, served screenshots). Refuses
        before any response exists, because there is no wire count to anchor
        on."""
        if self._last_usage_total is None:
            raise RuntimeError(
                "no response has been received, so there is no wire usage to "
                "measure the conversation by — turn-1 fit is priced before "
                "send (budget.price_payload), never asked of the conversation"
            )
        texts, images = [], []
        for entry in self._record[self._record_at_response :]:
            if entry[0] == "text":
                texts.append(entry[1])
            elif entry[0] == "tool":
                texts.append(entry[2])
            elif entry[0] == "image":
                images.append(entry[1])
        return (
            self._last_usage_total
            + (self._delta_text_tokens(texts) if texts else 0)
            + sum(self.image_tokens(path) for path in images)
        )

    @abstractmethod
    def _add_tool_result(self, call_id: str, content: str) -> None:
        """Append a tool-result turn answering the call with id call_id."""

    @abstractmethod
    def _get_response(
        self,
        *,
        response_schema: Any,
        tools: list | None = None,
    ) -> Response: ...

    def _config_manifest(self) -> dict:
        return {}

    def _persist_write(self, stem: str, suffix: str, content: str) -> None:
        with open(self._persist_dir / f"{stem}_{suffix}", "x") as f:
            f.write(content if content.endswith("\n") else content + "\n")

    def _persist_input(self, label: str | None) -> str | None:
        if self._persist_dir is None:
            return None
        self._persisted_calls += 1
        stem = (
            f"{self._persisted_calls}_{label}" if label else str(self._persisted_calls)
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        sections: list[str] = []
        role = None

        def emit(new_role: str, content: str) -> None:
            nonlocal role
            if new_role != role:
                sections.append(f"=== {new_role} ===")
                role = new_role
            sections.append(content)

        if self.system_prompt:
            emit("system", self.system_prompt)
        for entry in self._record:
            if entry[0] == "text":
                emit("user", entry[1])
            elif entry[0] == "image":
                shown = os.path.relpath(entry[1], self._persist_dir)
                emit("user", f'![{Path(entry[1]).name}]({shown} "sent as {entry[2]}")')
            elif entry[0] == "tool":
                emit("tool", f"[tool_result call_id={entry[1]}]\n{entry[2]}")
            elif entry[0] == "assistant_tool_calls":
                calls = "\n".join(
                    f"[tool_call id={c.id} {c.name}({json.dumps(c.arguments)})]"
                    for c in entry[1]
                )
                emit("assistant", (f"{entry[2]}\n{calls}" if entry[2] else calls))
            else:
                emit("assistant", entry[1])
        self._persist_write(stem, "input.txt", "\n\n".join(sections))
        return stem

    def _persist_result(
        self,
        stem: str | None,
        config: dict,
        label: str | None,
        response: Response | None = None,
        error: str | None = None,
        usage: dict | None = None,
        started_at: str | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        if stem is None:
            return
        if response is not None:
            self._persist_write(stem, "response.txt", reply_record(response))

        self._persist_write(
            stem,
            "meta.json",
            json.dumps(
                {
                    "call": self._persisted_calls,
                    "label": label,
                    "config": config,
                    "usage": response.usage_metadata if response is not None else usage,
                    "started": started_at,
                    "elapsed_s": None if elapsed_s is None else round(elapsed_s, 3),
                    "error": error,
                },
                indent=2,
                default=str,
            ),
        )

    @property
    @abstractmethod
    def execution_mode(self) -> str:
        """How this conversation produced its responses, recorded on every persisted payload so a reader knows what made them. Each implementation names its own, and a conversation that never reaches a model (pricing's tally, a test double) says so plainly."""
