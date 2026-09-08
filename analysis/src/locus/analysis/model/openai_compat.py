"""Local OpenAI-compatible backend — the fully-offline path (bundled mlx).

Client and server share a filesystem: images travel as plain absolute paths and the server reads them from disk, so request size never bounds visual context. Contracts with the locus-mlx server (upstream mlx-vlm behind a thin operational layer), preserved exactly:

- consecutive user text/images merge into one user message (the chat template expects a single interleaved user turn);
- the conversation only ever grows: every part is appended ahead of the call that sends it and stays, so each request's token sequence extends the last one's rather than rewriting its middle — the shape any server-side prefix cache reuses on, where one is enabled (analysis/README.md § Throughput, caching, concurrency);
- thinking arrives as `reasoning` on the message (`reasoning_content` is read first for compatibility); assistant replay turns carry both keys, and the server hands `reasoning_content` to the chat template, which renders it by the model family's own convention (Qwen renders it only for an assistant turn that comes after the conversation's last user turn, so a reply the conversation has since asked something about replays as its answer alone);
- the default timeout is None — infinite — because local generation can outrun any fixed budget;
- every request is priced before send and refused loud when the count plus max_tokens exceeds the card's context_tokens — before paying tokenization and upload for a doomed request. The turn-1 count walks the composed parts with the composition plane's own instruments (budget.py: the pinned tokenizer for text, the smart_resize closed form over real image dimensions — exact there, because everything on turn 1 is client-authored text and images); every later count, the refusal's and context_remaining's alike, anchors on the last response's usage total — the wire's own record — plus the priced delta of parts appended since it. The server enforces the same bound when MAX_KV_SIZE is set (its count includes expanded vision tokens); the client refusal is the cheap, informative half. The client arithmetic is one model family's wire behavior — the card declares its membership (`family`, validated in cards.py), so a model outside the family is refused at card load rather than silently priced and resized with math that is not its own;
- images are carried under the card's per-image token ceilings (`max_image_tokens`, one per kind of screenshot — the pushed activity set and the served pulls), realized by _within_ceiling;
- every request streams (`stream_options.include_usage`) and is reassembled here into the same persisted record a non-streamed call would leave: content and reasoning deltas concatenated, tool-call deltas merged by index, usage from the terminal usage chunk. Streaming is what makes a long generation legible while it runs — the deltas are the run's progress, and the server serves one request at a time from admission to the end of its response;
- that terminal usage chunk carries a `timings` sibling to `usage` (prompt/generation token rates and ms, cache hits, peak model memory) that the openai SDK does not model; it survives as `model_extra` and is folded into the persisted usage record — the local analogue of Gemini's usage breakdown. The response's `x-locus-request` header is folded in too (`server_request`): it is the id keying the server's per-request memory telemetry, so the persisted record names its own rows in the server's memory log.

Config is stated, never defaulted: the conversation is constructed from a model card (cards.py) and sends the model, the thinking budget, and every sampling value explicitly on every request — thinking is the backend's one mode, and enable_thinking rides the wire as an explicit true rather than falling to an upstream default. The persisted payload is therefore the complete record of the generation intent — nothing the server applies is absent from it. Omitted fields would fall to upstream's own schema defaults (temperature 0.0 — greedy).

Payload persistence is inherited from the Conversation protocol layer — the record home is stated at construction (record_dir), never optional.
"""

import json
from pathlib import Path
from typing import Any

from .protocol import Conversation, DryRun, Response, json_schema_of

NATIVE_PARAMS = {"temperature", "top_p", "presence_penalty"}


def _json_schema_format(schema: Any) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "strict": False,
            "schema": json_schema_of(schema),
        },
    }


class OpenAICompatConversation(Conversation):
    def __init__(
        self,
        system_prompt: str,
        *,
        card: dict,
        record_dir: str | Path,
        base_url: str | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
    ):
        from openai import OpenAI

        super().__init__(record_dir, system_prompt)
        # The card owns the endpoint — the same base_url the bundled server binds at boot.
        # An explicit base_url is only for a caller that intentionally redirects (tests).
        if base_url is None:
            try:
                base_url = card["base_url"]
            except KeyError:
                raise ValueError(
                    f"card for {card.get('name', card.get('model'))} declares "
                    f"no base_url — the local endpoint is the card's to "
                    f"declare"
                ) from None
        self.client = OpenAI(base_url=base_url, api_key="unused", timeout=timeout)
        self.base_url = base_url
        self.model = card["model"]
        self.thinking_budget = card["thinking_budget"]
        self.max_tokens = card["max_tokens"]
        self.max_image_tokens = dict(card["max_image_tokens"])
        self.context_tokens = card["context_tokens"]
        self.sampling = dict(card["sampling"])
        self.card_name = card.get("name", card["model"])
        self.prices = card.get("pricing")
        self.dry_run = dry_run
        self.contents: list[dict] = []
        self._tmpdir: str | None = None

    def _user_parts(self) -> list[dict]:
        if self.contents and self.contents[-1]["role"] == "user":
            return self.contents[-1]["content"]
        message = {"role": "user", "content": []}
        self.contents.append(message)
        return message["content"]

    def _add_user_text(self, text: str) -> None:
        self._user_parts().append({"type": "text", "text": text})

    def _add_user_image(self, image_path: str, pushed: bool) -> str:
        """The image rides under its kind's ceiling — activity_screenshots for a pushed image, pulled_screenshots for everything else."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        url = self._within_ceiling(path, self._ceiling(pushed))
        self._user_parts().append(
            {
                "type": "image_url",
                "image_url": {"url": url},
            }
        )
        return url

    def _ceiling(self, pushed: bool) -> int:
        return self.max_image_tokens[
            "activity_screenshots" if pushed else "pulled_screenshots"
        ]

    def _within_ceiling(self, path: Path, ceiling: int) -> str:
        """The card's per-image token ceiling for the image's kind, realized: an image under it is sent as its own file untouched; over it, the server is handed a physically smaller copy at the geometry the ceiling holds (budget.image_geometry — the same rule pricing counts, so the price is the price of the file sent). The server has no per-image API knob, so a smaller file is the only realization. The copy lives in a temp dir tied to this conversation's lifetime; the image it came from stays in the analysis directory and the persisted record names the copy actually sent, so the copy itself carries nothing the audit trail needs."""
        import shutil
        import tempfile
        import weakref

        from PIL import Image

        from ..budget import image_geometry

        with Image.open(path) as img:
            geometry = image_geometry(*img.size, ceiling)
            if geometry == img.size:
                return str(path.resolve())
            shrunk = img.resize(geometry, Image.LANCZOS)
            if self._tmpdir is None:
                self._tmpdir = tempfile.mkdtemp(prefix="locus_images_")
                weakref.finalize(self, shutil.rmtree, self._tmpdir, True)
            out = Path(self._tmpdir) / f"{path.stem}_{ceiling}tok{path.suffix}"
            shrunk.save(out)
            return str(out)

    def image_tokens(self, image_path: str) -> int:
        """What a pull costs: the geometry the pulled_screenshots ceiling sends it at. The pushed set is never priced here — it is priced as a window before turn 1 (budget.price_payload), at the activity ceiling."""
        from PIL import Image

        from ..budget import image_geometry, qwen_image_tokens

        with Image.open(image_path) as image:
            return qwen_image_tokens(
                *image_geometry(*image.size, self._ceiling(pushed=False))
            )

    def context_remaining(self) -> int:
        """What the model's context still holds for this conversation to grow into: context_tokens minus max_tokens (generation shares the context here, so the next call's ceiling stays reserved) minus the wire-anchored size — the last response's usage total, whose completion half counts thinking the template never replays, so the anchor errs conservative — plus the priced delta of parts appended since it (tool-result text through the pinned tokenizer, images at the geometry the card sends)."""
        return self.context_tokens - self.max_tokens - self._wire_anchored_used()

    def _billing(self) -> dict | None:
        """The card decides: the shipped local cards declare no pricing block, so their calls are free and the spend layer never engages; a card that does declare one has every call governed at its rates — which is also how the spend machinery is proven live for zero dollars, on a priced copy of a local card against the real server."""
        if self.prices is None:
            return None
        return {"model": self.card_name, "model_id": self.model, "prices": self.prices}

    def _spend_usage(self, usage_metadata: dict) -> dict:
        """The OpenAI usage shape into the ledger's vocabulary: prompt_tokens is everything sent, completion_tokens is everything generated with thinking inside it (the template streams thought and answer as one completion, undissectable here). No cache field is read — nothing on this wire bills a cached tier."""
        return {
            "input_tokens": usage_metadata.get("prompt_tokens") or 0,
            "cached_tokens": 0,
            "output_tokens": usage_metadata.get("completion_tokens") or 0,
            "thoughts_tokens": 0,
        }

    def _usage_total(self, usage_metadata: dict) -> int | None:
        return usage_metadata.get("total_tokens")

    def _delta_text_tokens(self, texts: list[str]) -> int:
        from ..budget import count_text_tokens

        return sum(count_text_tokens(text) for text in texts)

    def _add_tool_result(self, call_id: str, content: str) -> None:
        self.contents.append(
            {"role": "tool", "tool_call_id": call_id, "content": content}
        )

    def _wire_messages(self) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.contents)
        return messages

    def _request_params(self) -> dict:
        params: dict = {
            "model": self.model,
            "messages": self._wire_messages(),
            "max_tokens": self.max_tokens,
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget": self.thinking_budget,
            },
        }
        for key, value in self.sampling.items():
            if key in NATIVE_PARAMS:
                params[key] = value
            else:
                params["extra_body"][key] = value
        return params

    def _prompt_tokens(self) -> int:
        """What the conversation as it stands costs on the wire. After a response exists the count anchors on the wire's own record (_wire_anchored_used); before one — turn 1, where every part is client-authored text and images — the composed parts are walked with the composition plane's instruments (the pinned tokenizer, the smart_resize closed form)."""
        from PIL import Image

        from ..budget import count_text_tokens, qwen_image_tokens

        if self._last_usage_total is not None:
            return self._wire_anchored_used()

        text = count_text_tokens(self.system_prompt or "")
        images = 0
        for message in self.contents:
            content = message["content"]
            if isinstance(content, str):
                text += count_text_tokens(content)
                continue
            for part in content:
                if part["type"] == "text":
                    text += count_text_tokens(part["text"])
                else:
                    with Image.open(part["image_url"]["url"]) as image:
                        images += qwen_image_tokens(*image.size)
        return text + images

    def _refuse_oversized(self) -> None:
        prompt = self._prompt_tokens()
        if prompt + self.max_tokens > self.context_tokens:
            raise ValueError(
                f"request refused before send: {prompt} prompt tokens + "
                f"max_tokens {self.max_tokens} exceeds the model context "
                f"{self.context_tokens}. Compose a smaller window, or widen "
                f"the screenshot interval."
            )

    def _get_response(
        self,
        *,
        response_schema: Any = None,
        tools: list | None = None,
    ) -> Response:
        from .protocol import ToolCall

        if not self.contents:
            raise ValueError("conversation has no user content")
        self._refuse_oversized()

        params = self._request_params()
        if tools:
            params["tools"] = [{"type": "function", "function": t} for t in tools]
        if response_schema is not None:
            params["response_format"] = _json_schema_format(response_schema)

        if self.dry_run:
            raise DryRun(
                f"dry run: request to {self.model} assembled and persisted, not sent"
            )

        text_parts: list[str] = []
        thought_parts: list[str] = []
        calls_by_index: dict[int, dict] = {}
        usage_metadata: dict = {}

        # The phases the wire makes visible: the response headers arrive once the server has
        # admitted the request (mlx/serve.py holds the headers until then), and the
        # first delta is the first generated token — prefill over.
        self.progress = {
            "phase": "waiting",
            "line": "waiting for the server's admission",
        }
        # with_raw_response for the headers: x-locus-request, the server's telemetry id.
        raw = self.client.chat.completions.with_raw_response.create(
            stream=True, stream_options={"include_usage": True}, **params
        )
        server_request = raw.headers.get("x-locus-request")
        self.progress = {
            "phase": "prefilling",
            "server_request": server_request,
            "line": f"admitted as server request {server_request}, prefilling",
        }
        deltas = 0
        stream = raw.parse()
        with stream:
            for chunk in stream:
                error = (chunk.model_extra or {}).get("error")
                if error:
                    raise RuntimeError(f"server error mid-stream: {error}")
                if chunk.usage is not None:
                    usage_metadata = chunk.usage.model_dump(exclude_none=True)
                    timings = (chunk.model_extra or {}).get("timings")
                    if timings:
                        usage_metadata["timings"] = timings
                for choice in chunk.choices or []:
                    delta = choice.delta
                    if delta is None:
                        continue
                    deltas += 1
                    self.progress = {
                        "phase": "generating",
                        "server_request": server_request,
                        "line": f"server request {server_request} generating, {deltas} delta(s) received",
                    }
                    if delta.content:
                        text_parts.append(delta.content)
                    thought = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if thought:
                        thought_parts.append(thought)
                    for tc in delta.tool_calls or []:
                        slot = calls_by_index.setdefault(
                            tc.index, {"id": None, "name": None, "arguments": []}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"].append(tc.function.arguments)

        if server_request:
            usage_metadata["server_request"] = server_request
        # The server's non-streamed path strips both fields after splitting the
        # think block (responses_state._split_thinking); the deltas carry the
        # raw boundary whitespace, so strip here to reassemble the same record.
        text = "".join(text_parts).strip()
        thoughts = "".join(thought_parts).strip()
        wire_calls = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {
                    "name": slot["name"],
                    "arguments": "".join(slot["arguments"]),
                },
            }
            for _, slot in sorted(calls_by_index.items())
        ]

        def parsed_arguments(raw: str) -> dict | None:
            """The call's argument object, or None when the streamed text is not one — the protocol's defect marker, answered in the tool's own channel rather than crashing a run minutes in on the model's own malformed writing."""
            try:
                arguments = json.loads(raw or "{}")
            except json.JSONDecodeError:
                return None
            return arguments if isinstance(arguments, dict) else None

        tool_calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=parsed_arguments(call["function"]["arguments"]),
            )
            for call in wire_calls
        ]

        response = Response(
            text=text,
            thoughts_text=thoughts,
            usage_metadata=usage_metadata,
            tool_calls=tool_calls,
        )

        if tool_calls:
            self.contents.append(
                {"role": "assistant", "content": text, "tool_calls": wire_calls}
            )
        elif response.text:
            turn: dict = {"role": "assistant", "content": response.text}
            if response.thoughts_text:
                turn["reasoning_content"] = response.thoughts_text
                turn["reasoning"] = response.thoughts_text
            self.contents.append(turn)
        return response

    def progress_line(self) -> str:
        """While the call waits for admission, the line also says what the server holds — its /health carries the admission state — so a wait reads as a line with a length, never as a hang."""
        line = self.progress.get("line", "")
        if self.progress.get("phase") != "waiting":
            return line
        state = self._server_admission()
        if state is None:
            return line
        return (
            f"{line}: it is serving {state['running']} request with "
            f"{state['waiting']} waiting, one at a time in arrival order"
        )

    def _server_admission(self) -> dict | None:
        return server_admission(self.base_url)

    def _config_manifest(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "enable_thinking": True,
            "thinking_budget": self.thinking_budget,
            "max_tokens": self.max_tokens,
            "context_tokens": self.context_tokens,
            **self.sampling,
        }

    @property
    def execution_mode(self) -> str:
        return "stream"


def server_admission(base_url: str) -> dict | None:
    """The server's admission state right now — its /health `admission` block (`running`, `waiting`) — or None when nothing answers at the card's base_url: no server, or one still loading its weights."""
    from urllib.error import URLError
    from urllib.request import urlopen

    health = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        with urlopen(health, timeout=2) as response:
            return json.load(response).get("admission")
    except (URLError, OSError, ValueError):
        return None


def measured_serving(analyses_dir: Path, model_id: str, last: int = 5) -> dict | None:
    """How this machine has served this model: the medians over the most recent `last` completed answer-turn calls (`analyses/*/*_meta.json`, whose `usage.timings` is the server's own) of the prefill rate in prompt tokens per second, the generation rate, the tokens an answer generated, and the served time — the server's prompt_ms + predicted_ms, not the client's wall-clock, which is mostly the wait for admission. None until a call has completed here."""
    import statistics

    calls = []
    for meta_path in analyses_dir.glob("*/*_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        timings = (meta.get("usage") or {}).get("timings")
        if (
            not timings
            or meta.get("error")
            or (meta.get("config") or {}).get("model") != model_id
            or not meta.get("started")
        ):
            continue
        calls.append(
            {
                "started": meta["started"],
                "prompt_tokens_per_s": timings["prompt_per_second"],
                "generated_tokens_per_s": timings["predicted_per_second"],
                "generated_tokens": timings["predicted_n"],
                "served_s": (timings["prompt_ms"] + timings["predicted_ms"]) / 1000,
            }
        )
    if not calls:
        return None
    recent = sorted(calls, key=lambda c: c["started"])[-last:]
    return {
        "calls": len(recent),
        "prompt_tokens_per_s": round(
            statistics.median(c["prompt_tokens_per_s"] for c in recent)
        ),
        "generated_tokens_per_s": round(
            statistics.median(c["generated_tokens_per_s"] for c in recent)
        ),
        "generated_tokens_median": round(
            statistics.median(c["generated_tokens"] for c in recent)
        ),
        "served_s_median": round(statistics.median(c["served_s"] for c in recent)),
        "since": recent[0]["started"],
    }
