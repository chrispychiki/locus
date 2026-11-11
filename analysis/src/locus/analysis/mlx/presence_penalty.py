"""The presence penalty over the tokens the model has generated, and none of the prompt.

Qwen's recommended presence_penalty assumes this: vLLM and SGLang, the engines its model cards name, penalize only generated tokens. Upstream's server penalizes the last presence_context_size tokens of the whole sequence, prompt included, so it penalizes the names, ids and tool syntax the model has to quote back. Here, thinking tokens count as generated, and a request's presence_context_size is ignored.

Every upstream source the processor relies on is pinned; on drift the server refuses to boot and names what to re-verify (patching.py).
"""

EXPECTED_SHA256 = {
    "sample_utils.make_logits_processors": "c816942e4b33b27b3191b2ce3d52d237b965f2fc7933b3dad36bcf58af249bcb",
    "sample_utils.make_presence_penalty": "b8d09ef9aab984beaac7fa3448f80f115118f011914602444c3c3b2afb3d6371",
    "ResponseGenerator._make_logits_processors": "dbcfb849a7cd2af08d5c0cb5dd7bd12ed5ebe7646c32cc99bbb3ef2e3b697a97",
    "ResponseGenerator._start_batch_request": "af6ab4d3a1d425e23e8e0adf84998904587ab27859ef7951a314194f159503b7",
    "PromptProcessingBatch.__init__": "7ff5403e3d4852a0ef0021f0ea6050755bf73e7d4143478e6c7c8f1e885526fa",
    "PromptProcessingBatch.generate": "b83bcedbc806108884b94af4056f190a9984791cf75d713e8eb8778b88c07760",
    "GenerationBatch._step": "51cc7f9893b16eccfcff77c223cc6fe2f5c9aa69b46f75f221ff34390ac89115",
    "GenerationBatch.__init__": "5006caa3328ea8b7c385e22433f79fa4246687b199917cb7b666d39128d4386a",
    "GenerationBatch.empty": "9b3e514b61ca851ab8b34fe75d764461f1d73095d66b92501fc396bc56ffeeac",
    "GenerationBatch.extend": "7b02a4a8095b3153e78d096be82647c59c6c6665652476bdd8d43a632ddb4a67",
    "GenerationBatch.filter": "fa446fc00888a747bb5eff74ca474a1347a0edf139b0203d069dc07677cf079f",
    "GenerationBatch._fused_greedy_step": "87a0e4039a33b88b161e75cd19c5b52e59d0a265387525975056c3eac8541fc8",
    "BatchGenerator.insert": "09ee02ba820ad0958a9e42c77ba4687b297eec4a7a9887a8d68315c3dcc99ce4",
    "qwen3_5 LanguageModel.supports_fused_greedy_logits_processors": "df63f6a674b66568d2952313dea129cdc6b71ea6f9642fe64e93bf73ec7dc0f7",
}

OWES = (
    "The processor rests on make_logits_processors calling the module's make_presence_penalty "
    "once per request build, on each admitted request building its own processors, on the prompt "
    "batch calling each processor once over the prompt ids before the first draw, on the "
    "generation batch keeping each request's processors in its own slot as requests are inserted, "
    "joined and dropped, on that batch handing each drawn token to process_last_token once, and on "
    "its fused greedy decode declining the processor (the served Qwen3.5-family model accepts only "
    "processors that build a next-token mask) so a greedy draw takes that path — re-verify those "
    "against the installed version, then re-pin (analysis/src/locus/analysis/mlx/presence_penalty.py)."
)


class GeneratedPresencePenalty:
    """One request's presence penalty: `penalty` off the logit of every token it has generated."""

    def __init__(self, penalty: float):
        self.penalty = penalty
        self.prompt_length: int | None = None
        self.generated: list[int] = []
        self._seen: set[int] = set()

    def __call__(self, tokens, logits):
        """A call with the whole sequence so far: the first marks where generation starts and penalizes nothing; a later one takes the generated tokens from the sequence."""
        if self.prompt_length is None:
            self.prompt_length = len(tokens)
        else:
            for token in tokens[self.prompt_length :].tolist():
                self._add(token)
        return self._penalized(logits)

    def process_last_token(self, token: int, logits):
        """A call with the token just generated."""
        self._add(token)
        return self._penalized(logits)

    def _add(self, token: int) -> None:
        if token not in self._seen:
            self._seen.add(token)
            self.generated.append(token)

    def _penalized(self, logits):
        import mlx.core as mx

        if self.generated:
            logits[:, mx.array(self.generated)] -= self.penalty
        return logits


def make_presence_penalty(penalty: float, context_size: int = 20):
    """Upstream's factory signature, so make_logits_processors calls it unchanged; context_size has no meaning over the generated tokens."""
    return GeneratedPresencePenalty(penalty)


def install_presence_penalty() -> None:
    """Rebind upstream's presence-penalty factory to the generated-tokens processor."""
    from mlx_vlm import sample_utils
    from mlx_vlm.generate import ar
    from mlx_vlm.models.qwen3_5 import language as qwen3_5
    from mlx_vlm.server import generation

    from .patching import expect_source

    for what, fn in (
        ("sample_utils.make_logits_processors", sample_utils.make_logits_processors),
        ("sample_utils.make_presence_penalty", sample_utils.make_presence_penalty),
        (
            "ResponseGenerator._make_logits_processors",
            generation.ResponseGenerator._make_logits_processors,
        ),
        (
            "ResponseGenerator._start_batch_request",
            generation.ResponseGenerator._start_batch_request,
        ),
        ("PromptProcessingBatch.__init__", ar.PromptProcessingBatch.__init__),
        ("PromptProcessingBatch.generate", ar.PromptProcessingBatch.generate),
        ("GenerationBatch._step", ar.GenerationBatch._step),
        ("GenerationBatch.__init__", ar.GenerationBatch.__init__),
        ("GenerationBatch.empty", ar.GenerationBatch.empty),
        ("GenerationBatch.extend", ar.GenerationBatch.extend),
        ("GenerationBatch.filter", ar.GenerationBatch.filter),
        ("GenerationBatch._fused_greedy_step", ar.GenerationBatch._fused_greedy_step),
        ("BatchGenerator.insert", ar.BatchGenerator.insert),
        (
            "qwen3_5 LanguageModel.supports_fused_greedy_logits_processors",
            qwen3_5.LanguageModel.supports_fused_greedy_logits_processors,
        ),
    ):
        expect_source("presence penalty", what, fn, EXPECTED_SHA256[what], OWES)
    sample_utils.make_presence_penalty = make_presence_penalty
