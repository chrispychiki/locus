"""Unit tests for the presence-penalty patch — no model checkpoint, no server.

The patch's claims are that a request's penalty lands on every token it has generated and on none of its prompt, the same whichever way upstream hands the processor its tokens, that upstream's own factory call builds it, and that drift in any source it rests on refuses the boot.
"""

import locus.analysis.mlx.presence_penalty as pp
import mlx.core as mx
import pytest
from locus.analysis.mlx.presence_penalty import (
    GeneratedPresencePenalty,
    install_presence_penalty,
)
from mlx_vlm import sample_utils

VOCAB = 16
PROMPT = [3, 4, 5]


@pytest.fixture(autouse=True)
def restore_binding():
    original = sample_utils.make_presence_penalty
    yield
    sample_utils.make_presence_penalty = original


def zeros():
    return mx.zeros((1, VOCAB))


def penalized(logits) -> set[int]:
    return {i for i, value in enumerate(logits[0].tolist()) if value < 0}


def test_the_first_draw_penalizes_nothing_of_the_prompt():
    processor = GeneratedPresencePenalty(1.5)
    assert penalized(processor(mx.array(PROMPT), zeros())) == set()


def test_every_generated_token_is_penalized_once_and_the_prompt_never():
    processor = GeneratedPresencePenalty(1.5)
    processor(mx.array(PROMPT), zeros())
    logits = None
    for token in [7, 3, 7, 9]:
        logits = processor.process_last_token(token, zeros())
    assert penalized(logits) == {7, 3, 9}, (
        "a prompt token is penalized once the model has generated it"
    )
    assert logits[0, 7].item() == pytest.approx(-1.5), "presence, not frequency"
    assert penalized(logits).isdisjoint({4, 5}), "the prompt alone penalizes nothing"


def test_the_whole_sequence_form_agrees_with_the_last_token_form():
    by_token, by_sequence = GeneratedPresencePenalty(1.5), GeneratedPresencePenalty(1.5)
    by_token(mx.array(PROMPT), zeros())
    by_sequence(mx.array(PROMPT), zeros())
    sequence = list(PROMPT)
    for token in [7, 3, 7, 9, 11]:
        sequence.append(token)
        a = by_token.process_last_token(token, zeros())
        b = by_sequence(mx.array(sequence), zeros())
        assert mx.array_equal(a, b)


class FlatModel:
    """Every logit equal, so a greedy draw takes the lowest token id the penalty leaves untouched."""

    def __call__(self, inputs, cache=None, **kwargs):
        return mx.zeros((inputs.shape[0], 1, VOCAB))


def test_upstreams_generation_step_draws_past_every_generated_token_and_no_prompt_token():
    from mlx_vlm.generate.ar import GenerationBatch

    processor = GeneratedPresencePenalty(1.5)
    processor(mx.array(PROMPT), zeros())
    first = 7
    batch = GenerationBatch(
        FlatModel(),
        [0],
        mx.array([first]),
        [],
        lambda logprobs: mx.argmax(logprobs, axis=-1),
        None,
        [16],
        greedy_sampling=True,
        token_context=[PROMPT],
        logits_processors=[[processor]],
    )
    drawn = [batch._step()[0][0] for _ in range(7)]
    assert drawn == [first, 0, 1, 2, 3, 4, 5], (
        "each drawn token is penalized from the next step on; the prompt's 3, 4, 5 are "
        "drawn only once the model has generated everything below them"
    )


def test_qwens_fused_greedy_decode_declines_it_so_a_greedy_draw_runs_it():
    from types import SimpleNamespace

    from mlx_vlm.models.qwen3_5.language import LanguageModel

    assert not LanguageModel.supports_fused_greedy_logits_processors(
        SimpleNamespace(), [[GeneratedPresencePenalty(1.5)]]
    )


def test_upstreams_factory_call_builds_it_and_each_request_gets_its_own():
    install_presence_penalty()
    first = sample_utils.make_logits_processors(presence_penalty=1.5)
    second = sample_utils.make_logits_processors(presence_penalty=1.5)
    assert [type(p) for p in first] == [GeneratedPresencePenalty]
    assert first[0] is not second[0]
    assert sample_utils.make_logits_processors(presence_penalty=0.0) == []


@pytest.mark.parametrize("what", sorted(pp.EXPECTED_SHA256))
def test_refuses_boot_on_upstream_source_drift(monkeypatch, what):
    monkeypatch.setitem(pp.EXPECTED_SHA256, what, "0" * 64)
    before = sample_utils.make_presence_penalty
    with pytest.raises(SystemExit, match="upstream drift"):
        install_presence_penalty()
    assert sample_utils.make_presence_penalty is before
