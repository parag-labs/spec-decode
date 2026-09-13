"""Tests for the sampler mechanics and the speedup accounting."""

from __future__ import annotations

import numpy as np
import pytest

from spec_decode import (
    accept_probability,
    draft_from_target,
    expected_speedup,
    generate_speculative,
    generate_target_only,
    random_markov,
    residual,
)
from spec_decode.speculative import speculative_step


class TestAcceptProbability:
    def test_accepts_when_target_prefers_token(self) -> None:
        assert accept_probability(0.6, 0.3) == 1.0

    def test_partial_accept_when_draft_overproposes(self) -> None:
        assert accept_probability(0.2, 0.8) == pytest.approx(0.25)

    def test_zero_draft_prob_accepts(self) -> None:
        # A token the draft never proposes can't be rejected here.
        assert accept_probability(0.5, 0.0) == 1.0


class TestResidual:
    def test_residual_is_positive_part_normalized(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        r = residual(p, q)
        # only the first component has p>q
        assert r[0] == pytest.approx(1.0)
        assert r[1] == 0.0 and r[2] == 0.0

    def test_residual_normalizes_to_one(self) -> None:
        p = np.array([0.4, 0.4, 0.2])
        q = np.array([0.1, 0.1, 0.1])
        assert residual(p, q).sum() == pytest.approx(1.0)


class TestStep:
    def test_step_emits_between_one_and_k_plus_one(self) -> None:
        target = random_markov(6, seed=1)
        draft = draft_from_target(target, 0.5, seed=2)
        rng = np.random.default_rng(0)
        for _ in range(200):
            emitted = speculative_step(target, draft, [0], k=4, rng=rng)
            assert 1 <= len(emitted) <= 5

    def test_perfect_draft_always_emits_k_plus_one(self) -> None:
        # An identical draft is accepted every time, so each step emits k+1.
        target = random_markov(6, seed=3)
        draft = draft_from_target(target, agreement=1.0, seed=4)
        rng = np.random.default_rng(0)
        lengths = [len(speculative_step(target, draft, [0], k=3, rng=rng)) for _ in range(100)]
        assert set(lengths) == {4}


class TestGenerate:
    def test_generates_exact_token_count(self) -> None:
        target = random_markov(5, seed=5)
        draft = draft_from_target(target, 0.6, seed=6)
        rng = np.random.default_rng(0)
        trace = generate_speculative(target, draft, [0], n_tokens=37, k=4, rng=rng)
        assert len(trace.tokens) == 37

    def test_higher_agreement_raises_acceptance_and_tokens_per_step(self) -> None:
        target = random_markov(8, seed=7)
        rng = np.random.default_rng(0)
        low = generate_speculative(target, draft_from_target(target, 0.3, 8), [0], 3000, 4, rng)
        rng = np.random.default_rng(0)
        high = generate_speculative(target, draft_from_target(target, 0.9, 9), [0], 3000, 4, rng)
        assert high.acceptance_rate > low.acceptance_rate
        assert high.tokens_per_step > low.tokens_per_step

    def test_target_only_costs_one_pass_per_token(self) -> None:
        target = random_markov(5, seed=10)
        rng = np.random.default_rng(0)
        trace = generate_target_only(target, [0], 50, rng)
        assert trace.target_passes == 50
        assert len(trace.tokens) == 50

    def test_speculative_uses_fewer_target_passes_than_tokens(self) -> None:
        target = random_markov(8, seed=11)
        draft = draft_from_target(target, 0.8, 12)
        rng = np.random.default_rng(0)
        trace = generate_speculative(target, draft, [0], 1000, k=4, rng=rng)
        # The entire point: fewer expensive passes than tokens produced.
        assert trace.target_passes < len(trace.tokens)


class TestSpeedup:
    def test_speedup_scales_with_tokens_per_step(self) -> None:
        assert expected_speedup(3.0, k=4, draft_cost_ratio=0.0) == 3.0

    def test_draft_cost_reduces_speedup(self) -> None:
        cheap = expected_speedup(3.0, k=4, draft_cost_ratio=0.05)
        pricey = expected_speedup(3.0, k=4, draft_cost_ratio=0.5)
        assert cheap > pricey
