"""Tests for the exactness guarantee — the reason to trust speculative decoding.

The headline test asserts, in closed form, that the sampler emits the target's
distribution for arbitrary (target, draft) pairs. The empirical tests confirm
the end-to-end generator matches target-only sampling within sampling noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from spec_decode import (
    draft_from_target,
    generate_speculative,
    generate_target_only,
    induced_from_dists,
    induced_next_token_distribution,
    max_total_variation,
    random_markov,
)
from spec_decode.models import _normalize


def _rand_dist(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.dirichlet(np.full(n, 0.5))


class TestExactnessClosedForm:
    def test_induced_equals_target_for_arbitrary_distributions(self) -> None:
        # The core theorem: for ANY target p and draft q, the emitted-token
        # distribution equals p. Checked in closed form over many random pairs.
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(2000):
            n = int(rng.integers(2, 12))
            p = _rand_dist(rng, n)
            q = _rand_dist(rng, n)
            induced = induced_from_dists(p, q)
            worst = max(worst, float(np.abs(induced - p).max()))
        assert worst < 1e-12, f"max deviation {worst} exceeds float tolerance"

    def test_induced_is_a_valid_distribution(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(500):
            n = int(rng.integers(2, 10))
            induced = induced_from_dists(_rand_dist(rng, n), _rand_dist(rng, n))
            assert np.all(induced >= -1e-15)
            assert induced.sum() == pytest.approx(1.0)

    def test_identical_draft_accepts_everything_and_stays_exact(self) -> None:
        # When q == p every proposal is accepted, yet the emitted distribution is
        # still exactly p.
        rng = np.random.default_rng(2)
        p = _rand_dist(rng, 8)
        induced = induced_from_dists(p, p.copy())
        assert np.allclose(induced, p)

    def test_degenerate_draft_still_exact(self) -> None:
        # A draft that puts all mass on one token (a terrible draft) is still
        # corrected to the target distribution.
        p = np.array([0.4, 0.3, 0.2, 0.1])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        induced = induced_from_dists(p, q)
        assert np.allclose(induced, p)

    def test_max_total_variation_is_zero_across_contexts(self) -> None:
        target = random_markov(vocab_size=10, seed=3)
        draft = draft_from_target(target, agreement=0.5, seed=4)
        prefixes = [[i] for i in range(target.vocab_size)] + [[], [1, 2, 3]]
        assert max_total_variation(target, draft, prefixes) < 1e-12

    def test_induced_next_token_matches_target_next_dist(self) -> None:
        target = random_markov(vocab_size=6, seed=5)
        draft = draft_from_target(target, agreement=0.7, seed=6)
        for prefix in ([], [0], [3], [5, 1]):
            induced = induced_next_token_distribution(target, draft, prefix)
            assert np.allclose(induced, target.next_dist(prefix))


class TestExactnessEmpirical:
    def test_generation_frequencies_match_target_only(self) -> None:
        # End-to-end: the first emitted token's empirical frequency from
        # speculative decoding matches target-only sampling from the same prefix.
        target = random_markov(vocab_size=5, seed=7)
        draft = draft_from_target(target, agreement=0.6, seed=8)
        prefix = [2]
        trials = 40000

        spec_counts = np.zeros(target.vocab_size)
        base_counts = np.zeros(target.vocab_size)
        rng_s = np.random.default_rng(100)
        rng_b = np.random.default_rng(200)
        for _ in range(trials):
            spec_counts[generate_speculative(target, draft, prefix, 1, k=4, rng=rng_s).tokens[0]] += 1
            base_counts[generate_target_only(target, prefix, 1, rng=rng_b).tokens[0]] += 1

        spec_freq = spec_counts / trials
        target_dist = target.next_dist(prefix)
        # Within a few standard errors of the true distribution.
        assert np.abs(spec_freq - target_dist).max() < 0.02

    def test_multitoken_stream_stays_close_to_target(self) -> None:
        target = random_markov(vocab_size=4, seed=9)
        draft = draft_from_target(target, agreement=0.8, seed=10)
        rng_s = np.random.default_rng(1)
        rng_b = np.random.default_rng(1)
        spec = generate_speculative(target, draft, [0], 5000, k=3, rng=rng_s).tokens
        base = generate_target_only(target, [0], 5000, rng=rng_b).tokens
        # Overall token histograms agree.
        sh = np.bincount(spec, minlength=4) / len(spec)
        bh = np.bincount(base, minlength=4) / len(base)
        assert np.abs(sh - bh).max() < 0.03


def test_normalize_handles_zero_vector() -> None:
    out = _normalize(np.zeros(4))
    assert np.allclose(out, 0.25)
