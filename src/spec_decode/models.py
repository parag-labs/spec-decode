"""Deterministic toy models for speculative decoding.

Speculative decoding is a property of the *sampling procedure*, not of any
particular network — so to study and test it we don't need a GPU or real
weights. All we need from a "model" is the one thing the algorithm consumes: a
next-token probability distribution given a prefix. These toy models provide
exactly that, deterministically, over a small vocabulary, so the whole thing is
unit-testable and reproducible.

The same code path runs unchanged against a real model: implement ``next_dist``
to call your network's forward pass and return a probability vector.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

Tokens = tuple[int, ...]


class Model(Protocol):
    """A next-token distribution oracle.

    ``next_dist`` must return a 1-D probability vector over the vocabulary
    (non-negative, sums to 1) for the given prefix. It must be a pure function
    of ``tokens`` — the acceptance theorem assumes the target's distribution at
    a position does not depend on how that position was reached.
    """

    vocab_size: int

    def next_dist(self, tokens: Sequence[int]) -> np.ndarray: ...


class MarkovModel:
    """A first-order Markov model: the next-token distribution depends only on
    the last token, given by a row-stochastic transition matrix.

    It is a stand-in for a real model that is rich enough to make speculative
    decoding non-trivial (the distribution genuinely varies with context) while
    staying fully deterministic and cheap.
    """

    def __init__(self, transition: np.ndarray, start_token: int = 0) -> None:
        transition = np.asarray(transition, dtype=np.float64)
        if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
            raise ValueError("transition must be a square matrix")
        row_sums = transition.sum(axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("transition rows must sum to 1")
        self.transition = transition
        self.vocab_size = transition.shape[0]
        self.start_token = start_token

    def next_dist(self, tokens: Sequence[int]) -> np.ndarray:
        last = tokens[-1] if len(tokens) > 0 else self.start_token
        return self.transition[last].copy()


def random_markov(vocab_size: int, seed: int, concentration: float = 0.4) -> MarkovModel:
    """A random row-stochastic transition matrix.

    ``concentration`` controls peakiness: a smaller Dirichlet parameter makes
    each row more concentrated on a few tokens (more predictable text), which
    raises the achievable acceptance rate.
    """
    rng = np.random.default_rng(seed)
    rows = rng.dirichlet(np.full(vocab_size, concentration), size=vocab_size)
    return MarkovModel(rows)


def draft_from_target(target: MarkovModel, agreement: float, seed: int) -> MarkovModel:
    """Build a draft model that agrees with the target to a controllable degree.

    The draft's rows are a convex mix of the target's rows and independent
    random noise::

        draft_row = agreement * target_row + (1 - agreement) * noise_row

    ``agreement`` in [0, 1] tunes how often the draft's proposals will be
    accepted: 1.0 makes the draft identical to the target (every token
    accepted), 0.0 makes it independent noise (low acceptance). This is the
    single knob the benchmark sweeps to show speedup as a function of how well a
    cheap draft tracks the target.
    """
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("agreement must be in [0, 1]")
    rng = np.random.default_rng(seed)
    noise = rng.dirichlet(np.full(target.vocab_size, 0.4), size=target.vocab_size)
    rows = agreement * target.transition + (1.0 - agreement) * noise
    rows = rows / rows.sum(axis=1, keepdims=True)
    return MarkovModel(rows)


def sample(dist: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one token index from a probability vector."""
    return int(rng.choice(len(dist), p=_normalize(dist)))


def _normalize(dist: np.ndarray) -> np.ndarray:
    total = dist.sum()
    if total <= 0:
        # Degenerate residual (target ⊆ draft support); fall back to uniform.
        return np.full_like(dist, 1.0 / len(dist))
    return dist / total
