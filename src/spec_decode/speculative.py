"""Speculative decoding.

The idea in one line: a cheap **draft** model proposes several tokens; the
expensive **target** model checks them all in a single forward pass; and a
carefully designed accept/reject rule guarantees the tokens that come out are
distributed *exactly* as if they had been sampled from the target one at a time.
You get the target's quality at a fraction of its serial cost — with no
approximation.

This module implements the sampler from Leviathan et al., 2023 ("Fast Inference
from Transformers via Speculative Decoding") and Chen et al., 2023. The
correctness claim is not a slogan: ``proof.induced_next_token_distribution``
computes, in closed form, the distribution this sampler actually emits and the
test suite asserts it equals the target's — for arbitrary distributions.

Everything here is deterministic given a seeded RNG, so results are reproducible
and every branch is testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import Model, _normalize, sample


def residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The adjusted distribution to sample from after a rejection.

    When the draft's proposal is rejected we must *not* simply resample from the
    target, or the math breaks. We sample from the normalized positive part of
    ``p - q`` — the probability mass the target wanted that the draft under-
    supplied. This is the term that makes the whole scheme exact.
    """
    return _normalize(np.maximum(p - q, 0.0))


def accept_probability(p_t: float, q_t: float) -> float:
    """Probability of accepting a proposed token with target prob ``p_t`` and
    draft prob ``q_t``: ``min(1, p_t / q_t)``.

    A token the target likes at least as much as the draft is always accepted;
    one the draft over-proposes is accepted proportionally less often. ``q_t``
    is positive whenever the draft actually proposed the token.
    """
    if q_t <= 0.0:
        return 1.0
    return min(1.0, p_t / q_t)


@dataclass
class Trace:
    """Bookkeeping for one ``generate`` call — the raw material for measuring
    speedup honestly."""

    tokens: list[int] = field(default_factory=list)
    steps: int = 0
    target_passes: int = 0
    draft_passes: int = 0
    proposed: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_step(self) -> float:
        return len(self.tokens) / self.steps if self.steps else 0.0


def speculative_step(
    target: Model,
    draft: Model,
    prefix: list[int],
    k: int,
    rng: np.random.Generator,
    trace: Trace | None = None,
) -> list[int]:
    """Run one speculative step and return the tokens it emits (between 1 and
    ``k + 1`` of them).

    1. The draft autoregressively proposes ``k`` tokens, remembering the draft
       distribution used at each position.
    2. The target scores all ``k`` proposal positions plus one look-ahead
       position — in a real model this is a single batched forward pass; here it
       is ``k + 1`` cheap distribution lookups, counted as **one** target pass.
    3. Each proposal is accepted or rejected in order. The first rejection stops
       the step and emits a residual-sampled token in place of the rejected one.
    4. If every proposal is accepted, a bonus token is sampled from the target's
       look-ahead distribution — so a fully-accepted step of size ``k`` emits
       ``k + 1`` tokens for the price of one target pass.
    """
    # 1. Draft proposes k tokens.
    proposals: list[int] = []
    q_dists: list[np.ndarray] = []
    ctx = list(prefix)
    for _ in range(k):
        q = draft.next_dist(ctx)
        x = sample(q, rng)
        proposals.append(x)
        q_dists.append(q)
        ctx.append(x)
    if trace is not None:
        trace.draft_passes += k
        trace.proposed += k

    # 2. Target scores every proposal position and one look-ahead, in one pass.
    p_dists: list[np.ndarray] = []
    ctx = list(prefix)
    for i in range(k):
        p_dists.append(target.next_dist(ctx))
        ctx.append(proposals[i])
    p_lookahead = target.next_dist(ctx)  # position k+1
    if trace is not None:
        trace.target_passes += 1
        trace.steps += 1

    # 3. Accept/reject in order.
    emitted: list[int] = []
    for i in range(k):
        x = proposals[i]
        a = accept_probability(float(p_dists[i][x]), float(q_dists[i][x]))
        if rng.random() < a:
            emitted.append(x)
            if trace is not None:
                trace.accepted += 1
        else:
            # Rejected: emit a residual sample and stop this step.
            emitted.append(sample(residual(p_dists[i], q_dists[i]), rng))
            if trace is not None:
                trace.tokens.extend(emitted)
            return emitted

    # 4. All accepted: free bonus token from the look-ahead distribution.
    emitted.append(sample(p_lookahead, rng))
    if trace is not None:
        trace.tokens.extend(emitted)
    return emitted


def generate_speculative(
    target: Model,
    draft: Model,
    prefix: list[int],
    n_tokens: int,
    k: int,
    rng: np.random.Generator,
) -> Trace:
    """Generate ``n_tokens`` via speculative decoding, returning a full
    :class:`Trace` (tokens plus the pass/acceptance counters used for speedup)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    trace = Trace()
    ctx = list(prefix)
    while len(trace.tokens) < n_tokens:
        emitted = speculative_step(target, draft, ctx, k, rng, trace)
        ctx.extend(emitted)
    # A step can overshoot the requested count; trim so the output is exact.
    if len(trace.tokens) > n_tokens:
        trace.tokens = trace.tokens[:n_tokens]
    return trace


def generate_target_only(
    target: Model,
    prefix: list[int],
    n_tokens: int,
    rng: np.random.Generator,
) -> Trace:
    """The baseline: sample ``n_tokens`` from the target one at a time. Costs one
    target pass per token — the serial cost speculative decoding beats."""
    trace = Trace()
    ctx = list(prefix)
    for _ in range(n_tokens):
        p = target.next_dist(ctx)
        x = sample(p, rng)
        trace.tokens.append(x)
        trace.target_passes += 1
        trace.steps += 1
        ctx.append(x)
    return trace


def expected_speedup(tokens_per_step: float, k: int, draft_cost_ratio: float) -> float:
    """Wall-clock speedup implied by an observed ``tokens_per_step``.

    Per step the cost is one target pass plus ``k`` draft passes, where
    ``draft_cost_ratio`` = draft-pass-cost / target-pass-cost. Target-only would
    spend one target pass per emitted token. So::

        speedup = tokens_per_step / (1 + k * draft_cost_ratio)

    With a much cheaper draft (small ratio) the speedup approaches the average
    number of tokens emitted per step, capped at ``k + 1``.
    """
    return tokens_per_step / (1.0 + k * draft_cost_ratio)
