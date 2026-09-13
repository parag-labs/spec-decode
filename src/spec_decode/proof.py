"""The correctness proof, made executable.

Speculative decoding's whole appeal is that it is **exact**: the tokens it emits
are distributed identically to sampling the target model directly, with no
approximation and no quality loss. That is a theorem — but a theorem is only as
trustworthy as the implementation that claims to follow it. This module closes
the gap by computing, in closed form, the distribution the sampler *actually*
produces, so the test suite can assert it equals the target's distribution for
arbitrary inputs.

The key observation: the first token a speculative step emits is decided
entirely at the first proposal position. Either the draft's first proposal is
accepted (and emitted), or it is rejected and a residual token is emitted in its
place — the step stops on the first rejection, so later positions never affect
the first emitted token. Every emitted token is a "first token" relative to its
own prefix (an accepted token, a residual token, or the bonus token), so proving
the first-token marginal for arbitrary distributions proves the scheme end to
end.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .models import Model, _normalize
from .speculative import accept_probability, residual


def induced_from_dists(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Closed-form distribution of the token a speculative step emits, given the
    target distribution ``p`` and the draft distribution ``q`` at that position.

    Emitting token ``t`` happens two mutually exclusive ways:

    * the draft proposes ``t`` (prob ``q[t]``) and it is accepted
      (prob ``min(1, p[t]/q[t])``); or
    * *some* proposal is rejected (total prob ``sum_x q[x]·(1 - accept(x))``)
      and the residual sample lands on ``t`` (prob ``residual(p, q)[t]``).

    Summing these gives, exactly, ``p`` — which the tests assert.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    accept = np.array([accept_probability(float(p[i]), float(q[i])) for i in range(len(p))])

    accepted_mass = q * accept  # prob of accepting each specific token
    rejection_mass = float((q * (1.0 - accept)).sum())
    r = residual(p, q)
    return accepted_mass + rejection_mass * r


def induced_next_token_distribution(
    target: Model, draft: Model, prefix: Sequence[int]
) -> np.ndarray:
    """The closed-form distribution of the next token a speculative step emits
    from ``prefix`` — which the theorem says equals ``target.next_dist(prefix)``.
    """
    p = target.next_dist(prefix)
    q = draft.next_dist(prefix)
    return induced_from_dists(p, q)


def max_total_variation(
    target: Model,
    draft: Model,
    prefixes: Sequence[Sequence[int]],
) -> float:
    """Largest total-variation distance between the induced emission distribution
    and the target distribution across a set of prefixes.

    A correct implementation returns ~0 (floating-point noise only). This is the
    single number that certifies exactness over a whole context set.
    """
    worst = 0.0
    for prefix in prefixes:
        induced = induced_next_token_distribution(target, draft, prefix)
        p = target.next_dist(prefix)
        tv = 0.5 * float(np.abs(induced - p).sum())
        worst = max(worst, tv)
    return worst
