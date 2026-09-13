"""spec-decode — speculative decoding from scratch, provably exact.

A cheap draft model proposes tokens; the target verifies them in one pass; the
emitted tokens are distributed exactly as sampling the target alone.

Public API::

    from spec_decode import (
        MarkovModel, random_markov, draft_from_target,
        generate_speculative, generate_target_only, expected_speedup,
        induced_next_token_distribution, max_total_variation,
    )
"""

from .models import (
    MarkovModel,
    Model,
    draft_from_target,
    random_markov,
    sample,
)
from .proof import (
    induced_from_dists,
    induced_next_token_distribution,
    max_total_variation,
)
from .speculative import (
    Trace,
    accept_probability,
    expected_speedup,
    generate_speculative,
    generate_target_only,
    residual,
    speculative_step,
)

__all__ = [
    "MarkovModel",
    "Model",
    "Trace",
    "accept_probability",
    "draft_from_target",
    "expected_speedup",
    "generate_speculative",
    "generate_target_only",
    "induced_from_dists",
    "induced_next_token_distribution",
    "max_total_variation",
    "random_markov",
    "residual",
    "sample",
    "speculative_step",
]
