"""Command-line demo for spec-decode.

  spec-decode verify   [--vocab N] [--trials N]   # certify exactness numerically
  spec-decode run      [--agreement A] [--k K] [--tokens N]   # show a run + speedup
  spec-decode sweep    [--k K]                     # speedup across acceptance rates

There is no GPU and no download: the "models" are deterministic toy
distributions, which is enough to demonstrate and verify the *sampling*
algorithm — the part speculative decoding is actually about.
"""

from __future__ import annotations

import argparse

import numpy as np

from .models import draft_from_target, random_markov
from .proof import max_total_variation
from .speculative import expected_speedup, generate_speculative, generate_target_only


def _verify(args: argparse.Namespace) -> None:
    target = random_markov(args.vocab, seed=1, concentration=0.3)
    # Certify exactness across a range of drafts and every single-token context.
    prefixes = [[i] for i in range(args.vocab)] + [[]]
    worst = 0.0
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        draft = draft_from_target(target, agreement=a, seed=2)
        worst = max(worst, max_total_variation(target, draft, prefixes))
    print(f"vocab={args.vocab}, drafts at agreement 0..1")
    print(f"max total-variation distance (induced vs target): {worst:.2e}")
    if worst < 1e-12:
        print("PASS — the emitted distribution equals the target's, exactly.")
    else:
        print("FAIL — implementation does not match the theorem.")


def _run(args: argparse.Namespace) -> None:
    target = random_markov(args.vocab, seed=1, concentration=0.3)
    draft = draft_from_target(target, agreement=args.agreement, seed=2)
    rng = np.random.default_rng(0)
    tr = generate_speculative(target, draft, [0], args.tokens, k=args.k, rng=rng)
    speedup = expected_speedup(tr.tokens_per_step, args.k, args.draft_cost_ratio)

    base = generate_target_only(target, [0], args.tokens, np.random.default_rng(0))
    print(f"generated {len(tr.tokens)} tokens  (draft agreement {args.agreement}, k={args.k})")
    print(f"  target passes: {tr.target_passes:>6}   (target-only would need {base.target_passes})")
    print(f"  acceptance rate: {tr.acceptance_rate:.3f}")
    print(f"  tokens / step:   {tr.tokens_per_step:.3f}")
    print(f"  tokens / target pass: {len(tr.tokens) / tr.target_passes:.3f}")
    print(f"  speedup (draft cost {args.draft_cost_ratio}x): {speedup:.2f}x")


def _sweep(args: argparse.Namespace) -> None:
    target = random_markov(32, seed=1, concentration=0.3)
    print(f"k={args.k}   acceptance   tokens/step   speedup")
    for a in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        draft = draft_from_target(target, agreement=a, seed=2)
        rng = np.random.default_rng(0)
        tr = generate_speculative(target, draft, [0], args.tokens, k=args.k, rng=rng)
        speedup = expected_speedup(tr.tokens_per_step, args.k, args.draft_cost_ratio)
        print(f"      {tr.acceptance_rate:>6.3f}       {tr.tokens_per_step:>6.3f}      {speedup:>5.2f}x")


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--vocab", type=int, default=32)
    common.add_argument("--tokens", type=int, default=5000)
    common.add_argument("--k", type=int, default=4, help="speculation length")
    common.add_argument("--agreement", type=float, default=0.8, help="draft/target agreement in [0,1]")
    common.add_argument("--draft-cost-ratio", type=float, default=0.1)

    p = argparse.ArgumentParser(prog="spec-decode", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", parents=[common], help="numerically certify exactness")
    sub.add_parser("run", parents=[common], help="run one generation and report speedup")
    sub.add_parser("sweep", parents=[common], help="speedup across acceptance rates")

    args = p.parse_args(argv)
    {"verify": _verify, "run": _run, "sweep": _sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
