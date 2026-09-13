"""Benchmark: measured speedup of speculative decoding as a function of how well
the draft tracks the target, and of the speculation length k.

Everything is measured on a plain CPU from this committed script — run it
yourself with ``python bench/benchmark.py``. It writes a JSON summary and, if
matplotlib is installed, two graphs under ``bench/results/``.

The numbers are honest about their model: we count *target forward passes* (the
expensive part) exactly, and convert to a wall-clock speedup with an explicit
``draft_cost_ratio`` (how cheap the draft is relative to the target). We do not
claim a hardware number we didn't measure — we measure the algorithmic quantity
that hardware speedup is proportional to (tokens emitted per target pass) and
state the cost assumption in the open.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spec_decode import (
    draft_from_target,
    expected_speedup,
    generate_speculative,
    generate_target_only,
    random_markov,
)

VOCAB = 32
N_TOKENS = 20000
AGREEMENTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
KS = [1, 2, 4, 8]
DRAFT_COST_RATIO = 0.1  # draft pass costs 10% of a target pass
SEED = 20240601


def run() -> dict:
    target = random_markov(VOCAB, seed=SEED, concentration=0.3)

    # Sanity: target-only baseline costs one pass per token.
    base = generate_target_only(target, [0], N_TOKENS, np.random.default_rng(SEED))
    assert base.target_passes == N_TOKENS

    by_k: dict[int, list[dict]] = {}
    for k in KS:
        rows = []
        for a in AGREEMENTS:
            draft = draft_from_target(target, agreement=a, seed=SEED + 1)
            rng = np.random.default_rng(SEED + 2)
            tr = generate_speculative(target, draft, [0], N_TOKENS, k=k, rng=rng)
            speedup = expected_speedup(tr.tokens_per_step, k, DRAFT_COST_RATIO)
            rows.append(
                {
                    "agreement": a,
                    "acceptance_rate": round(tr.acceptance_rate, 4),
                    "tokens_per_step": round(tr.tokens_per_step, 4),
                    "target_passes": tr.target_passes,
                    "tokens_per_target_pass": round(len(tr.tokens) / tr.target_passes, 4),
                    "speedup": round(speedup, 4),
                }
            )
        by_k[k] = rows

    return {
        "config": {
            "vocab": VOCAB,
            "n_tokens": N_TOKENS,
            "draft_cost_ratio": DRAFT_COST_RATIO,
            "seed": SEED,
        },
        "by_k": by_k,
    }


def plot(data: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping graphs")
        return

    # Speedup vs acceptance rate, one line per k.
    plt.figure(figsize=(7, 4.5))
    for k, rows in data["by_k"].items():
        xs = [r["acceptance_rate"] for r in rows]
        ys = [r["speedup"] for r in rows]
        plt.plot(xs, ys, marker="o", label=f"k = {k}")
    plt.axhline(1.0, color="gray", ls="--", lw=1, label="target-only (1x)")
    plt.xlabel("acceptance rate (how often the draft is right)")
    plt.ylabel(f"speedup vs target-only (draft cost = {data['config']['draft_cost_ratio']}x)")
    plt.title("Speculative decoding speedup")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_vs_acceptance.png", dpi=130)
    plt.close()

    # Tokens per target pass vs k, at a few agreements.
    plt.figure(figsize=(7, 4.5))
    ks = list(data["by_k"].keys())
    for a_idx, label in [(-1, "0.95"), (-3, "0.8"), (-6, "0.5")]:
        ys = [data["by_k"][k][a_idx]["tokens_per_target_pass"] for k in ks]
        plt.plot(ks, ys, marker="s", label=f"acceptance ~ {label}")
    plt.xlabel("speculation length k")
    plt.ylabel("tokens emitted per target pass")
    plt.title("How many tokens each expensive pass buys")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tokens_per_pass_vs_k.png", dpi=130)
    plt.close()
    print(f"wrote graphs to {out_dir}")


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    data = run()
    (out_dir / "summary.json").write_text(json.dumps(data, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")

    # Print a compact table for k=4.
    print("\nk=4:  acceptance  tokens/step  tokens/pass  speedup")
    for r in data["by_k"][4]:
        print(
            f"      {r['acceptance_rate']:.2f}         "
            f"{r['tokens_per_step']:.2f}          "
            f"{r['tokens_per_target_pass']:.2f}        {r['speedup']:.2f}x"
        )
    plot(data, out_dir)


if __name__ == "__main__":
    main()
