import { describe, expect, it } from "vitest";
import { MarkovModel, normalize } from "./models.js";
import {
  inducedFromDists,
  inducedNextTokenDistribution,
  maxTotalVariation,
} from "./proof.js";
import { mulberry32 } from "./testUtil.js";

function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > worst) {
      worst = d;
    }
  }
  return worst;
}

function randomDist(rng: { nextUniform(): number }, n: number): number[] {
  const raw = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    raw[i] = rng.nextUniform();
  }
  return normalize(raw);
}

function targetModel(): MarkovModel {
  return new MarkovModel([
    [0.2, 0.5, 0.3],
    [0.6, 0.3, 0.1],
    [0.1, 0.2, 0.7],
  ], 0);
}

function differentDraft(): MarkovModel {
  return new MarkovModel([
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3],
    [0.5, 0.25, 0.25],
  ], 0);
}

describe("inducedFromDists", () => {
  it("equals the target for arbitrary distributions", () => {
    // The core theorem: for any target p and draft q, the emitted-token
    // distribution equals p. Checked over many pseudo-random pairs.
    const rng = mulberry32(0);
    let worst = 0;
    for (let trial = 0; trial < 5000; trial++) {
      const n = 2 + Math.floor(rng.nextUniform() * 10.0);
      const p = randomDist(rng, n);
      const q = randomDist(rng, n);
      worst = Math.max(worst, maxAbsDiff(inducedFromDists(p, q), p));
    }
    expect(worst).toBeLessThan(1e-12);
  });

  it("is a valid probability distribution", () => {
    const rng = mulberry32(1);
    for (let trial = 0; trial < 500; trial++) {
      const n = 2 + Math.floor(rng.nextUniform() * 8.0);
      const p = randomDist(rng, n);
      const q = randomDist(rng, n);
      const induced = inducedFromDists(p, q);
      let sum = 0;
      for (const v of induced) {
        expect(v).toBeGreaterThanOrEqual(-1e-15);
        sum += v;
      }
      expect(Math.abs(sum - 1.0)).toBeLessThan(1e-12);
    }
  });

  it("stays exact for an identical draft", () => {
    const p = normalize([0.4, 0.1, 0.25, 0.25]);
    expect(maxAbsDiff(inducedFromDists(p, [...p]), p)).toBeLessThan(1e-15);
  });

  it("corrects a degenerate draft back to the target", () => {
    const p = [0.4, 0.3, 0.2, 0.1];
    const q = [1.0, 0.0, 0.0, 0.0];
    expect(maxAbsDiff(inducedFromDists(p, q), p)).toBeLessThan(1e-15);
  });
});

describe("maxTotalVariation", () => {
  it("is zero across contexts", () => {
    const prefixes = [[], [0], [1], [2], [2, 1]];
    expect(maxTotalVariation(targetModel(), differentDraft(), prefixes)).toBeLessThan(1e-12);
  });

  it("makes the induced next-token distribution match the target", () => {
    const target = targetModel();
    const draft = differentDraft();
    for (const prefix of [[], [0], [1], [2]]) {
      const induced = inducedNextTokenDistribution(target, draft, prefix);
      expect(maxAbsDiff(induced, target.nextDist(prefix))).toBeLessThan(1e-12);
    }
  });
});
