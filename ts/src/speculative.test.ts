import { describe, expect, it } from "vitest";
import { MarkovModel } from "./models.js";
import {
  Trace,
  acceptProbability,
  expectedSpeedup,
  generateSpeculative,
  generateTargetOnly,
  residual,
  speculativeStep,
} from "./speculative.js";
import { mulberry32, scriptRng } from "./testUtil.js";

// targetModel and oneHotDraft form a hand-computable pair: the draft always
// proposes token 1 then token 2 from prefix [0], so scripted-RNG scenarios have
// a single deterministic outcome.
function targetModel(): MarkovModel {
  return new MarkovModel([
    [0.2, 0.5, 0.3],
    [0.6, 0.3, 0.1],
    [0.1, 0.2, 0.7],
  ], 0);
}

function oneHotDraft(): MarkovModel {
  return new MarkovModel([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
  ], 0);
}

describe("acceptProbability", () => {
  it("always accepts a token the target prefers", () => {
    expect(acceptProbability(0.6, 0.3)).toBe(1.0);
  });

  it("accepts partially when the draft over-proposes", () => {
    expect(Math.abs(acceptProbability(0.2, 0.8) - 0.25)).toBeLessThan(1e-15);
  });

  it("accepts a token the draft never proposes", () => {
    expect(acceptProbability(0.5, 0.0)).toBe(1.0);
  });

  it("accepts with probability one when equal", () => {
    expect(acceptProbability(0.4, 0.4)).toBe(1.0);
  });
});

describe("residual", () => {
  it("is the normalized positive part", () => {
    const r = residual([0.5, 0.3, 0.2], [0.1, 0.6, 0.3]);
    expect(Math.abs(r[0] - 1.0)).toBeLessThan(1e-15);
    expect(r[1]).toBe(0.0);
    expect(r[2]).toBe(0.0);
  });

  it("normalizes to one", () => {
    const r = residual([0.4, 0.4, 0.2], [0.1, 0.1, 0.1]);
    expect(Math.abs(r[0] + r[1] + r[2] - 1.0)).toBeLessThan(1e-15);
  });

  it("falls back to uniform when fully dominated", () => {
    expect(residual([0.1, 0.2], [0.9, 0.8])).toEqual([0.5, 0.5]);
  });
});

describe("speculativeStep", () => {
  it("emits the residual and stops on a forced rejection", () => {
    const trace = new Trace();
    const rng = scriptRng([0.0, 0.0, 0.1, 0.5, 0.1]);
    const emitted = speculativeStep(targetModel(), oneHotDraft(), [0], 2, rng, trace);
    expect(emitted).toEqual([1, 0]);
    expect(trace.accepted).toBe(1);
    expect(trace.proposed).toBe(2);
    expect(trace.draftPasses).toBe(2);
    expect(trace.targetPasses).toBe(1);
    expect(trace.steps).toBe(1);
    expect(trace.tokens).toEqual([1, 0]);
  });

  it("appends a bonus token after a full acceptance", () => {
    const trace = new Trace();
    const rng = scriptRng([0.0, 0.0, 0.1, 0.05, 0.5]);
    const emitted = speculativeStep(targetModel(), oneHotDraft(), [0], 2, rng, trace);
    expect(emitted).toEqual([1, 2, 2]);
    expect(trace.accepted).toBe(2);
  });

  it("emits between one and k+1 tokens", () => {
    const target = targetModel();
    const draft = oneHotDraft();
    const rng = mulberry32(42);
    for (let i = 0; i < 200; i++) {
      const emitted = speculativeStep(target, draft, [0], 4, rng);
      expect(emitted.length).toBeGreaterThanOrEqual(1);
      expect(emitted.length).toBeLessThanOrEqual(5);
    }
  });

  it("always emits k+1 for a perfect draft", () => {
    const m = targetModel();
    const rng = scriptRng([0.5]);
    for (let i = 0; i < 100; i++) {
      expect(speculativeStep(m, m, [0], 3, rng).length).toBe(4);
    }
  });
});

describe("generateSpeculative", () => {
  it("produces exactly the requested token count", () => {
    const m = targetModel();
    const trace = generateSpeculative(m, m, [0], 37, 4, scriptRng([0.5]));
    expect(trace.tokens.length).toBe(37);
  });

  it("rejects k below one", () => {
    const m = targetModel();
    expect(() => generateSpeculative(m, m, [0], 5, 0, scriptRng([0.5]))).toThrow();
  });

  it("uses fewer target passes than tokens", () => {
    const m = targetModel();
    const trace = generateSpeculative(m, m, [0], 1000, 4, scriptRng([0.5]));
    expect(trace.targetPasses).toBeLessThan(trace.tokens.length);
  });
});

describe("generateTargetOnly", () => {
  it("costs one pass per token", () => {
    const m = targetModel();
    const trace = generateTargetOnly(m, [0], 50, mulberry32(7));
    expect(trace.targetPasses).toBe(50);
    expect(trace.tokens.length).toBe(50);
    expect(trace.steps).toBe(50);
  });
});

describe("expectedSpeedup", () => {
  it("scales with tokens per step when the draft is free", () => {
    expect(expectedSpeedup(3.0, 4, 0.0)).toBe(3.0);
  });

  it("shrinks as the draft cost rises", () => {
    expect(expectedSpeedup(3.0, 4, 0.05)).toBeGreaterThan(expectedSpeedup(3.0, 4, 0.5));
  });
});

describe("Trace", () => {
  it("reports zero rates when empty", () => {
    const tr = new Trace();
    expect(tr.acceptanceRate).toBe(0.0);
    expect(tr.tokensPerStep).toBe(0.0);
  });

  it("computes acceptance rate and tokens per step", () => {
    const tr = new Trace();
    tr.tokens = [1, 2, 3, 4];
    tr.steps = 2;
    tr.proposed = 8;
    tr.accepted = 6;
    expect(Math.abs(tr.acceptanceRate - 0.75)).toBeLessThan(1e-15);
    expect(Math.abs(tr.tokensPerStep - 2.0)).toBeLessThan(1e-15);
  });
});
