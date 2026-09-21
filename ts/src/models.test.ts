import { describe, expect, it } from "vitest";
import { MarkovModel, normalize, sample } from "./models.js";
import { mulberry32, scriptRng } from "./testUtil.js";

describe("normalize", () => {
  it("turns a zero vector into a uniform distribution", () => {
    expect(normalize([0, 0, 0, 0])).toEqual([0.25, 0.25, 0.25, 0.25]);
  });

  it("scales weights to sum to one", () => {
    const out = normalize([1, 3]);
    expect(Math.abs(out[0] - 0.25)).toBeLessThan(1e-15);
    expect(Math.abs(out[1] - 0.75)).toBeLessThan(1e-15);
  });

  it("preserves an already-normalized distribution", () => {
    const input = [0.2, 0.3, 0.5];
    const out = normalize(input);
    input.forEach((v, i) => expect(Math.abs(out[i] - v)).toBeLessThan(1e-15));
  });

  it("treats a negative total as uniform", () => {
    expect(normalize([-1, -1])).toEqual([0.5, 0.5]);
  });
});

describe("MarkovModel", () => {
  it("uses the last token to pick the row", () => {
    const m = new MarkovModel([[0.1, 0.9], [0.7, 0.3]], 0);
    expect(m.nextDist([0])).toEqual([0.1, 0.9]);
    expect(m.nextDist([1])).toEqual([0.7, 0.3]);
  });

  it("uses the start token for an empty prefix", () => {
    const m = new MarkovModel([[0.1, 0.9], [0.7, 0.3]], 1);
    expect(m.nextDist([])).toEqual([0.7, 0.3]);
  });

  it("reports the vocabulary size", () => {
    const m = new MarkovModel([[0.5, 0.5], [0.5, 0.5]], 0);
    expect(m.vocabSize).toBe(2);
  });

  it("returns a fresh copy that cannot mutate the model", () => {
    const m = new MarkovModel([[0.1, 0.9], [0.7, 0.3]], 0);
    const got = m.nextDist([0]);
    got[0] = 42;
    expect(m.nextDist([0])[0]).toBe(0.1);
  });

  it("rejects a non-square matrix", () => {
    expect(() => new MarkovModel([[0.5, 0.5]], 0)).toThrow();
  });

  it("rejects rows that do not sum to one", () => {
    expect(() => new MarkovModel([[0.5, 0.6], [0.5, 0.5]], 0)).toThrow();
  });

  it("accepts tiny row-sum drift within tolerance", () => {
    const m = new MarkovModel([[0.5 + 1e-6, 0.5], [0.2, 0.8]], 0);
    expect(m.vocabSize).toBe(2);
  });

  it("rejects a small but real row-sum drift", () => {
    expect(() => new MarkovModel([[0.5 + 1e-4, 0.5], [0.2, 0.8]], 0)).toThrow();
  });
});

describe("sample", () => {
  it("uses inverse-CDF lookup", () => {
    const dist = [0.2, 0.5, 0.3];
    expect(sample(dist, scriptRng([0.1]))).toBe(0);
    expect(sample(dist, scriptRng([0.5]))).toBe(1);
    expect(sample(dist, scriptRng([0.95]))).toBe(2);
  });

  it("never returns a zero-probability token", () => {
    const rng = mulberry32(1);
    for (let i = 0; i < 1000; i++) {
      expect(sample([0, 1, 0], rng)).toBe(1);
    }
  });
});
