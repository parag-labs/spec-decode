/**
 * A source of uniform random doubles in the half-open interval [0, 1).
 *
 * The speculative sampler is parameterised over this interface so that its
 * accept/reject logic is fully deterministic and testable.
 */
export interface Rng {
  /** Returns the next uniform sample in [0, 1). */
  nextUniform(): number;
}

/**
 * An autoregressive model: given a token prefix it yields the probability
 * distribution over the next token.
 */
export interface Model {
  /** Number of tokens in the vocabulary. */
  readonly vocabSize: number;
  /**
   * Returns the next-token distribution for `tokens`. The returned array is a
   * fresh copy the caller may mutate.
   */
  nextDist(tokens: readonly number[]): number[];
}

// numpy.allclose default tolerance: atol + rtol against a target of 1.0.
const ROW_SUM_TOLERANCE = 1e-8 + 1e-5;

/**
 * A first-order Markov model backed by an explicit row-stochastic transition
 * matrix. This is the deterministic, dependency-free core used by the exactness
 * proof and the sampler tests.
 */
export class MarkovModel implements Model {
  private readonly transition: number[][];
  private readonly startToken: number;
  readonly vocabSize: number;

  /**
   * Builds a model from a square transition matrix whose rows sum to 1.
   *
   * @throws if the matrix is not square or a row does not sum to 1.
   */
  constructor(transition: readonly (readonly number[])[], startToken = 0) {
    const n = transition.length;
    if (n === 0) {
      throw new Error("transition must be a non-empty square matrix");
    }
    this.transition = [];
    for (let i = 0; i < n; i++) {
      const row = transition[i];
      if (row.length !== n) {
        throw new Error("transition must be a square matrix");
      }
      let sum = 0;
      for (const v of row) {
        sum += v;
      }
      if (Math.abs(sum - 1.0) > ROW_SUM_TOLERANCE) {
        throw new Error(`transition row ${i} must sum to 1, got ${sum}`);
      }
      this.transition.push([...row]);
    }
    this.vocabSize = n;
    this.startToken = startToken;
  }

  nextDist(tokens: readonly number[]): number[] {
    const last = tokens.length > 0 ? tokens[tokens.length - 1] : this.startToken;
    return [...this.transition[last]];
  }
}

/**
 * Returns a probability distribution proportional to `dist`. A vector whose
 * entries sum to zero (or less) normalises to uniform.
 */
export function normalize(dist: readonly number[]): number[] {
  const n = dist.length;
  let total = 0;
  for (const v of dist) {
    total += v;
  }
  const out = new Array<number>(n);
  if (total <= 0) {
    const u = 1 / n;
    for (let i = 0; i < n; i++) {
      out[i] = u;
    }
    return out;
  }
  for (let i = 0; i < n; i++) {
    out[i] = dist[i] / total;
  }
  return out;
}

/**
 * Draws an index from `dist` using one uniform sample and inverse-CDF lookup.
 * The distribution is normalised first, so an index with zero probability is
 * never returned.
 */
export function sample(dist: readonly number[], rng: Rng): number {
  const nd = normalize(dist);
  const u = rng.nextUniform();
  let cum = 0;
  for (let i = 0; i < nd.length; i++) {
    cum += nd[i];
    if (u < cum) {
      return i;
    }
  }
  return nd.length - 1;
}
