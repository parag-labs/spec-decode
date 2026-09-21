import { type Model, type Rng, normalize, sample } from "./models.js";

/**
 * Bookkeeping for a speculative-decoding run: the emitted tokens plus the
 * pass/acceptance counters used to measure efficiency.
 */
export class Trace {
  /** Tokens emitted so far, in order. */
  tokens: number[] = [];
  /** Number of speculative steps taken. */
  steps = 0;
  /** Number of target-model forward passes. */
  targetPasses = 0;
  /** Number of draft-model forward passes. */
  draftPasses = 0;
  /** Number of draft tokens proposed. */
  proposed = 0;
  /** Number of proposed tokens accepted. */
  accepted = 0;

  /** Fraction of proposed tokens that were accepted. */
  get acceptanceRate(): number {
    return this.proposed === 0 ? 0 : this.accepted / this.proposed;
  }

  /** Average number of emitted tokens per speculative step. */
  get tokensPerStep(): number {
    return this.steps === 0 ? 0 : this.tokens.length / this.steps;
  }
}

/**
 * Returns the normalised positive part of `p - q`: the residual distribution
 * sampled when a draft token is rejected.
 */
export function residual(p: readonly number[], q: readonly number[]): number[] {
  const diff = new Array<number>(p.length);
  for (let i = 0; i < diff.length; i++) {
    const d = p[i] - q[i];
    diff[i] = d > 0 ? d : 0;
  }
  return normalize(diff);
}

/**
 * Returns the probability of accepting a draft token whose target and draft
 * probabilities are `pt` and `qt`.
 */
export function acceptProbability(pt: number, qt: number): number {
  if (qt <= 0) {
    return 1.0;
  }
  return Math.min(1.0, pt / qt);
}

/**
 * Runs one speculative step: propose `k` draft tokens, score them against the
 * target, and emit the accepted prefix plus one bonus/residual token. Emits at
 * most `k + 1` tokens.
 */
export function speculativeStep(
  target: Model,
  draft: Model,
  prefix: readonly number[],
  k: number,
  rng: Rng,
  trace?: Trace,
): number[] {
  const proposals: number[] = [];
  const qDists: number[][] = [];
  let ctx = [...prefix];
  for (let i = 0; i < k; i++) {
    const q = draft.nextDist(ctx);
    const x = sample(q, rng);
    proposals.push(x);
    qDists.push(q);
    ctx.push(x);
  }
  if (trace) {
    trace.draftPasses += k;
    trace.proposed += k;
  }

  const pDists: number[][] = [];
  ctx = [...prefix];
  for (let i = 0; i < k; i++) {
    pDists.push(target.nextDist(ctx));
    ctx.push(proposals[i]);
  }
  const pLookahead = target.nextDist(ctx);
  if (trace) {
    trace.targetPasses += 1;
    trace.steps += 1;
  }

  const emitted: number[] = [];
  for (let i = 0; i < k; i++) {
    const x = proposals[i];
    const a = acceptProbability(pDists[i][x], qDists[i][x]);
    if (rng.nextUniform() < a) {
      emitted.push(x);
      if (trace) {
        trace.accepted += 1;
      }
    } else {
      emitted.push(sample(residual(pDists[i], qDists[i]), rng));
      if (trace) {
        trace.tokens.push(...emitted);
      }
      return emitted;
    }
  }
  emitted.push(sample(pLookahead, rng));
  if (trace) {
    trace.tokens.push(...emitted);
  }
  return emitted;
}

/**
 * Generates at least `nTokens` tokens via repeated speculative steps, trimming
 * any overshoot from the final step.
 *
 * @throws if `k < 1`.
 */
export function generateSpeculative(
  target: Model,
  draft: Model,
  prefix: readonly number[],
  nTokens: number,
  k: number,
  rng: Rng,
): Trace {
  if (k < 1) {
    throw new Error("k must be >= 1");
  }
  const trace = new Trace();
  const ctx = [...prefix];
  while (trace.tokens.length < nTokens) {
    const emitted = speculativeStep(target, draft, ctx, k, rng, trace);
    ctx.push(...emitted);
  }
  if (trace.tokens.length > nTokens) {
    trace.tokens.length = nTokens;
  }
  return trace;
}

/**
 * Generates `nTokens` tokens by sampling the target model directly, one forward
 * pass per token. Used as the baseline.
 */
export function generateTargetOnly(
  target: Model,
  prefix: readonly number[],
  nTokens: number,
  rng: Rng,
): Trace {
  const trace = new Trace();
  const ctx = [...prefix];
  for (let i = 0; i < nTokens; i++) {
    const p = target.nextDist(ctx);
    const x = sample(p, rng);
    trace.tokens.push(x);
    trace.targetPasses += 1;
    trace.steps += 1;
    ctx.push(x);
  }
  return trace;
}

/**
 * Returns the idealised expected speedup: emitted tokens per step divided by the
 * relative cost of one step (one target pass plus `k` draft passes each costing
 * `draftCostRatio`).
 */
export function expectedSpeedup(
  tokensPerStep: number,
  k: number,
  draftCostRatio: number,
): number {
  return tokensPerStep / (1.0 + k * draftCostRatio);
}
