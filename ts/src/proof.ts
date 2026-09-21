import { type Model } from "./models.js";
import { acceptProbability, residual } from "./speculative.js";

/**
 * The exactness proof, expressed constructively. Speculative decoding is
 * distribution-preserving: the token actually emitted for a given context is
 * distributed exactly according to the target model, regardless of the draft.
 * These helpers compute that induced distribution in closed form so tests can
 * assert it matches the target to floating-point tolerance.
 */

/**
 * Returns the closed-form distribution of the token emitted by one accept/reject
 * round given target distribution `p` and draft distribution `q`. Equals `p`
 * exactly.
 */
export function inducedFromDists(p: readonly number[], q: readonly number[]): number[] {
  const n = p.length;
  const accepted = new Array<number>(n);
  let rejectionMass = 0;
  for (let i = 0; i < n; i++) {
    const a = acceptProbability(p[i], q[i]);
    accepted[i] = q[i] * a;
    rejectionMass += q[i] * (1 - a);
  }
  const r = residual(p, q);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = accepted[i] + rejectionMass * r[i];
  }
  return out;
}

/**
 * Returns the induced next-token distribution for a concrete target/draft pair
 * and token `prefix`.
 */
export function inducedNextTokenDistribution(
  target: Model,
  draft: Model,
  prefix: readonly number[],
): number[] {
  return inducedFromDists(target.nextDist(prefix), draft.nextDist(prefix));
}

/**
 * Returns the maximum total-variation distance between the induced distribution
 * and the target distribution across the supplied `prefixes`. For a correct
 * sampler this is zero up to rounding.
 */
export function maxTotalVariation(
  target: Model,
  draft: Model,
  prefixes: readonly (readonly number[])[],
): number {
  let worst = 0;
  for (const prefix of prefixes) {
    const induced = inducedNextTokenDistribution(target, draft, prefix);
    const p = target.nextDist(prefix);
    let tv = 0;
    for (let i = 0; i < p.length; i++) {
      tv += Math.abs(induced[i] - p[i]);
    }
    tv *= 0.5;
    if (tv > worst) {
      worst = tv;
    }
  }
  return worst;
}
