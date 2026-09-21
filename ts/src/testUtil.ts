import { type Rng } from "./models.js";

/** Replays a fixed, cycling list of uniforms for exact scenarios. */
export function scriptRng(values: number[]): Rng {
  let index = 0;
  return {
    nextUniform(): number {
      const v = values[index % values.length];
      index++;
      return v;
    },
  };
}

/**
 * A small 32-bit generator (mulberry32). Deterministic and dependency free; only
 * used to drive statistical "many trials" tests. PRNG streams need not match the
 * other language ports — only the scripted deterministic scenarios do.
 */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return {
    nextUniform(): number {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
}
