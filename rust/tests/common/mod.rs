#![allow(dead_code)]
//! Deterministic RNG helpers shared by the integration tests.

use spec_decode::Rng;

/// Replays a fixed list of uniforms, cycling when exhausted, so hand-computed
/// accept/reject scenarios have a single deterministic outcome.
pub struct ScriptRng {
    vals: Vec<f64>,
    i: usize,
}

impl ScriptRng {
    pub fn new(vals: Vec<f64>) -> Self {
        Self { vals, i: 0 }
    }
}

impl Rng for ScriptRng {
    fn next_uniform(&mut self) -> f64 {
        let v = self.vals[self.i % self.vals.len()];
        self.i += 1;
        v
    }
}

/// A small deterministic uniform source for "many random trials" tests. It need
/// not match any other language's RNG; the invariants hold for any input stream.
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Rng for LcgRng {
    fn next_uniform(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}
