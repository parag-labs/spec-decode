//! The speculative-sampling accept/reject rule, one step, generation, and the
//! cost accounting.

use crate::models::{normalize, sample, Model, Rng};

/// The adjusted distribution to sample from after a rejection: the normalized
/// positive part of `p - q`. This is the term that makes the scheme exact.
pub fn residual(p: &[f64], q: &[f64]) -> Vec<f64> {
    let diff: Vec<f64> = p.iter().zip(q).map(|(pi, qi)| (pi - qi).max(0.0)).collect();
    normalize(&diff)
}

/// Probability of accepting a proposed token: `min(1, pt/qt)`. A non-positive
/// `qt` (a token the draft never proposes) is accepted with probability 1.
pub fn accept_probability(pt: f64, qt: f64) -> f64 {
    if qt <= 0.0 {
        1.0
    } else {
        (pt / qt).min(1.0)
    }
}

/// Token output plus pass/acceptance counters for one generate call.
#[derive(Debug, Default, Clone)]
pub struct Trace {
    /// Emitted tokens.
    pub tokens: Vec<usize>,
    /// Number of speculative (or target-only) steps taken.
    pub steps: usize,
    /// Number of target forward passes.
    pub target_passes: usize,
    /// Number of draft forward passes.
    pub draft_passes: usize,
    /// Number of tokens proposed by the draft.
    pub proposed: usize,
    /// Number of proposals accepted.
    pub accepted: usize,
}

impl Trace {
    /// Accepted divided by proposed, or 0 when nothing was proposed.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// Tokens divided by steps, or 0 when there were no steps.
    pub fn tokens_per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.tokens.len() as f64 / self.steps as f64
        }
    }
}

/// Runs one speculative step from `prefix` and returns the tokens it emits
/// (between 1 and `k + 1`). When `trace` is `Some`, its counters and token list
/// are updated. The first rejection emits a residual sample and stops the step; a
/// fully accepted step appends a bonus token from the look-ahead distribution.
pub fn speculative_step(
    target: &dyn Model,
    draft: &dyn Model,
    prefix: &[usize],
    k: usize,
    rng: &mut dyn Rng,
    mut trace: Option<&mut Trace>,
) -> Vec<usize> {
    let mut proposals: Vec<usize> = Vec::with_capacity(k);
    let mut q_dists: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut ctx = prefix.to_vec();
    for _ in 0..k {
        let q = draft.next_dist(&ctx);
        let x = sample(&q, rng);
        proposals.push(x);
        q_dists.push(q);
        ctx.push(x);
    }
    if let Some(t) = trace.as_deref_mut() {
        t.draft_passes += k;
        t.proposed += k;
    }

    let mut p_dists: Vec<Vec<f64>> = Vec::with_capacity(k);
    ctx = prefix.to_vec();
    for &prop in &proposals {
        p_dists.push(target.next_dist(&ctx));
        ctx.push(prop);
    }
    let p_lookahead = target.next_dist(&ctx);
    if let Some(t) = trace.as_deref_mut() {
        t.target_passes += 1;
        t.steps += 1;
    }

    let mut emitted: Vec<usize> = Vec::with_capacity(k + 1);
    for i in 0..k {
        let x = proposals[i];
        let a = accept_probability(p_dists[i][x], q_dists[i][x]);
        if rng.next_uniform() < a {
            emitted.push(x);
            if let Some(t) = trace.as_deref_mut() {
                t.accepted += 1;
            }
        } else {
            emitted.push(sample(&residual(&p_dists[i], &q_dists[i]), rng));
            if let Some(t) = trace.as_deref_mut() {
                t.tokens.extend_from_slice(&emitted);
            }
            return emitted;
        }
    }

    emitted.push(sample(&p_lookahead, rng));
    if let Some(t) = trace {
        t.tokens.extend_from_slice(&emitted);
    }
    emitted
}

/// Generates `n_tokens` via speculative decoding and returns the full trace.
/// Returns `Err` when `k == 0`. A step can overshoot the requested count, so the
/// token list is trimmed to exactly `n_tokens`.
pub fn generate_speculative(
    target: &dyn Model,
    draft: &dyn Model,
    prefix: &[usize],
    n_tokens: usize,
    k: usize,
    rng: &mut dyn Rng,
) -> Result<Trace, String> {
    if k == 0 {
        return Err("k must be >= 1".to_string());
    }
    let mut trace = Trace::default();
    let mut ctx = prefix.to_vec();
    while trace.tokens.len() < n_tokens {
        let emitted = speculative_step(target, draft, &ctx, k, rng, Some(&mut trace));
        ctx.extend_from_slice(&emitted);
    }
    if trace.tokens.len() > n_tokens {
        trace.tokens.truncate(n_tokens);
    }
    Ok(trace)
}

/// The baseline: sample `n_tokens` from the target one at a time, costing one
/// target pass per token.
pub fn generate_target_only(
    target: &dyn Model,
    prefix: &[usize],
    n_tokens: usize,
    rng: &mut dyn Rng,
) -> Trace {
    let mut trace = Trace::default();
    let mut ctx = prefix.to_vec();
    for _ in 0..n_tokens {
        let p = target.next_dist(&ctx);
        let x = sample(&p, rng);
        trace.tokens.push(x);
        trace.target_passes += 1;
        trace.steps += 1;
        ctx.push(x);
    }
    trace
}

/// The wall-clock speedup implied by an observed `tokens_per_step`:
/// `tokens_per_step / (1 + k * draft_cost_ratio)`.
pub fn expected_speedup(tokens_per_step: f64, k: usize, draft_cost_ratio: f64) -> f64 {
    tokens_per_step / (1.0 + k as f64 * draft_cost_ratio)
}
