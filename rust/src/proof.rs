//! The correctness proof, made executable: the closed-form distribution the
//! sampler emits, which the theorem says equals the target's.

use crate::models::Model;
use crate::speculative::{accept_probability, residual};

/// Closed-form distribution of the token a speculative step emits, given target
/// distribution `p` and draft distribution `q` at that position. Summed over the
/// accepted and residual paths this equals `p` exactly — the exactness theorem.
pub fn induced_from_dists(p: &[f64], q: &[f64]) -> Vec<f64> {
    let mut accepted: Vec<f64> = Vec::with_capacity(p.len());
    let mut rejection_mass = 0.0;
    for (pi, qi) in p.iter().zip(q) {
        let a = accept_probability(*pi, *qi);
        accepted.push(qi * a);
        rejection_mass += qi * (1.0 - a);
    }
    let r = residual(p, q);
    accepted
        .iter()
        .zip(&r)
        .map(|(a, ri)| a + rejection_mass * ri)
        .collect()
}

/// The closed-form distribution of the next token a speculative step emits from
/// `prefix`, which the theorem says equals `target.next_dist(prefix)`.
pub fn induced_next_token_distribution(
    target: &dyn Model,
    draft: &dyn Model,
    prefix: &[usize],
) -> Vec<f64> {
    induced_from_dists(&target.next_dist(prefix), &draft.next_dist(prefix))
}

/// Largest total-variation distance between the induced emission distribution and
/// the target distribution across a set of prefixes. A correct implementation
/// returns ~0 (floating-point noise only).
pub fn max_total_variation(target: &dyn Model, draft: &dyn Model, prefixes: &[Vec<usize>]) -> f64 {
    let mut worst: f64 = 0.0;
    for prefix in prefixes {
        let induced = induced_next_token_distribution(target, draft, prefix);
        let p = target.next_dist(prefix);
        let tv: f64 = 0.5
            * induced
                .iter()
                .zip(&p)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
        worst = worst.max(tv);
    }
    worst
}
