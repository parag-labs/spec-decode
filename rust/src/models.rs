//! Models and the sampling primitives they feed.

/// The single source of randomness the sampler consumes. `next_uniform` must
/// return a value in the half-open interval `[0, 1)`. Abstracting the RNG keeps
/// the accept/reject logic pure and deterministically testable without
/// reproducing NumPy's PCG64 byte stream.
pub trait Rng {
    /// Returns the next uniform draw in `[0, 1)`.
    fn next_uniform(&mut self) -> f64;
}

/// A next-token distribution oracle. `next_dist` must be a pure function of the
/// prefix — the acceptance theorem assumes the target's distribution at a
/// position does not depend on how that position was reached.
pub trait Model {
    /// Number of tokens in the vocabulary.
    fn vocab_size(&self) -> usize;
    /// Probability vector over the vocabulary for the given prefix.
    fn next_dist(&self, tokens: &[usize]) -> Vec<f64>;
}

/// Returns `dist` scaled to sum to 1, or the uniform distribution when the total
/// is non-positive (a degenerate residual). Mirrors the Python `_normalize`.
pub fn normalize(dist: &[f64]) -> Vec<f64> {
    let total: f64 = dist.iter().sum();
    if total <= 0.0 {
        let u = 1.0 / dist.len() as f64;
        return vec![u; dist.len()];
    }
    dist.iter().map(|v| v / total).collect()
}

/// Draws one token index from a probability vector using a single uniform draw
/// and inverse-CDF selection. The vector is normalized first. This is a
/// deterministic categorical sampler and does not reproduce NumPy's `rng.choice`.
pub fn sample(dist: &[f64], rng: &mut dyn Rng) -> usize {
    let nd = normalize(dist);
    let u = rng.next_uniform();
    let mut cum = 0.0;
    for (i, v) in nd.iter().enumerate() {
        cum += v;
        if u < cum {
            return i;
        }
    }
    nd.len() - 1
}

/// numpy.allclose(row_sums, 1.0) with default rtol=1e-5 and atol=1e-8: a row sum
/// `s` is accepted when `|s - 1| <= atol + rtol`.
const ROW_SUM_TOLERANCE: f64 = 1e-8 + 1e-5;

/// A first-order Markov model: the next-token distribution depends only on the
/// last token, given by a row-stochastic transition matrix. Constructed from an
/// explicit matrix only; the Dirichlet-based random constructors are out of scope.
pub struct MarkovModel {
    transition: Vec<Vec<f64>>,
    vocab_size: usize,
    start_token: usize,
}

impl MarkovModel {
    /// Validates that `transition` is square and row-stochastic and returns a
    /// model whose empty-prefix distribution is row `start_token`.
    pub fn new(transition: Vec<Vec<f64>>, start_token: usize) -> Result<Self, String> {
        let n = transition.len();
        if n == 0 {
            return Err("transition must be a square matrix".to_string());
        }
        for row in &transition {
            if row.len() != n {
                return Err("transition must be a square matrix".to_string());
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > ROW_SUM_TOLERANCE {
                return Err("transition rows must sum to 1".to_string());
            }
        }
        Ok(Self {
            transition,
            vocab_size: n,
            start_token,
        })
    }
}

impl Model for MarkovModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn next_dist(&self, tokens: &[usize]) -> Vec<f64> {
        let last = tokens.last().copied().unwrap_or(self.start_token);
        self.transition[last].clone()
    }
}
