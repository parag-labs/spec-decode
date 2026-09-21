//! Pure-logic port of the deterministic core of spec-decode.
//!
//! A cheap draft model proposes tokens; the target verifies them in one pass; an
//! accept/reject rule guarantees the emitted tokens are distributed exactly as if
//! sampled from the target directly. This crate ports the deterministic parts of
//! that scheme — the models (as explicit transition matrices), the sampler logic
//! (parameterised over an injected [`Rng`]), and the closed-form exactness proof.
//! It deliberately excludes NumPy's PCG64 random stream and the Dirichlet-based
//! random model construction.

pub mod models;
pub mod proof;
pub mod speculative;

pub use models::{normalize, sample, MarkovModel, Model, Rng};
pub use proof::{induced_from_dists, induced_next_token_distribution, max_total_variation};
pub use speculative::{
    accept_probability, expected_speedup, generate_speculative, generate_target_only, residual,
    speculative_step, Trace,
};
