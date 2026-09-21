mod common;

use common::{LcgRng, ScriptRng};
use spec_decode::{
    accept_probability, expected_speedup, generate_speculative, generate_target_only, residual,
    speculative_step, MarkovModel, Trace,
};

fn target_model() -> MarkovModel {
    MarkovModel::new(
        vec![
            vec![0.2, 0.5, 0.3],
            vec![0.6, 0.3, 0.1],
            vec![0.1, 0.2, 0.7],
        ],
        0,
    )
    .unwrap()
}

fn one_hot_draft() -> MarkovModel {
    MarkovModel::new(
        vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ],
        0,
    )
    .unwrap()
}

#[test]
fn accept_probability_target_prefers_token() {
    assert_eq!(accept_probability(0.6, 0.3), 1.0);
}

#[test]
fn accept_probability_partial_when_draft_overproposes() {
    assert!((accept_probability(0.2, 0.8) - 0.25).abs() < 1e-15);
}

#[test]
fn accept_probability_zero_draft_accepts() {
    assert_eq!(accept_probability(0.5, 0.0), 1.0);
}

#[test]
fn accept_probability_equal_is_one() {
    assert_eq!(accept_probability(0.4, 0.4), 1.0);
}

#[test]
fn residual_is_positive_part_normalized() {
    let r = residual(&[0.5, 0.3, 0.2], &[0.1, 0.6, 0.3]);
    assert!((r[0] - 1.0).abs() < 1e-15 && r[1] == 0.0 && r[2] == 0.0);
}

#[test]
fn residual_normalizes_to_one() {
    let r = residual(&[0.4, 0.4, 0.2], &[0.1, 0.1, 0.1]);
    assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-15);
}

#[test]
fn residual_fully_dominated_is_uniform() {
    let r = residual(&[0.1, 0.2], &[0.9, 0.8]);
    assert_eq!(r, vec![0.5, 0.5]);
}

#[test]
fn step_forced_rejection_emits_residual_and_stops() {
    let target = target_model();
    let draft = one_hot_draft();
    let mut rng = ScriptRng::new(vec![0.0, 0.0, 0.1, 0.5, 0.1]);
    let mut trace = Trace::default();
    let emitted = speculative_step(&target, &draft, &[0], 2, &mut rng, Some(&mut trace));
    assert_eq!(emitted, vec![1, 0]);
    assert_eq!(trace.accepted, 1);
    assert_eq!(trace.proposed, 2);
    assert_eq!(trace.draft_passes, 2);
    assert_eq!(trace.target_passes, 1);
    assert_eq!(trace.steps, 1);
    assert_eq!(trace.tokens, vec![1, 0]);
}

#[test]
fn step_full_accept_appends_bonus_token() {
    let target = target_model();
    let draft = one_hot_draft();
    let mut rng = ScriptRng::new(vec![0.0, 0.0, 0.1, 0.05, 0.5]);
    let mut trace = Trace::default();
    let emitted = speculative_step(&target, &draft, &[0], 2, &mut rng, Some(&mut trace));
    assert_eq!(emitted, vec![1, 2, 2]);
    assert_eq!(trace.accepted, 2);
}

#[test]
fn step_emits_between_one_and_k_plus_one() {
    let target = target_model();
    let draft = one_hot_draft();
    let mut rng = LcgRng::new(42);
    for _ in 0..200 {
        let emitted = speculative_step(&target, &draft, &[0], 4, &mut rng, None);
        assert!(!emitted.is_empty() && emitted.len() <= 5);
    }
}

#[test]
fn step_perfect_draft_always_emits_k_plus_one() {
    let m = target_model();
    let mut rng = ScriptRng::new(vec![0.5]);
    for _ in 0..100 {
        let emitted = speculative_step(&m, &m, &[0], 3, &mut rng, None);
        assert_eq!(emitted.len(), 4);
    }
}

#[test]
fn generate_speculative_exact_token_count() {
    let m = target_model();
    let mut rng = ScriptRng::new(vec![0.5]);
    let trace = generate_speculative(&m, &m, &[0], 37, 4, &mut rng).unwrap();
    assert_eq!(trace.tokens.len(), 37);
}

#[test]
fn generate_speculative_rejects_k_below_one() {
    let m = target_model();
    let mut rng = ScriptRng::new(vec![0.5]);
    assert!(generate_speculative(&m, &m, &[0], 5, 0, &mut rng).is_err());
}

#[test]
fn generate_speculative_uses_fewer_target_passes_than_tokens() {
    let m = target_model();
    let mut rng = ScriptRng::new(vec![0.5]);
    let trace = generate_speculative(&m, &m, &[0], 1000, 4, &mut rng).unwrap();
    assert!(trace.target_passes < trace.tokens.len());
}

#[test]
fn generate_target_only_costs_one_pass_per_token() {
    let m = target_model();
    let mut rng = LcgRng::new(7);
    let trace = generate_target_only(&m, &[0], 50, &mut rng);
    assert_eq!(trace.target_passes, 50);
    assert_eq!(trace.tokens.len(), 50);
    assert_eq!(trace.steps, 50);
}

#[test]
fn expected_speedup_scales_with_tokens_per_step() {
    assert_eq!(expected_speedup(3.0, 4, 0.0), 3.0);
}

#[test]
fn expected_speedup_draft_cost_reduces_speedup() {
    assert!(expected_speedup(3.0, 4, 0.05) > expected_speedup(3.0, 4, 0.5));
}

#[test]
fn trace_empty_rates_are_zero() {
    let tr = Trace::default();
    assert_eq!(tr.acceptance_rate(), 0.0);
    assert_eq!(tr.tokens_per_step(), 0.0);
}

#[test]
fn trace_computed_rates() {
    let tr = Trace {
        tokens: vec![1, 2, 3, 4],
        steps: 2,
        proposed: 8,
        accepted: 6,
        ..Trace::default()
    };
    assert!((tr.acceptance_rate() - 0.75).abs() < 1e-15);
    assert!((tr.tokens_per_step() - 2.0).abs() < 1e-15);
}
