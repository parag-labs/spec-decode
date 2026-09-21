mod common;

use common::{LcgRng, ScriptRng};
use spec_decode::{normalize, sample, MarkovModel, Model};

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-15)
}

#[test]
fn normalize_zero_vector_is_uniform() {
    assert!(close(
        &normalize(&[0.0, 0.0, 0.0, 0.0]),
        &[0.25, 0.25, 0.25, 0.25]
    ));
}

#[test]
fn normalize_scales_to_sum_one() {
    assert!(close(&normalize(&[1.0, 3.0]), &[0.25, 0.75]));
}

#[test]
fn normalize_preserves_normalized() {
    assert!(close(&normalize(&[0.2, 0.3, 0.5]), &[0.2, 0.3, 0.5]));
}

#[test]
fn normalize_negative_total_is_uniform() {
    assert!(close(&normalize(&[-1.0, -1.0]), &[0.5, 0.5]));
}

#[test]
fn markov_next_dist_uses_last_token() {
    let m = MarkovModel::new(vec![vec![0.1, 0.9], vec![0.7, 0.3]], 0).unwrap();
    assert!(close(&m.next_dist(&[0]), &[0.1, 0.9]));
    assert!(close(&m.next_dist(&[1]), &[0.7, 0.3]));
}

#[test]
fn markov_empty_prefix_uses_start_token() {
    let m = MarkovModel::new(vec![vec![0.1, 0.9], vec![0.7, 0.3]], 1).unwrap();
    assert!(close(&m.next_dist(&[]), &[0.7, 0.3]));
}

#[test]
fn markov_vocab_size() {
    let m = MarkovModel::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]], 0).unwrap();
    assert_eq!(m.vocab_size(), 2);
}

#[test]
fn markov_next_dist_returns_fresh_vec() {
    let m = MarkovModel::new(vec![vec![0.1, 0.9], vec![0.7, 0.3]], 0).unwrap();
    let mut d = m.next_dist(&[0]);
    d[0] = 42.0;
    assert!(close(&m.next_dist(&[0]), &[0.1, 0.9]));
}

#[test]
fn markov_rejects_non_square() {
    assert!(MarkovModel::new(vec![vec![0.5, 0.5]], 0).is_err());
}

#[test]
fn markov_rejects_rows_not_summing_to_one() {
    assert!(MarkovModel::new(vec![vec![0.5, 0.6], vec![0.5, 0.5]], 0).is_err());
}

#[test]
fn markov_accepts_tiny_row_sum_drift() {
    assert!(MarkovModel::new(vec![vec![0.5 + 1e-6, 0.5], vec![0.2, 0.8]], 0).is_ok());
}

#[test]
fn markov_rejects_small_but_real_drift() {
    assert!(MarkovModel::new(vec![vec![0.5 + 1e-4, 0.5], vec![0.2, 0.8]], 0).is_err());
}

#[test]
fn sample_uses_inverse_cdf() {
    // dist [0.2, 0.5, 0.3]; cumulative 0.2, 0.7, 1.0.
    assert_eq!(sample(&[0.2, 0.5, 0.3], &mut ScriptRng::new(vec![0.1])), 0);
    assert_eq!(sample(&[0.2, 0.5, 0.3], &mut ScriptRng::new(vec![0.5])), 1);
    assert_eq!(sample(&[0.2, 0.5, 0.3], &mut ScriptRng::new(vec![0.95])), 2);
}

#[test]
fn sample_never_returns_zero_prob_token() {
    let mut rng = LcgRng::new(1);
    for _ in 0..1000 {
        assert_eq!(sample(&[0.0, 1.0, 0.0], &mut rng), 1);
    }
}
