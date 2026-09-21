mod common;

use common::LcgRng;
use spec_decode::{
    induced_from_dists, induced_next_token_distribution, max_total_variation, normalize,
    MarkovModel, Model, Rng,
};

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn random_dist(rng: &mut LcgRng, n: usize) -> Vec<f64> {
    let raw: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
    normalize(&raw)
}

#[test]
fn induced_equals_target_for_arbitrary_distributions() {
    // The core theorem: for any target p and draft q, the emitted-token
    // distribution equals p. Checked over many pseudo-random pairs.
    let mut rng = LcgRng::new(0);
    let mut worst = 0.0_f64;
    for _ in 0..5000 {
        let n = 2 + (rng.next_uniform() * 10.0) as usize;
        let p = random_dist(&mut rng, n);
        let q = random_dist(&mut rng, n);
        worst = worst.max(max_abs_diff(&induced_from_dists(&p, &q), &p));
    }
    assert!(worst < 1e-12, "max deviation {worst} exceeds tolerance");
}

#[test]
fn induced_is_a_valid_distribution() {
    let mut rng = LcgRng::new(1);
    for _ in 0..500 {
        let n = 2 + (rng.next_uniform() * 8.0) as usize;
        let p = random_dist(&mut rng, n);
        let q = random_dist(&mut rng, n);
        let induced = induced_from_dists(&p, &q);
        assert!(induced.iter().all(|&v| v >= -1e-15));
        assert!((induced.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn induced_identical_draft_stays_exact() {
    let p = normalize(&[0.4, 0.1, 0.25, 0.25]);
    let induced = induced_from_dists(&p, &p);
    assert!(max_abs_diff(&induced, &p) < 1e-15);
}

#[test]
fn induced_degenerate_draft_still_exact() {
    let p = [0.4, 0.3, 0.2, 0.1];
    let q = [1.0, 0.0, 0.0, 0.0];
    let induced = induced_from_dists(&p, &q);
    assert!(max_abs_diff(&induced, &p) < 1e-15);
}

#[test]
fn max_total_variation_is_zero_across_contexts() {
    let target = MarkovModel::new(
        vec![
            vec![0.2, 0.5, 0.3],
            vec![0.6, 0.3, 0.1],
            vec![0.1, 0.2, 0.7],
        ],
        0,
    )
    .unwrap();
    let draft = MarkovModel::new(
        vec![
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.5, 0.25, 0.25],
        ],
        0,
    )
    .unwrap();
    let prefixes = vec![vec![], vec![0], vec![1], vec![2], vec![2, 1]];
    assert!(max_total_variation(&target, &draft, &prefixes) < 1e-12);
}

#[test]
fn induced_next_token_matches_target_next_dist() {
    let target = MarkovModel::new(
        vec![
            vec![0.2, 0.5, 0.3],
            vec![0.6, 0.3, 0.1],
            vec![0.1, 0.2, 0.7],
        ],
        0,
    )
    .unwrap();
    let draft = MarkovModel::new(
        vec![
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.5, 0.25, 0.25],
        ],
        0,
    )
    .unwrap();
    for prefix in [vec![], vec![0], vec![1], vec![2]] {
        let induced = induced_next_token_distribution(&target, &draft, &prefix);
        assert!(max_abs_diff(&induced, &target.next_dist(&prefix)) < 1e-12);
    }
}
