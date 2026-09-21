// Package specdecode is a pure-logic port of the deterministic core of
// spec-decode: the speculative-sampling accept/reject rule and the closed-form
// exactness proof that certifies it.
//
// A cheap draft model proposes tokens; the target verifies them in one pass; an
// accept/reject rule guarantees the emitted tokens are distributed exactly as if
// sampled from the target directly. This package ports the deterministic parts of
// that scheme — the models (as explicit transition matrices), the sampler logic
// (parameterised over an injected RNG), and the proof — but not NumPy's PCG64
// random stream or the Dirichlet-based random model construction.
package specdecode

// RNG is the single source of randomness the sampler consumes. NextUniform must
// return a float64 in the half-open interval [0, 1). Abstracting the RNG keeps
// the sampler's accept/reject logic pure and deterministically testable without
// reproducing NumPy's PCG64 byte stream.
type RNG interface {
	NextUniform() float64
}

// Normalize returns dist scaled to sum to 1. If the total is non-positive
// (a degenerate residual whose support the draft fully covered) it falls back to
// the uniform distribution, matching the Python reference's _normalize.
func Normalize(dist []float64) []float64 {
	total := 0.0
	for _, v := range dist {
		total += v
	}
	out := make([]float64, len(dist))
	if total <= 0.0 {
		u := 1.0 / float64(len(dist))
		for i := range out {
			out[i] = u
		}
		return out
	}
	for i, v := range dist {
		out[i] = v / total
	}
	return out
}

// Sample draws one token index from a probability vector using a single uniform
// draw and inverse-CDF selection. The vector is normalized first, mirroring the
// Python reference's sample(dist, rng). This is a deterministic categorical
// sampler; it deliberately does not reproduce NumPy's rng.choice byte stream.
func Sample(dist []float64, rng RNG) int {
	nd := Normalize(dist)
	u := rng.NextUniform()
	cum := 0.0
	for i, v := range nd {
		cum += v
		if u < cum {
			return i
		}
	}
	return len(nd) - 1
}
