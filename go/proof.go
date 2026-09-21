package specdecode

import "math"

// InducedFromDists is the closed-form distribution of the token a speculative
// step emits, given the target distribution p and draft distribution q at that
// position. A token t is emitted either because the draft proposed it and it was
// accepted (mass q[t]*accept[t]) or because some proposal was rejected (total
// mass sum(q*(1-accept))) and the residual sample landed on t. Summed, this
// equals p exactly — the exactness theorem the tests assert.
func InducedFromDists(p, q []float64) []float64 {
	n := len(p)
	accepted := make([]float64, n)
	rejectionMass := 0.0
	for i := 0; i < n; i++ {
		a := AcceptProbability(p[i], q[i])
		accepted[i] = q[i] * a
		rejectionMass += q[i] * (1.0 - a)
	}
	r := Residual(p, q)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = accepted[i] + rejectionMass*r[i]
	}
	return out
}

// InducedNextTokenDistribution is the closed-form distribution of the next token
// a speculative step emits from prefix, which the theorem says equals
// target.NextDist(prefix).
func InducedNextTokenDistribution(target, draft Model, prefix []int) []float64 {
	return InducedFromDists(target.NextDist(prefix), draft.NextDist(prefix))
}

// MaxTotalVariation is the largest total-variation distance between the induced
// emission distribution and the target distribution across a set of prefixes. A
// correct implementation returns ~0 (floating-point noise only).
func MaxTotalVariation(target, draft Model, prefixes [][]int) float64 {
	worst := 0.0
	for _, prefix := range prefixes {
		induced := InducedNextTokenDistribution(target, draft, prefix)
		p := target.NextDist(prefix)
		tv := 0.0
		for i := range p {
			tv += math.Abs(induced[i] - p[i])
		}
		tv *= 0.5
		if tv > worst {
			worst = tv
		}
	}
	return worst
}
