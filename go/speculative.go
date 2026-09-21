package specdecode

import (
	"errors"
	"math"
)

// Model is a next-token distribution oracle. NextDist returns a probability
// vector over the vocabulary for the given prefix and must be a pure function of
// its argument — the acceptance theorem assumes the target's distribution at a
// position does not depend on how that position was reached.
type Model interface {
	VocabSize() int
	NextDist(tokens []int) []float64
}

// MarkovModel is a first-order Markov model: the next-token distribution depends
// only on the last token, given by a row-stochastic transition matrix. Unlike
// the Python reference it is constructed from an explicit matrix only; the
// Dirichlet-based random constructors are out of scope for the port.
type MarkovModel struct {
	transition [][]float64
	vocabSize  int
	startToken int
}

// rowSumTolerance mirrors numpy.allclose(row_sums, 1.0) with its default
// rtol=1e-5 and atol=1e-8: a row sum s is accepted when |s - 1| <= atol + rtol.
const rowSumTolerance = 1e-8 + 1e-5

// NewMarkovModel validates that transition is square and row-stochastic and
// returns a model whose empty-prefix distribution is row startToken.
func NewMarkovModel(transition [][]float64, startToken int) (*MarkovModel, error) {
	n := len(transition)
	if n == 0 {
		return nil, errors.New("transition must be a square matrix")
	}
	rows := make([][]float64, n)
	for i, row := range transition {
		if len(row) != n {
			return nil, errors.New("transition must be a square matrix")
		}
		sum := 0.0
		for _, v := range row {
			sum += v
		}
		if math.Abs(sum-1.0) > rowSumTolerance {
			return nil, errors.New("transition rows must sum to 1")
		}
		cp := make([]float64, n)
		copy(cp, row)
		rows[i] = cp
	}
	return &MarkovModel{transition: rows, vocabSize: n, startToken: startToken}, nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (m *MarkovModel) VocabSize() int { return m.vocabSize }

// NextDist returns a copy of the transition row for the last token, or the
// start-token row when the prefix is empty.
func (m *MarkovModel) NextDist(tokens []int) []float64 {
	last := m.startToken
	if len(tokens) > 0 {
		last = tokens[len(tokens)-1]
	}
	out := make([]float64, m.vocabSize)
	copy(out, m.transition[last])
	return out
}

// Residual is the adjusted distribution to sample from after a rejection: the
// normalized positive part of p - q. This is the term that makes the scheme
// exact.
func Residual(p, q []float64) []float64 {
	diff := make([]float64, len(p))
	for i := range p {
		d := p[i] - q[i]
		if d > 0.0 {
			diff[i] = d
		}
	}
	return Normalize(diff)
}

// AcceptProbability is min(1, pt/qt), the probability of accepting a proposed
// token with target probability pt and draft probability qt. A non-positive qt
// (a token the draft never proposes) is accepted with probability 1.
func AcceptProbability(pt, qt float64) float64 {
	if qt <= 0.0 {
		return 1.0
	}
	return math.Min(1.0, pt/qt)
}

// Trace records the token output and pass/acceptance counters for one generate
// call — the raw material for measuring speedup honestly.
type Trace struct {
	Tokens       []int
	Steps        int
	TargetPasses int
	DraftPasses  int
	Proposed     int
	Accepted     int
}

// AcceptanceRate is accepted/proposed, or 0 when nothing was proposed.
func (t *Trace) AcceptanceRate() float64 {
	if t.Proposed == 0 {
		return 0.0
	}
	return float64(t.Accepted) / float64(t.Proposed)
}

// TokensPerStep is len(tokens)/steps, or 0 when there were no steps.
func (t *Trace) TokensPerStep() float64 {
	if t.Steps == 0 {
		return 0.0
	}
	return float64(len(t.Tokens)) / float64(t.Steps)
}

// SpeculativeStep runs one speculative step from prefix and returns the tokens it
// emits (between 1 and k+1). If trace is non-nil its counters and token list are
// updated. The draft proposes k tokens, the target scores all k positions plus a
// look-ahead as one pass, and proposals are accepted or rejected in order; the
// first rejection emits a residual sample and stops the step, while a fully
// accepted step appends a bonus token from the look-ahead distribution.
func SpeculativeStep(target, draft Model, prefix []int, k int, rng RNG, trace *Trace) []int {
	proposals := make([]int, 0, k)
	qDists := make([][]float64, 0, k)
	ctx := append([]int(nil), prefix...)
	for i := 0; i < k; i++ {
		q := draft.NextDist(ctx)
		x := Sample(q, rng)
		proposals = append(proposals, x)
		qDists = append(qDists, q)
		ctx = append(ctx, x)
	}
	if trace != nil {
		trace.DraftPasses += k
		trace.Proposed += k
	}

	pDists := make([][]float64, 0, k)
	ctx = append([]int(nil), prefix...)
	for i := 0; i < k; i++ {
		pDists = append(pDists, target.NextDist(ctx))
		ctx = append(ctx, proposals[i])
	}
	pLookahead := target.NextDist(ctx)
	if trace != nil {
		trace.TargetPasses++
		trace.Steps++
	}

	emitted := make([]int, 0, k+1)
	for i := 0; i < k; i++ {
		x := proposals[i]
		a := AcceptProbability(pDists[i][x], qDists[i][x])
		if rng.NextUniform() < a {
			emitted = append(emitted, x)
			if trace != nil {
				trace.Accepted++
			}
		} else {
			emitted = append(emitted, Sample(Residual(pDists[i], qDists[i]), rng))
			if trace != nil {
				trace.Tokens = append(trace.Tokens, emitted...)
			}
			return emitted
		}
	}

	emitted = append(emitted, Sample(pLookahead, rng))
	if trace != nil {
		trace.Tokens = append(trace.Tokens, emitted...)
	}
	return emitted
}

// GenerateSpeculative generates nTokens via speculative decoding and returns the
// full trace. It returns an error when k < 1. A step can overshoot the requested
// count, so the token list is trimmed to exactly nTokens.
func GenerateSpeculative(target, draft Model, prefix []int, nTokens, k int, rng RNG) (*Trace, error) {
	if k < 1 {
		return nil, errors.New("k must be >= 1")
	}
	trace := &Trace{}
	ctx := append([]int(nil), prefix...)
	for len(trace.Tokens) < nTokens {
		emitted := SpeculativeStep(target, draft, ctx, k, rng, trace)
		ctx = append(ctx, emitted...)
	}
	if len(trace.Tokens) > nTokens {
		trace.Tokens = trace.Tokens[:nTokens]
	}
	return trace, nil
}

// GenerateTargetOnly is the baseline: sample nTokens from the target one at a
// time, costing one target pass per token.
func GenerateTargetOnly(target Model, prefix []int, nTokens int, rng RNG) *Trace {
	trace := &Trace{}
	ctx := append([]int(nil), prefix...)
	for i := 0; i < nTokens; i++ {
		p := target.NextDist(ctx)
		x := Sample(p, rng)
		trace.Tokens = append(trace.Tokens, x)
		trace.TargetPasses++
		trace.Steps++
		ctx = append(ctx, x)
	}
	return trace
}

// ExpectedSpeedup is the wall-clock speedup implied by an observed tokensPerStep:
// tokensPerStep / (1 + k*draftCostRatio), where draftCostRatio is the ratio of a
// draft pass cost to a target pass cost.
func ExpectedSpeedup(tokensPerStep float64, k int, draftCostRatio float64) float64 {
	return tokensPerStep / (1.0 + float64(k)*draftCostRatio)
}
