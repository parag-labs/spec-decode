package specdecode

import (
	"math"
	"testing"
)

// targetModel and oneHotDraft form a hand-computable pair: the draft always
// proposes token 1 then token 2 from prefix [0], so scripted-RNG scenarios have
// a single deterministic outcome.
func targetModel(t *testing.T) *MarkovModel {
	t.Helper()
	m, err := NewMarkovModel([][]float64{
		{0.2, 0.5, 0.3},
		{0.6, 0.3, 0.1},
		{0.1, 0.2, 0.7},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func oneHotDraft(t *testing.T) *MarkovModel {
	t.Helper()
	m, err := NewMarkovModel([][]float64{
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{1.0, 0.0, 0.0},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestAcceptProbabilityTargetPrefersToken(t *testing.T) {
	if AcceptProbability(0.6, 0.3) != 1.0 {
		t.Fatal("target preferring the token should always accept")
	}
}

func TestAcceptProbabilityPartialWhenDraftOverproposes(t *testing.T) {
	if math.Abs(AcceptProbability(0.2, 0.8)-0.25) > 1e-15 {
		t.Fatalf("got %v", AcceptProbability(0.2, 0.8))
	}
}

func TestAcceptProbabilityZeroDraftAccepts(t *testing.T) {
	if AcceptProbability(0.5, 0.0) != 1.0 {
		t.Fatal("token the draft never proposes cannot be rejected")
	}
}

func TestAcceptProbabilityEqualIsOne(t *testing.T) {
	if AcceptProbability(0.4, 0.4) != 1.0 {
		t.Fatal("equal probabilities accept with probability 1")
	}
}

func TestResidualIsPositivePartNormalized(t *testing.T) {
	r := Residual([]float64{0.5, 0.3, 0.2}, []float64{0.1, 0.6, 0.3})
	if math.Abs(r[0]-1.0) > 1e-15 || r[1] != 0.0 || r[2] != 0.0 {
		t.Fatalf("got %v", r)
	}
}

func TestResidualNormalizesToOne(t *testing.T) {
	r := Residual([]float64{0.4, 0.4, 0.2}, []float64{0.1, 0.1, 0.1})
	sum := r[0] + r[1] + r[2]
	if math.Abs(sum-1.0) > 1e-15 {
		t.Fatalf("sum = %v", sum)
	}
}

func TestResidualFullyDominatedIsUniform(t *testing.T) {
	// p entirely below q: positive part is all zeros -> uniform fallback.
	r := Residual([]float64{0.1, 0.2}, []float64{0.9, 0.8})
	if r[0] != 0.5 || r[1] != 0.5 {
		t.Fatalf("got %v", r)
	}
}

func TestStepForcedRejectionEmitsResidualAndStops(t *testing.T) {
	target := targetModel(t)
	draft := oneHotDraft(t)
	rng := &scriptRNG{vals: []float64{0.0, 0.0, 0.1, 0.5, 0.1}}
	trace := &Trace{}
	emitted := SpeculativeStep(target, draft, []int{0}, 2, rng, trace)
	if !vecEqualInt(emitted, []int{1, 0}) {
		t.Fatalf("expected [1 0], got %v", emitted)
	}
	if trace.Accepted != 1 || trace.Proposed != 2 || trace.DraftPasses != 2 || trace.TargetPasses != 1 || trace.Steps != 1 {
		t.Fatalf("trace mismatch: %+v", trace)
	}
	if !vecEqualInt(trace.Tokens, []int{1, 0}) {
		t.Fatalf("trace tokens %v", trace.Tokens)
	}
}

func TestStepFullAcceptAppendsBonusToken(t *testing.T) {
	target := targetModel(t)
	draft := oneHotDraft(t)
	rng := &scriptRNG{vals: []float64{0.0, 0.0, 0.1, 0.05, 0.5}}
	trace := &Trace{}
	emitted := SpeculativeStep(target, draft, []int{0}, 2, rng, trace)
	if !vecEqualInt(emitted, []int{1, 2, 2}) {
		t.Fatalf("expected [1 2 2], got %v", emitted)
	}
	if trace.Accepted != 2 {
		t.Fatalf("expected 2 accepted, got %d", trace.Accepted)
	}
}

func TestStepEmitsBetweenOneAndKPlusOne(t *testing.T) {
	target := targetModel(t)
	draft := oneHotDraft(t)
	rng := newLCG(42)
	for i := 0; i < 200; i++ {
		emitted := SpeculativeStep(target, draft, []int{0}, 4, rng, nil)
		if len(emitted) < 1 || len(emitted) > 5 {
			t.Fatalf("emitted length %d out of range", len(emitted))
		}
	}
}

func TestStepPerfectDraftAlwaysEmitsKPlusOne(t *testing.T) {
	m := targetModel(t)
	rng := &scriptRNG{vals: []float64{0.5}}
	for i := 0; i < 100; i++ {
		emitted := SpeculativeStep(m, m, []int{0}, 3, rng, nil)
		if len(emitted) != 4 {
			t.Fatalf("perfect draft should emit k+1=4, got %d", len(emitted))
		}
	}
}

func TestGenerateSpeculativeExactTokenCount(t *testing.T) {
	m := targetModel(t)
	rng := &scriptRNG{vals: []float64{0.5}}
	trace, err := GenerateSpeculative(m, m, []int{0}, 37, 4, rng)
	if err != nil {
		t.Fatal(err)
	}
	if len(trace.Tokens) != 37 {
		t.Fatalf("expected 37 tokens, got %d", len(trace.Tokens))
	}
}

func TestGenerateSpeculativeRejectsKBelowOne(t *testing.T) {
	m := targetModel(t)
	rng := &scriptRNG{vals: []float64{0.5}}
	if _, err := GenerateSpeculative(m, m, []int{0}, 5, 0, rng); err == nil {
		t.Fatal("k < 1 should error")
	}
}

func TestGenerateSpeculativeUsesFewerTargetPassesThanTokens(t *testing.T) {
	m := targetModel(t)
	rng := &scriptRNG{vals: []float64{0.5}}
	trace, err := GenerateSpeculative(m, m, []int{0}, 1000, 4, rng)
	if err != nil {
		t.Fatal(err)
	}
	if trace.TargetPasses >= len(trace.Tokens) {
		t.Fatalf("target passes %d not < tokens %d", trace.TargetPasses, len(trace.Tokens))
	}
}

func TestGenerateTargetOnlyCostsOnePassPerToken(t *testing.T) {
	m := targetModel(t)
	rng := newLCG(7)
	trace := GenerateTargetOnly(m, []int{0}, 50, rng)
	if trace.TargetPasses != 50 || len(trace.Tokens) != 50 || trace.Steps != 50 {
		t.Fatalf("trace mismatch: %+v", trace)
	}
}

func TestExpectedSpeedupScalesWithTokensPerStep(t *testing.T) {
	if ExpectedSpeedup(3.0, 4, 0.0) != 3.0 {
		t.Fatalf("got %v", ExpectedSpeedup(3.0, 4, 0.0))
	}
}

func TestExpectedSpeedupDraftCostReducesSpeedup(t *testing.T) {
	cheap := ExpectedSpeedup(3.0, 4, 0.05)
	pricey := ExpectedSpeedup(3.0, 4, 0.5)
	if !(cheap > pricey) {
		t.Fatalf("cheap %v should exceed pricey %v", cheap, pricey)
	}
}

func TestTraceEmptyRatesAreZero(t *testing.T) {
	tr := &Trace{}
	if tr.AcceptanceRate() != 0.0 || tr.TokensPerStep() != 0.0 {
		t.Fatalf("empty trace rates should be zero")
	}
}

func TestTraceComputedRates(t *testing.T) {
	tr := &Trace{Tokens: []int{1, 2, 3, 4}, Steps: 2, Proposed: 8, Accepted: 6}
	if math.Abs(tr.AcceptanceRate()-0.75) > 1e-15 {
		t.Fatalf("acceptance rate %v", tr.AcceptanceRate())
	}
	if math.Abs(tr.TokensPerStep()-2.0) > 1e-15 {
		t.Fatalf("tokens per step %v", tr.TokensPerStep())
	}
}
