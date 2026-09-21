package specdecode

import (
	"math"
	"testing"
)

func maxAbsDiff(a, b []float64) float64 {
	worst := 0.0
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if d > worst {
			worst = d
		}
	}
	return worst
}

func TestInducedEqualsTargetForArbitraryDistributions(t *testing.T) {
	// The core theorem: for any target p and draft q, the emitted-token
	// distribution equals p. Checked over many pseudo-random pairs.
	rng := newLCG(0)
	worst := 0.0
	for trial := 0; trial < 5000; trial++ {
		n := 2 + int(rng.NextUniform()*10.0)
		p := make([]float64, n)
		q := make([]float64, n)
		for i := 0; i < n; i++ {
			p[i] = rng.NextUniform()
			q[i] = rng.NextUniform()
		}
		p = Normalize(p)
		q = Normalize(q)
		worst = math.Max(worst, maxAbsDiff(InducedFromDists(p, q), p))
	}
	if worst >= 1e-12 {
		t.Fatalf("max deviation %v exceeds tolerance", worst)
	}
}

func TestInducedIsAValidDistribution(t *testing.T) {
	rng := newLCG(1)
	for trial := 0; trial < 500; trial++ {
		n := 2 + int(rng.NextUniform()*8.0)
		p := make([]float64, n)
		q := make([]float64, n)
		for i := 0; i < n; i++ {
			p[i] = rng.NextUniform()
			q[i] = rng.NextUniform()
		}
		p = Normalize(p)
		q = Normalize(q)
		induced := InducedFromDists(p, q)
		sum := 0.0
		for _, v := range induced {
			if v < -1e-15 {
				t.Fatalf("negative mass %v", v)
			}
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-12 {
			t.Fatalf("sum %v", sum)
		}
	}
}

func TestInducedIdenticalDraftStaysExact(t *testing.T) {
	p := Normalize([]float64{0.4, 0.1, 0.25, 0.25})
	induced := InducedFromDists(p, append([]float64(nil), p...))
	if maxAbsDiff(induced, p) > 1e-15 {
		t.Fatalf("identical draft should reproduce p, got %v", induced)
	}
}

func TestInducedDegenerateDraftStillExact(t *testing.T) {
	p := []float64{0.4, 0.3, 0.2, 0.1}
	q := []float64{1.0, 0.0, 0.0, 0.0}
	induced := InducedFromDists(p, q)
	if maxAbsDiff(induced, p) > 1e-15 {
		t.Fatalf("degenerate draft should be corrected to p, got %v", induced)
	}
}

func TestMaxTotalVariationIsZeroAcrossContexts(t *testing.T) {
	target := targetModel(t)
	// A different-but-valid draft over the same vocabulary.
	draft, err := NewMarkovModel([][]float64{
		{0.3, 0.4, 0.3},
		{0.2, 0.5, 0.3},
		{0.5, 0.25, 0.25},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	prefixes := [][]int{{}, {0}, {1}, {2}, {2, 1}}
	if tv := MaxTotalVariation(target, draft, prefixes); tv >= 1e-12 {
		t.Fatalf("max total variation %v should be ~0", tv)
	}
}

func TestInducedNextTokenMatchesTargetNextDist(t *testing.T) {
	target := targetModel(t)
	draft, _ := NewMarkovModel([][]float64{
		{0.3, 0.4, 0.3},
		{0.2, 0.5, 0.3},
		{0.5, 0.25, 0.25},
	}, 0)
	for _, prefix := range [][]int{{}, {0}, {1}, {2}} {
		induced := InducedNextTokenDistribution(target, draft, prefix)
		if maxAbsDiff(induced, target.NextDist(prefix)) >= 1e-12 {
			t.Fatalf("induced != target for prefix %v", prefix)
		}
	}
}
