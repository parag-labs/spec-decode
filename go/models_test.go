package specdecode

import (
	"math"
	"testing"
)

func TestNormalizeZeroVectorIsUniform(t *testing.T) {
	out := Normalize([]float64{0, 0, 0, 0})
	for _, v := range out {
		if v != 0.25 {
			t.Fatalf("expected uniform 0.25, got %v", out)
		}
	}
}

func TestNormalizeScalesToSumOne(t *testing.T) {
	out := Normalize([]float64{1, 3})
	if math.Abs(out[0]-0.25) > 1e-15 || math.Abs(out[1]-0.75) > 1e-15 {
		t.Fatalf("got %v", out)
	}
}

func TestNormalizePreservesNormalized(t *testing.T) {
	in := []float64{0.2, 0.3, 0.5}
	out := Normalize(in)
	for i := range in {
		if math.Abs(out[i]-in[i]) > 1e-15 {
			t.Fatalf("got %v", out)
		}
	}
}

func TestNormalizeNegativeTotalIsUniform(t *testing.T) {
	out := Normalize([]float64{-1, -1})
	if out[0] != 0.5 || out[1] != 0.5 {
		t.Fatalf("got %v", out)
	}
}

func TestMarkovNextDistUsesLastToken(t *testing.T) {
	m, err := NewMarkovModel([][]float64{{0.1, 0.9}, {0.7, 0.3}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := m.NextDist([]int{0})
	if got[0] != 0.1 || got[1] != 0.9 {
		t.Fatalf("got %v", got)
	}
	got = m.NextDist([]int{1})
	if got[0] != 0.7 || got[1] != 0.3 {
		t.Fatalf("got %v", got)
	}
}

func TestMarkovEmptyPrefixUsesStartToken(t *testing.T) {
	m, _ := NewMarkovModel([][]float64{{0.1, 0.9}, {0.7, 0.3}}, 1)
	got := m.NextDist(nil)
	if got[0] != 0.7 || got[1] != 0.3 {
		t.Fatalf("expected start-token row, got %v", got)
	}
}

func TestMarkovVocabSize(t *testing.T) {
	m, _ := NewMarkovModel([][]float64{{0.5, 0.5}, {0.5, 0.5}}, 0)
	if m.VocabSize() != 2 {
		t.Fatalf("got %d", m.VocabSize())
	}
}

func TestMarkovNextDistReturnsCopy(t *testing.T) {
	m, _ := NewMarkovModel([][]float64{{0.1, 0.9}, {0.7, 0.3}}, 0)
	got := m.NextDist([]int{0})
	got[0] = 42.0
	again := m.NextDist([]int{0})
	if again[0] != 0.1 {
		t.Fatalf("mutation leaked into model: %v", again)
	}
}

func TestMarkovRejectsNonSquare(t *testing.T) {
	if _, err := NewMarkovModel([][]float64{{0.5, 0.5}}, 0); err == nil {
		t.Fatal("expected error for non-square matrix")
	}
}

func TestMarkovRejectsRowsNotSummingToOne(t *testing.T) {
	if _, err := NewMarkovModel([][]float64{{0.5, 0.6}, {0.5, 0.5}}, 0); err == nil {
		t.Fatal("expected error for bad row sum")
	}
}

func TestMarkovAcceptsTinyRowSumDrift(t *testing.T) {
	if _, err := NewMarkovModel([][]float64{{0.5 + 1e-6, 0.5}, {0.2, 0.8}}, 0); err != nil {
		t.Fatalf("row sum within tolerance should be accepted: %v", err)
	}
}

func TestMarkovRejectsSmallButRealDrift(t *testing.T) {
	if _, err := NewMarkovModel([][]float64{{0.5 + 1e-4, 0.5}, {0.2, 0.8}}, 0); err == nil {
		t.Fatal("row sum drift of 1e-4 should be rejected")
	}
}

func TestSampleUsesInverseCdf(t *testing.T) {
	// dist [0.2,0.5,0.3]; cumulative 0.2,0.7,1.0.
	rng := &scriptRNG{vals: []float64{0.1}}
	if got := Sample([]float64{0.2, 0.5, 0.3}, rng); got != 0 {
		t.Fatalf("u=0.1 -> token 0, got %d", got)
	}
	rng = &scriptRNG{vals: []float64{0.5}}
	if got := Sample([]float64{0.2, 0.5, 0.3}, rng); got != 1 {
		t.Fatalf("u=0.5 -> token 1, got %d", got)
	}
	rng = &scriptRNG{vals: []float64{0.95}}
	if got := Sample([]float64{0.2, 0.5, 0.3}, rng); got != 2 {
		t.Fatalf("u=0.95 -> token 2, got %d", got)
	}
}

func TestSampleNeverReturnsZeroProbToken(t *testing.T) {
	rng := newLCG(1)
	for i := 0; i < 1000; i++ {
		got := Sample([]float64{0.0, 1.0, 0.0}, rng)
		if got != 1 {
			t.Fatalf("only token 1 has mass, got %d", got)
		}
	}
}
