package specdecode

// scriptRNG replays a fixed list of uniforms, cycling when exhausted. It drives
// the sampler deterministically so hand-computed accept/reject scenarios can be
// asserted exactly.
type scriptRNG struct {
	vals []float64
	i    int
}

func (r *scriptRNG) NextUniform() float64 {
	v := r.vals[r.i%len(r.vals)]
	r.i++
	return v
}

// lcgRNG is a small deterministic uniform source for "many random trials" tests.
// It need not match any other language's RNG; the invariants it checks hold for
// any input stream.
type lcgRNG struct{ state uint64 }

func newLCG(seed uint64) *lcgRNG { return &lcgRNG{state: seed} }

func (r *lcgRNG) NextUniform() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>11) * (1.0 / 9007199254740992.0)
}

func vecEqualInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
