package com.specdecode;

/**
 * A small 64-bit linear congruential generator. Deterministic and dependency
 * free; only used to drive statistical "many trials" tests.
 */
final class LcgRng implements Rng {
    private long state;

    LcgRng(long seed) {
        this.state = seed;
    }

    @Override
    public double nextUniform() {
        state = state * 6364136223846793005L + 1442695040888963407L;
        return (state >>> 11) * (1.0 / 9007199254740992.0);
    }
}
