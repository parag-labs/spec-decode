package com.specdecode;

import java.util.ArrayList;
import java.util.List;

/**
 * Bookkeeping for a speculative-decoding run: the emitted tokens plus the
 * pass/acceptance counters used to measure efficiency.
 */
public final class Trace {
    /** Tokens emitted so far, in order. */
    public final List<Integer> tokens = new ArrayList<>();

    /** Number of speculative steps taken. */
    public int steps;

    /** Number of target-model forward passes. */
    public int targetPasses;

    /** Number of draft-model forward passes. */
    public int draftPasses;

    /** Number of draft tokens proposed. */
    public int proposed;

    /** Number of proposed tokens accepted. */
    public int accepted;

    /** Returns the fraction of proposed tokens that were accepted. */
    public double acceptanceRate() {
        return proposed == 0 ? 0.0 : (double) accepted / proposed;
    }

    /** Returns the average number of emitted tokens per speculative step. */
    public double tokensPerStep() {
        return steps == 0 ? 0.0 : (double) tokens.size() / steps;
    }
}
