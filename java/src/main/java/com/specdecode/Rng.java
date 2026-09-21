package com.specdecode;

/**
 * A source of uniform random doubles in the half-open interval {@code [0, 1)}.
 *
 * <p>The speculative sampler is parameterised over this interface so that its
 * accept/reject logic is fully deterministic and testable.
 */
public interface Rng {
    /** Returns the next uniform sample in {@code [0, 1)}. */
    double nextUniform();
}
