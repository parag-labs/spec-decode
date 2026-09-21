package com.specdecode;

/**
 * An autoregressive model: given a token prefix it yields the probability
 * distribution over the next token.
 */
public interface Model {
    /** Returns the number of tokens in the vocabulary. */
    int vocabSize();

    /**
     * Returns the next-token distribution for {@code tokens}. The returned
     * array is a fresh copy the caller may mutate.
     *
     * @param tokens the token prefix (may be empty)
     * @return a distribution over the vocabulary
     */
    double[] nextDist(int[] tokens);
}
