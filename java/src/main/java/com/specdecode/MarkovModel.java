package com.specdecode;

/**
 * A first-order Markov model backed by an explicit row-stochastic transition
 * matrix. This is the deterministic, dependency-free core used by the exactness
 * proof and the sampler tests.
 */
public final class MarkovModel implements Model {
    // numpy.allclose default tolerance: atol + rtol against a target of 1.0.
    private static final double ROW_SUM_TOLERANCE = 1e-8 + 1e-5;

    private final double[][] transition;
    private final int startToken;

    /**
     * Builds a model from a square transition matrix whose rows sum to 1.
     *
     * @param transition row-stochastic transition matrix
     * @param startToken token assumed to precede an empty prefix
     * @throws IllegalArgumentException if the matrix is not square or a row
     *     does not sum to 1
     */
    public MarkovModel(double[][] transition, int startToken) {
        int n = transition.length;
        if (n == 0) {
            throw new IllegalArgumentException("transition must be a non-empty square matrix");
        }
        this.transition = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] row = transition[i];
            if (row.length != n) {
                throw new IllegalArgumentException("transition must be a square matrix");
            }
            double sum = 0.0;
            for (double v : row) {
                sum += v;
            }
            if (Math.abs(sum - 1.0) > ROW_SUM_TOLERANCE) {
                throw new IllegalArgumentException("transition row " + i + " must sum to 1, got " + sum);
            }
            this.transition[i] = row.clone();
        }
        this.startToken = startToken;
    }

    @Override
    public int vocabSize() {
        return transition.length;
    }

    @Override
    public double[] nextDist(int[] tokens) {
        int last = tokens.length > 0 ? tokens[tokens.length - 1] : startToken;
        return transition[last].clone();
    }
}
