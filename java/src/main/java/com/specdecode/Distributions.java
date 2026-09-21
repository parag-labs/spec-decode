package com.specdecode;

/** Distribution helpers shared by the sampler and the proof. */
public final class Distributions {
    private Distributions() {
    }

    /**
     * Returns a probability distribution proportional to {@code dist}. A vector
     * whose entries sum to zero (or less) normalises to uniform.
     *
     * @param dist unnormalised weights
     * @return a fresh distribution that sums to 1
     */
    public static double[] normalize(double[] dist) {
        int n = dist.length;
        double total = 0.0;
        for (double v : dist) {
            total += v;
        }
        double[] out = new double[n];
        if (total <= 0.0) {
            double u = 1.0 / n;
            for (int i = 0; i < n; i++) {
                out[i] = u;
            }
            return out;
        }
        for (int i = 0; i < n; i++) {
            out[i] = dist[i] / total;
        }
        return out;
    }

    /**
     * Draws an index from {@code dist} using one uniform sample and inverse-CDF
     * lookup. The distribution is normalised first, so an index with zero
     * probability is never returned.
     *
     * @param dist unnormalised weights
     * @param rng the uniform source
     * @return the sampled index
     */
    public static int sample(double[] dist, Rng rng) {
        double[] nd = normalize(dist);
        double u = rng.nextUniform();
        double cum = 0.0;
        for (int i = 0; i < nd.length; i++) {
            cum += nd[i];
            if (u < cum) {
                return i;
            }
        }
        return nd.length - 1;
    }
}
