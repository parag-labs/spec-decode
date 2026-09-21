package com.specdecode;

import java.util.List;

/**
 * The exactness proof, expressed constructively. Speculative decoding is
 * distribution-preserving: the token actually emitted for a given context is
 * distributed exactly according to the target model, regardless of the draft.
 * These helpers compute that induced distribution in closed form so tests can
 * assert it matches the target to floating-point tolerance.
 */
public final class Proof {
    private Proof() {
    }

    /**
     * Returns the closed-form distribution of the token emitted by one
     * accept/reject round given target distribution {@code p} and draft
     * distribution {@code q}. Equals {@code p} exactly.
     *
     * @param p the target distribution
     * @param q the draft distribution
     * @return the induced emitted-token distribution
     */
    public static double[] inducedFromDists(double[] p, double[] q) {
        int n = p.length;
        double[] accepted = new double[n];
        double rejectionMass = 0.0;
        for (int i = 0; i < n; i++) {
            double a = Speculative.acceptProbability(p[i], q[i]);
            accepted[i] = q[i] * a;
            rejectionMass += q[i] * (1.0 - a);
        }
        double[] r = Speculative.residual(p, q);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = accepted[i] + rejectionMass * r[i];
        }
        return out;
    }

    /**
     * Returns the induced next-token distribution for a concrete target/draft
     * pair and token {@code prefix}.
     *
     * @param target the target model
     * @param draft the draft model
     * @param prefix the token prefix
     * @return the induced next-token distribution
     */
    public static double[] inducedNextTokenDistribution(Model target, Model draft, int[] prefix) {
        return inducedFromDists(target.nextDist(prefix), draft.nextDist(prefix));
    }

    /**
     * Returns the maximum total-variation distance between the induced
     * distribution and the target distribution across the supplied
     * {@code prefixes}. For a correct sampler this is zero up to rounding.
     *
     * @param target the target model
     * @param draft the draft model
     * @param prefixes the contexts to check
     * @return the worst-case total-variation distance
     */
    public static double maxTotalVariation(Model target, Model draft, List<int[]> prefixes) {
        double worst = 0.0;
        for (int[] prefix : prefixes) {
            double[] induced = inducedNextTokenDistribution(target, draft, prefix);
            double[] p = target.nextDist(prefix);
            double tv = 0.0;
            for (int i = 0; i < p.length; i++) {
                tv += Math.abs(induced[i] - p[i]);
            }
            tv *= 0.5;
            if (tv > worst) {
                worst = tv;
            }
        }
        return worst;
    }
}
