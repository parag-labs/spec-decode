package com.specdecode;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class ProofTest {
    private static double maxAbsDiff(double[] a, double[] b) {
        double worst = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.abs(a[i] - b[i]);
            if (d > worst) {
                worst = d;
            }
        }
        return worst;
    }

    private static double[] randomDist(LcgRng rng, int n) {
        double[] raw = new double[n];
        for (int i = 0; i < n; i++) {
            raw[i] = rng.nextUniform();
        }
        return Distributions.normalize(raw);
    }

    private static MarkovModel targetModel() {
        return new MarkovModel(new double[][] {
                {0.2, 0.5, 0.3},
                {0.6, 0.3, 0.1},
                {0.1, 0.2, 0.7},
        }, 0);
    }

    private static MarkovModel differentDraft() {
        return new MarkovModel(new double[][] {
                {0.3, 0.4, 0.3},
                {0.2, 0.5, 0.3},
                {0.5, 0.25, 0.25},
        }, 0);
    }

    @Test
    void inducedEqualsTargetForArbitraryDistributions() {
        LcgRng rng = new LcgRng(0);
        double worst = 0.0;
        for (int trial = 0; trial < 5000; trial++) {
            int n = 2 + (int) (rng.nextUniform() * 10.0);
            double[] p = randomDist(rng, n);
            double[] q = randomDist(rng, n);
            worst = Math.max(worst, maxAbsDiff(Proof.inducedFromDists(p, q), p));
        }
        assertTrue(worst < 1e-12, "max deviation " + worst + " exceeds tolerance");
    }

    @Test
    void inducedIsAValidDistribution() {
        LcgRng rng = new LcgRng(1);
        for (int trial = 0; trial < 500; trial++) {
            int n = 2 + (int) (rng.nextUniform() * 8.0);
            double[] p = randomDist(rng, n);
            double[] q = randomDist(rng, n);
            double[] induced = Proof.inducedFromDists(p, q);
            double sum = 0.0;
            for (double v : induced) {
                assertTrue(v >= -1e-15);
                sum += v;
            }
            assertTrue(Math.abs(sum - 1.0) < 1e-12);
        }
    }

    @Test
    void inducedIdenticalDraftStaysExact() {
        double[] p = Distributions.normalize(new double[] {0.4, 0.1, 0.25, 0.25});
        double[] induced = Proof.inducedFromDists(p, p.clone());
        assertTrue(maxAbsDiff(induced, p) < 1e-15);
    }

    @Test
    void inducedDegenerateDraftStillExact() {
        double[] p = {0.4, 0.3, 0.2, 0.1};
        double[] q = {1.0, 0.0, 0.0, 0.0};
        double[] induced = Proof.inducedFromDists(p, q);
        assertTrue(maxAbsDiff(induced, p) < 1e-15);
    }

    @Test
    void maxTotalVariationIsZeroAcrossContexts() {
        List<int[]> prefixes = List.of(
                new int[] {}, new int[] {0}, new int[] {1}, new int[] {2}, new int[] {2, 1});
        double tv = Proof.maxTotalVariation(targetModel(), differentDraft(), prefixes);
        assertTrue(tv < 1e-12, "max total variation " + tv + " should be ~0");
    }

    @Test
    void inducedNextTokenMatchesTargetNextDist() {
        MarkovModel target = targetModel();
        MarkovModel draft = differentDraft();
        for (int[] prefix : List.of(new int[] {}, new int[] {0}, new int[] {1}, new int[] {2})) {
            double[] induced = Proof.inducedNextTokenDistribution(target, draft, prefix);
            assertTrue(maxAbsDiff(induced, target.nextDist(prefix)) < 1e-12);
        }
    }
}
