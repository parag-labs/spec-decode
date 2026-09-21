package com.specdecode;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ModelsTest {
    @Test
    void normalizeZeroVectorIsUniform() {
        double[] out = Distributions.normalize(new double[] {0, 0, 0, 0});
        for (double v : out) {
            assertEquals(0.25, v);
        }
    }

    @Test
    void normalizeScalesToSumOne() {
        double[] out = Distributions.normalize(new double[] {1, 3});
        assertTrue(Math.abs(out[0] - 0.25) < 1e-15);
        assertTrue(Math.abs(out[1] - 0.75) < 1e-15);
    }

    @Test
    void normalizePreservesNormalized() {
        double[] in = {0.2, 0.3, 0.5};
        double[] out = Distributions.normalize(in);
        for (int i = 0; i < in.length; i++) {
            assertTrue(Math.abs(out[i] - in[i]) < 1e-15);
        }
    }

    @Test
    void normalizeNegativeTotalIsUniform() {
        double[] out = Distributions.normalize(new double[] {-1, -1});
        assertEquals(0.5, out[0]);
        assertEquals(0.5, out[1]);
    }

    @Test
    void markovNextDistUsesLastToken() {
        MarkovModel m = new MarkovModel(new double[][] {{0.1, 0.9}, {0.7, 0.3}}, 0);
        double[] got = m.nextDist(new int[] {0});
        assertEquals(0.1, got[0]);
        assertEquals(0.9, got[1]);
        got = m.nextDist(new int[] {1});
        assertEquals(0.7, got[0]);
        assertEquals(0.3, got[1]);
    }

    @Test
    void markovEmptyPrefixUsesStartToken() {
        MarkovModel m = new MarkovModel(new double[][] {{0.1, 0.9}, {0.7, 0.3}}, 1);
        double[] got = m.nextDist(new int[] {});
        assertEquals(0.7, got[0]);
        assertEquals(0.3, got[1]);
    }

    @Test
    void markovVocabSize() {
        MarkovModel m = new MarkovModel(new double[][] {{0.5, 0.5}, {0.5, 0.5}}, 0);
        assertEquals(2, m.vocabSize());
    }

    @Test
    void markovNextDistReturnsCopy() {
        MarkovModel m = new MarkovModel(new double[][] {{0.1, 0.9}, {0.7, 0.3}}, 0);
        double[] got = m.nextDist(new int[] {0});
        got[0] = 42.0;
        double[] again = m.nextDist(new int[] {0});
        assertEquals(0.1, again[0]);
    }

    @Test
    void markovRejectsNonSquare() {
        assertThrows(IllegalArgumentException.class,
                () -> new MarkovModel(new double[][] {{0.5, 0.5}}, 0));
    }

    @Test
    void markovRejectsRowsNotSummingToOne() {
        assertThrows(IllegalArgumentException.class,
                () -> new MarkovModel(new double[][] {{0.5, 0.6}, {0.5, 0.5}}, 0));
    }

    @Test
    void markovAcceptsTinyRowSumDrift() {
        MarkovModel m = new MarkovModel(new double[][] {{0.5 + 1e-6, 0.5}, {0.2, 0.8}}, 0);
        assertEquals(2, m.vocabSize());
    }

    @Test
    void markovRejectsSmallButRealDrift() {
        assertThrows(IllegalArgumentException.class,
                () -> new MarkovModel(new double[][] {{0.5 + 1e-4, 0.5}, {0.2, 0.8}}, 0));
    }

    @Test
    void sampleUsesInverseCdf() {
        double[] dist = {0.2, 0.5, 0.3};
        assertEquals(0, Distributions.sample(dist, new ScriptRng(0.1)));
        assertEquals(1, Distributions.sample(dist, new ScriptRng(0.5)));
        assertEquals(2, Distributions.sample(dist, new ScriptRng(0.95)));
    }

    @Test
    void sampleNeverReturnsZeroProbToken() {
        LcgRng rng = new LcgRng(1);
        double[] dist = {0.0, 1.0, 0.0};
        for (int i = 0; i < 1000; i++) {
            assertEquals(1, Distributions.sample(dist, rng));
        }
    }
}
