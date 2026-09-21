package com.specdecode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class SpeculativeTest {
    // targetModel and oneHotDraft form a hand-computable pair: the draft always
    // proposes token 1 then token 2 from prefix [0], so scripted-RNG scenarios
    // have a single deterministic outcome.
    private static MarkovModel targetModel() {
        return new MarkovModel(new double[][] {
                {0.2, 0.5, 0.3},
                {0.6, 0.3, 0.1},
                {0.1, 0.2, 0.7},
        }, 0);
    }

    private static MarkovModel oneHotDraft() {
        return new MarkovModel(new double[][] {
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0},
                {1.0, 0.0, 0.0},
        }, 0);
    }

    @Test
    void acceptProbabilityTargetPrefersToken() {
        assertEquals(1.0, Speculative.acceptProbability(0.6, 0.3));
    }

    @Test
    void acceptProbabilityPartialWhenDraftOverproposes() {
        assertTrue(Math.abs(Speculative.acceptProbability(0.2, 0.8) - 0.25) < 1e-15);
    }

    @Test
    void acceptProbabilityZeroDraftAccepts() {
        assertEquals(1.0, Speculative.acceptProbability(0.5, 0.0));
    }

    @Test
    void acceptProbabilityEqualIsOne() {
        assertEquals(1.0, Speculative.acceptProbability(0.4, 0.4));
    }

    @Test
    void residualIsPositivePartNormalized() {
        double[] r = Speculative.residual(new double[] {0.5, 0.3, 0.2}, new double[] {0.1, 0.6, 0.3});
        assertTrue(Math.abs(r[0] - 1.0) < 1e-15);
        assertEquals(0.0, r[1]);
        assertEquals(0.0, r[2]);
    }

    @Test
    void residualNormalizesToOne() {
        double[] r = Speculative.residual(new double[] {0.4, 0.4, 0.2}, new double[] {0.1, 0.1, 0.1});
        assertTrue(Math.abs(r[0] + r[1] + r[2] - 1.0) < 1e-15);
    }

    @Test
    void residualFullyDominatedIsUniform() {
        double[] r = Speculative.residual(new double[] {0.1, 0.2}, new double[] {0.9, 0.8});
        assertEquals(0.5, r[0]);
        assertEquals(0.5, r[1]);
    }

    @Test
    void stepForcedRejectionEmitsResidualAndStops() {
        Trace trace = new Trace();
        ScriptRng rng = new ScriptRng(0.0, 0.0, 0.1, 0.5, 0.1);
        List<Integer> emitted = Speculative.speculativeStep(targetModel(), oneHotDraft(), new int[] {0}, 2, rng, trace);
        assertEquals(List.of(1, 0), emitted);
        assertEquals(1, trace.accepted);
        assertEquals(2, trace.proposed);
        assertEquals(2, trace.draftPasses);
        assertEquals(1, trace.targetPasses);
        assertEquals(1, trace.steps);
        assertEquals(List.of(1, 0), trace.tokens);
    }

    @Test
    void stepFullAcceptAppendsBonusToken() {
        Trace trace = new Trace();
        ScriptRng rng = new ScriptRng(0.0, 0.0, 0.1, 0.05, 0.5);
        List<Integer> emitted = Speculative.speculativeStep(targetModel(), oneHotDraft(), new int[] {0}, 2, rng, trace);
        assertEquals(List.of(1, 2, 2), emitted);
        assertEquals(2, trace.accepted);
    }

    @Test
    void stepEmitsBetweenOneAndKPlusOne() {
        MarkovModel target = targetModel();
        MarkovModel draft = oneHotDraft();
        LcgRng rng = new LcgRng(42);
        for (int i = 0; i < 200; i++) {
            List<Integer> emitted = Speculative.speculativeStep(target, draft, new int[] {0}, 4, rng, null);
            assertTrue(emitted.size() >= 1 && emitted.size() <= 5);
        }
    }

    @Test
    void stepPerfectDraftAlwaysEmitsKPlusOne() {
        MarkovModel m = targetModel();
        ScriptRng rng = new ScriptRng(0.5);
        for (int i = 0; i < 100; i++) {
            List<Integer> emitted = Speculative.speculativeStep(m, m, new int[] {0}, 3, rng, null);
            assertEquals(4, emitted.size());
        }
    }

    @Test
    void generateSpeculativeExactTokenCount() {
        MarkovModel m = targetModel();
        Trace trace = Speculative.generateSpeculative(m, m, new int[] {0}, 37, 4, new ScriptRng(0.5));
        assertEquals(37, trace.tokens.size());
    }

    @Test
    void generateSpeculativeRejectsKBelowOne() {
        MarkovModel m = targetModel();
        assertThrows(IllegalArgumentException.class,
                () -> Speculative.generateSpeculative(m, m, new int[] {0}, 5, 0, new ScriptRng(0.5)));
    }

    @Test
    void generateSpeculativeUsesFewerTargetPassesThanTokens() {
        MarkovModel m = targetModel();
        Trace trace = Speculative.generateSpeculative(m, m, new int[] {0}, 1000, 4, new ScriptRng(0.5));
        assertTrue(trace.targetPasses < trace.tokens.size());
    }

    @Test
    void generateTargetOnlyCostsOnePassPerToken() {
        MarkovModel m = targetModel();
        Trace trace = Speculative.generateTargetOnly(m, new int[] {0}, 50, new LcgRng(7));
        assertEquals(50, trace.targetPasses);
        assertEquals(50, trace.tokens.size());
        assertEquals(50, trace.steps);
    }

    @Test
    void expectedSpeedupScalesWithTokensPerStep() {
        assertEquals(3.0, Speculative.expectedSpeedup(3.0, 4, 0.0));
    }

    @Test
    void expectedSpeedupDraftCostReducesSpeedup() {
        double cheap = Speculative.expectedSpeedup(3.0, 4, 0.05);
        double pricey = Speculative.expectedSpeedup(3.0, 4, 0.5);
        assertTrue(cheap > pricey);
    }

    @Test
    void traceEmptyRatesAreZero() {
        Trace tr = new Trace();
        assertEquals(0.0, tr.acceptanceRate());
        assertEquals(0.0, tr.tokensPerStep());
    }

    @Test
    void traceComputedRates() {
        Trace tr = new Trace();
        tr.tokens.addAll(List.of(1, 2, 3, 4));
        tr.steps = 2;
        tr.proposed = 8;
        tr.accepted = 6;
        assertTrue(Math.abs(tr.acceptanceRate() - 0.75) < 1e-15);
        assertTrue(Math.abs(tr.tokensPerStep() - 2.0) < 1e-15);
    }
}
