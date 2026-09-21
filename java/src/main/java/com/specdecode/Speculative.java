package com.specdecode;

import java.util.ArrayList;
import java.util.List;

/**
 * The speculative-decoding sampler. Every routine is parameterised over an
 * {@link Rng} so the accept/reject state machine is deterministic.
 */
public final class Speculative {
    private Speculative() {
    }

    /**
     * Returns the normalised positive part of {@code p - q}: the residual
     * distribution sampled when a draft token is rejected.
     *
     * @param p the target distribution
     * @param q the draft distribution
     * @return the residual distribution
     */
    public static double[] residual(double[] p, double[] q) {
        double[] diff = new double[p.length];
        for (int i = 0; i < diff.length; i++) {
            double d = p[i] - q[i];
            diff[i] = d > 0.0 ? d : 0.0;
        }
        return Distributions.normalize(diff);
    }

    /**
     * Returns the probability of accepting a draft token whose target and draft
     * probabilities are {@code pt} and {@code qt}.
     *
     * @param pt the target probability of the token
     * @param qt the draft probability of the token
     * @return the acceptance probability
     */
    public static double acceptProbability(double pt, double qt) {
        if (qt <= 0.0) {
            return 1.0;
        }
        return Math.min(1.0, pt / qt);
    }

    private static int[] toArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    /**
     * Runs one speculative step: propose {@code k} draft tokens, score them
     * against the target, and emit the accepted prefix plus one bonus/residual
     * token. Emits at most {@code k + 1} tokens.
     *
     * @param target the target model
     * @param draft the draft model
     * @param prefix the current context
     * @param k the number of speculative proposals
     * @param rng the uniform source
     * @param trace optional bookkeeping (may be {@code null})
     * @return the emitted tokens
     */
    public static List<Integer> speculativeStep(
            Model target, Model draft, int[] prefix, int k, Rng rng, Trace trace) {
        List<Integer> proposals = new ArrayList<>(k);
        List<double[]> qDists = new ArrayList<>(k);
        List<Integer> ctx = new ArrayList<>();
        for (int t : prefix) {
            ctx.add(t);
        }
        for (int i = 0; i < k; i++) {
            double[] q = draft.nextDist(toArray(ctx));
            int x = Distributions.sample(q, rng);
            proposals.add(x);
            qDists.add(q);
            ctx.add(x);
        }
        if (trace != null) {
            trace.draftPasses += k;
            trace.proposed += k;
        }

        List<double[]> pDists = new ArrayList<>(k);
        ctx = new ArrayList<>();
        for (int t : prefix) {
            ctx.add(t);
        }
        for (int i = 0; i < k; i++) {
            pDists.add(target.nextDist(toArray(ctx)));
            ctx.add(proposals.get(i));
        }
        double[] pLookahead = target.nextDist(toArray(ctx));
        if (trace != null) {
            trace.targetPasses += 1;
            trace.steps += 1;
        }

        List<Integer> emitted = new ArrayList<>(k + 1);
        for (int i = 0; i < k; i++) {
            int x = proposals.get(i);
            double a = acceptProbability(pDists.get(i)[x], qDists.get(i)[x]);
            if (rng.nextUniform() < a) {
                emitted.add(x);
                if (trace != null) {
                    trace.accepted += 1;
                }
            } else {
                emitted.add(Distributions.sample(residual(pDists.get(i), qDists.get(i)), rng));
                if (trace != null) {
                    trace.tokens.addAll(emitted);
                }
                return emitted;
            }
        }
        emitted.add(Distributions.sample(pLookahead, rng));
        if (trace != null) {
            trace.tokens.addAll(emitted);
        }
        return emitted;
    }

    /**
     * Generates at least {@code nTokens} tokens via repeated speculative steps,
     * trimming any overshoot from the final step.
     *
     * @param target the target model
     * @param draft the draft model
     * @param prefix the initial context
     * @param nTokens the number of tokens to generate
     * @param k the number of speculative proposals per step
     * @param rng the uniform source
     * @return the completed trace
     * @throws IllegalArgumentException if {@code k < 1}
     */
    public static Trace generateSpeculative(
            Model target, Model draft, int[] prefix, int nTokens, int k, Rng rng) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1");
        }
        Trace trace = new Trace();
        List<Integer> ctx = new ArrayList<>();
        for (int t : prefix) {
            ctx.add(t);
        }
        while (trace.tokens.size() < nTokens) {
            List<Integer> emitted = speculativeStep(target, draft, toArray(ctx), k, rng, trace);
            ctx.addAll(emitted);
        }
        while (trace.tokens.size() > nTokens) {
            trace.tokens.remove(trace.tokens.size() - 1);
        }
        return trace;
    }

    /**
     * Generates {@code nTokens} tokens by sampling the target model directly,
     * one forward pass per token. Used as the baseline.
     *
     * @param target the target model
     * @param prefix the initial context
     * @param nTokens the number of tokens to generate
     * @param rng the uniform source
     * @return the completed trace
     */
    public static Trace generateTargetOnly(Model target, int[] prefix, int nTokens, Rng rng) {
        Trace trace = new Trace();
        List<Integer> ctx = new ArrayList<>();
        for (int t : prefix) {
            ctx.add(t);
        }
        for (int i = 0; i < nTokens; i++) {
            double[] p = target.nextDist(toArray(ctx));
            int x = Distributions.sample(p, rng);
            trace.tokens.add(x);
            trace.targetPasses += 1;
            trace.steps += 1;
            ctx.add(x);
        }
        return trace;
    }

    /**
     * Returns the idealised expected speedup: emitted tokens per step divided by
     * the relative cost of one step (one target pass plus {@code k} draft passes
     * each costing {@code draftCostRatio}).
     *
     * @param tokensPerStep average tokens emitted per step
     * @param k the number of speculative proposals per step
     * @param draftCostRatio cost of a draft pass relative to a target pass
     * @return the expected speedup factor
     */
    public static double expectedSpeedup(double tokensPerStep, int k, double draftCostRatio) {
        return tokensPerStep / (1.0 + k * draftCostRatio);
    }
}
