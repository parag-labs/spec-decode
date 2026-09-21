using System;
using System.Collections.Generic;

namespace SpecDecode;

/// <summary>
/// Bookkeeping for a speculative-decoding run: the emitted tokens plus the
/// pass/acceptance counters used to measure efficiency.
/// </summary>
public sealed class Trace
{
    /// <summary>Tokens emitted so far, in order.</summary>
    public List<int> Tokens { get; } = new();

    /// <summary>Number of speculative steps taken.</summary>
    public int Steps { get; set; }

    /// <summary>Number of target-model forward passes.</summary>
    public int TargetPasses { get; set; }

    /// <summary>Number of draft-model forward passes.</summary>
    public int DraftPasses { get; set; }

    /// <summary>Number of draft tokens proposed.</summary>
    public int Proposed { get; set; }

    /// <summary>Number of proposed tokens accepted.</summary>
    public int Accepted { get; set; }

    /// <summary>Fraction of proposed tokens that were accepted.</summary>
    public double AcceptanceRate => Proposed == 0 ? 0.0 : (double)Accepted / Proposed;

    /// <summary>Average number of emitted tokens per speculative step.</summary>
    public double TokensPerStep => Steps == 0 ? 0.0 : (double)Tokens.Count / Steps;
}

/// <summary>
/// The speculative-decoding sampler. Every routine is parameterised over an
/// <see cref="IRng"/> so the accept/reject state machine is deterministic.
/// </summary>
public static class Speculative
{
    /// <summary>
    /// Returns the normalised positive part of <c>p - q</c>: the residual
    /// distribution sampled when a draft token is rejected.
    /// </summary>
    public static double[] Residual(IReadOnlyList<double> p, IReadOnlyList<double> q)
    {
        if (p is null)
        {
            throw new ArgumentNullException(nameof(p));
        }

        if (q is null)
        {
            throw new ArgumentNullException(nameof(q));
        }

        double[] diff = new double[p.Count];
        for (int i = 0; i < diff.Length; i++)
        {
            double d = p[i] - q[i];
            diff[i] = d > 0.0 ? d : 0.0;
        }

        return Distributions.Normalize(diff);
    }

    /// <summary>
    /// Probability of accepting a draft token whose target and draft
    /// probabilities are <paramref name="pt"/> and <paramref name="qt"/>.
    /// </summary>
    public static double AcceptProbability(double pt, double qt)
    {
        if (qt <= 0.0)
        {
            return 1.0;
        }

        return Math.Min(1.0, pt / qt);
    }

    /// <summary>
    /// Runs one speculative step: propose <paramref name="k"/> draft tokens,
    /// score them against the target, and emit the accepted prefix plus one
    /// bonus/residual token. Emits at most <c>k + 1</c> tokens.
    /// </summary>
    public static List<int> SpeculativeStep(
        IModel target,
        IModel draft,
        IReadOnlyList<int> prefix,
        int k,
        IRng rng,
        Trace? trace = null)
    {
        List<int> proposals = new(k);
        List<double[]> qDists = new(k);
        List<int> ctx = new(prefix);
        for (int i = 0; i < k; i++)
        {
            double[] q = draft.NextDist(ctx);
            int x = Distributions.Sample(q, rng);
            proposals.Add(x);
            qDists.Add(q);
            ctx.Add(x);
        }

        if (trace != null)
        {
            trace.DraftPasses += k;
            trace.Proposed += k;
        }

        List<double[]> pDists = new(k);
        ctx = new List<int>(prefix);
        for (int i = 0; i < k; i++)
        {
            pDists.Add(target.NextDist(ctx));
            ctx.Add(proposals[i]);
        }

        double[] pLookahead = target.NextDist(ctx);
        if (trace != null)
        {
            trace.TargetPasses += 1;
            trace.Steps += 1;
        }

        List<int> emitted = new(k + 1);
        for (int i = 0; i < k; i++)
        {
            int x = proposals[i];
            double a = AcceptProbability(pDists[i][x], qDists[i][x]);
            if (rng.NextUniform() < a)
            {
                emitted.Add(x);
                if (trace != null)
                {
                    trace.Accepted += 1;
                }
            }
            else
            {
                emitted.Add(Distributions.Sample(Residual(pDists[i], qDists[i]), rng));
                trace?.Tokens.AddRange(emitted);
                return emitted;
            }
        }

        emitted.Add(Distributions.Sample(pLookahead, rng));
        trace?.Tokens.AddRange(emitted);
        return emitted;
    }

    /// <summary>
    /// Generates at least <paramref name="nTokens"/> tokens via repeated
    /// speculative steps, trimming any overshoot from the final step.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when <paramref name="k"/> &lt; 1.</exception>
    public static Trace GenerateSpeculative(
        IModel target,
        IModel draft,
        IReadOnlyList<int> prefix,
        int nTokens,
        int k,
        IRng rng)
    {
        if (k < 1)
        {
            throw new ArgumentException("k must be >= 1");
        }

        Trace trace = new();
        List<int> ctx = new(prefix);
        while (trace.Tokens.Count < nTokens)
        {
            List<int> emitted = SpeculativeStep(target, draft, ctx, k, rng, trace);
            ctx.AddRange(emitted);
        }

        if (trace.Tokens.Count > nTokens)
        {
            trace.Tokens.RemoveRange(nTokens, trace.Tokens.Count - nTokens);
        }

        return trace;
    }

    /// <summary>
    /// Generates <paramref name="nTokens"/> tokens by sampling the target
    /// model directly, one forward pass per token. Used as the baseline.
    /// </summary>
    public static Trace GenerateTargetOnly(
        IModel target,
        IReadOnlyList<int> prefix,
        int nTokens,
        IRng rng)
    {
        Trace trace = new();
        List<int> ctx = new(prefix);
        for (int i = 0; i < nTokens; i++)
        {
            double[] p = target.NextDist(ctx);
            int x = Distributions.Sample(p, rng);
            trace.Tokens.Add(x);
            trace.TargetPasses += 1;
            trace.Steps += 1;
            ctx.Add(x);
        }

        return trace;
    }

    /// <summary>
    /// Idealised expected speedup: emitted tokens per step divided by the
    /// relative cost of one step (one target pass plus <paramref name="k"/>
    /// draft passes each costing <paramref name="draftCostRatio"/>).
    /// </summary>
    public static double ExpectedSpeedup(double tokensPerStep, int k, double draftCostRatio)
    {
        return tokensPerStep / (1.0 + (k * draftCostRatio));
    }
}
