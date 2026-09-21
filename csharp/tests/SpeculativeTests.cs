using System;
using System.Collections.Generic;
using SpecDecode;
using Xunit;

namespace SpecDecode.Tests;

public class SpeculativeTests
{
    // targetModel and OneHotDraft form a hand-computable pair: the draft always
    // proposes token 1 then token 2 from prefix [0], so scripted-RNG scenarios
    // have a single deterministic outcome.
    private static MarkovModel TargetModel() => new(
        new[]
        {
            new[] { 0.2, 0.5, 0.3 },
            new[] { 0.6, 0.3, 0.1 },
            new[] { 0.1, 0.2, 0.7 },
        },
        0);

    private static MarkovModel OneHotDraft() => new(
        new[]
        {
            new[] { 0.0, 1.0, 0.0 },
            new[] { 0.0, 0.0, 1.0 },
            new[] { 1.0, 0.0, 0.0 },
        },
        0);

    [Fact]
    public void AcceptProbabilityTargetPrefersToken()
    {
        Assert.Equal(1.0, Speculative.AcceptProbability(0.6, 0.3));
    }

    [Fact]
    public void AcceptProbabilityPartialWhenDraftOverproposes()
    {
        Assert.True(Math.Abs(Speculative.AcceptProbability(0.2, 0.8) - 0.25) < 1e-15);
    }

    [Fact]
    public void AcceptProbabilityZeroDraftAccepts()
    {
        Assert.Equal(1.0, Speculative.AcceptProbability(0.5, 0.0));
    }

    [Fact]
    public void AcceptProbabilityEqualIsOne()
    {
        Assert.Equal(1.0, Speculative.AcceptProbability(0.4, 0.4));
    }

    [Fact]
    public void ResidualIsPositivePartNormalized()
    {
        double[] r = Speculative.Residual(new[] { 0.5, 0.3, 0.2 }, new[] { 0.1, 0.6, 0.3 });
        Assert.True(Math.Abs(r[0] - 1.0) < 1e-15);
        Assert.Equal(0.0, r[1]);
        Assert.Equal(0.0, r[2]);
    }

    [Fact]
    public void ResidualNormalizesToOne()
    {
        double[] r = Speculative.Residual(new[] { 0.4, 0.4, 0.2 }, new[] { 0.1, 0.1, 0.1 });
        Assert.True(Math.Abs(r[0] + r[1] + r[2] - 1.0) < 1e-15);
    }

    [Fact]
    public void ResidualFullyDominatedIsUniform()
    {
        double[] r = Speculative.Residual(new[] { 0.1, 0.2 }, new[] { 0.9, 0.8 });
        Assert.Equal(0.5, r[0]);
        Assert.Equal(0.5, r[1]);
    }

    [Fact]
    public void StepForcedRejectionEmitsResidualAndStops()
    {
        Trace trace = new();
        ScriptRng rng = new(0.0, 0.0, 0.1, 0.5, 0.1);
        List<int> emitted = Speculative.SpeculativeStep(TargetModel(), OneHotDraft(), new[] { 0 }, 2, rng, trace);
        Assert.Equal(new[] { 1, 0 }, emitted);
        Assert.Equal(1, trace.Accepted);
        Assert.Equal(2, trace.Proposed);
        Assert.Equal(2, trace.DraftPasses);
        Assert.Equal(1, trace.TargetPasses);
        Assert.Equal(1, trace.Steps);
        Assert.Equal(new[] { 1, 0 }, trace.Tokens);
    }

    [Fact]
    public void StepFullAcceptAppendsBonusToken()
    {
        Trace trace = new();
        ScriptRng rng = new(0.0, 0.0, 0.1, 0.05, 0.5);
        List<int> emitted = Speculative.SpeculativeStep(TargetModel(), OneHotDraft(), new[] { 0 }, 2, rng, trace);
        Assert.Equal(new[] { 1, 2, 2 }, emitted);
        Assert.Equal(2, trace.Accepted);
    }

    [Fact]
    public void StepEmitsBetweenOneAndKPlusOne()
    {
        MarkovModel target = TargetModel();
        MarkovModel draft = OneHotDraft();
        LcgRng rng = new(42);
        for (int i = 0; i < 200; i++)
        {
            List<int> emitted = Speculative.SpeculativeStep(target, draft, new[] { 0 }, 4, rng);
            Assert.InRange(emitted.Count, 1, 5);
        }
    }

    [Fact]
    public void StepPerfectDraftAlwaysEmitsKPlusOne()
    {
        MarkovModel m = TargetModel();
        ScriptRng rng = new(0.5);
        for (int i = 0; i < 100; i++)
        {
            List<int> emitted = Speculative.SpeculativeStep(m, m, new[] { 0 }, 3, rng);
            Assert.Equal(4, emitted.Count);
        }
    }

    [Fact]
    public void GenerateSpeculativeExactTokenCount()
    {
        MarkovModel m = TargetModel();
        Trace trace = Speculative.GenerateSpeculative(m, m, new[] { 0 }, 37, 4, new ScriptRng(0.5));
        Assert.Equal(37, trace.Tokens.Count);
    }

    [Fact]
    public void GenerateSpeculativeRejectsKBelowOne()
    {
        MarkovModel m = TargetModel();
        Assert.Throws<ArgumentException>(() =>
            Speculative.GenerateSpeculative(m, m, new[] { 0 }, 5, 0, new ScriptRng(0.5)));
    }

    [Fact]
    public void GenerateSpeculativeUsesFewerTargetPassesThanTokens()
    {
        MarkovModel m = TargetModel();
        Trace trace = Speculative.GenerateSpeculative(m, m, new[] { 0 }, 1000, 4, new ScriptRng(0.5));
        Assert.True(trace.TargetPasses < trace.Tokens.Count);
    }

    [Fact]
    public void GenerateTargetOnlyCostsOnePassPerToken()
    {
        MarkovModel m = TargetModel();
        Trace trace = Speculative.GenerateTargetOnly(m, new[] { 0 }, 50, new LcgRng(7));
        Assert.Equal(50, trace.TargetPasses);
        Assert.Equal(50, trace.Tokens.Count);
        Assert.Equal(50, trace.Steps);
    }

    [Fact]
    public void ExpectedSpeedupScalesWithTokensPerStep()
    {
        Assert.Equal(3.0, Speculative.ExpectedSpeedup(3.0, 4, 0.0));
    }

    [Fact]
    public void ExpectedSpeedupDraftCostReducesSpeedup()
    {
        double cheap = Speculative.ExpectedSpeedup(3.0, 4, 0.05);
        double pricey = Speculative.ExpectedSpeedup(3.0, 4, 0.5);
        Assert.True(cheap > pricey);
    }

    [Fact]
    public void TraceEmptyRatesAreZero()
    {
        Trace tr = new();
        Assert.Equal(0.0, tr.AcceptanceRate);
        Assert.Equal(0.0, tr.TokensPerStep);
    }

    [Fact]
    public void TraceComputedRates()
    {
        Trace tr = new();
        tr.Tokens.AddRange(new[] { 1, 2, 3, 4 });
        tr.Steps = 2;
        tr.Proposed = 8;
        tr.Accepted = 6;
        Assert.True(Math.Abs(tr.AcceptanceRate - 0.75) < 1e-15);
        Assert.True(Math.Abs(tr.TokensPerStep - 2.0) < 1e-15);
    }
}
