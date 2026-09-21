using System;
using System.Collections.Generic;
using SpecDecode;
using Xunit;

namespace SpecDecode.Tests;

public class ProofTests
{
    private static double MaxAbsDiff(double[] a, double[] b)
    {
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs(a[i] - b[i]);
            if (d > worst)
            {
                worst = d;
            }
        }

        return worst;
    }

    private static double[] RandomDist(LcgRng rng, int n)
    {
        double[] raw = new double[n];
        for (int i = 0; i < n; i++)
        {
            raw[i] = rng.NextUniform();
        }

        return Distributions.Normalize(raw);
    }

    private static MarkovModel TargetModel() => new(
        new[]
        {
            new[] { 0.2, 0.5, 0.3 },
            new[] { 0.6, 0.3, 0.1 },
            new[] { 0.1, 0.2, 0.7 },
        },
        0);

    private static MarkovModel DifferentDraft() => new(
        new[]
        {
            new[] { 0.3, 0.4, 0.3 },
            new[] { 0.2, 0.5, 0.3 },
            new[] { 0.5, 0.25, 0.25 },
        },
        0);

    [Fact]
    public void InducedEqualsTargetForArbitraryDistributions()
    {
        LcgRng rng = new(0);
        double worst = 0.0;
        for (int trial = 0; trial < 5000; trial++)
        {
            int n = 2 + (int)(rng.NextUniform() * 10.0);
            double[] p = RandomDist(rng, n);
            double[] q = RandomDist(rng, n);
            worst = Math.Max(worst, MaxAbsDiff(Proof.InducedFromDists(p, q), p));
        }

        Assert.True(worst < 1e-12, $"max deviation {worst} exceeds tolerance");
    }

    [Fact]
    public void InducedIsAValidDistribution()
    {
        LcgRng rng = new(1);
        for (int trial = 0; trial < 500; trial++)
        {
            int n = 2 + (int)(rng.NextUniform() * 8.0);
            double[] p = RandomDist(rng, n);
            double[] q = RandomDist(rng, n);
            double[] induced = Proof.InducedFromDists(p, q);
            double sum = 0.0;
            foreach (double v in induced)
            {
                Assert.True(v >= -1e-15);
                sum += v;
            }

            Assert.True(Math.Abs(sum - 1.0) < 1e-12);
        }
    }

    [Fact]
    public void InducedIdenticalDraftStaysExact()
    {
        double[] p = Distributions.Normalize(new[] { 0.4, 0.1, 0.25, 0.25 });
        double[] induced = Proof.InducedFromDists(p, (double[])p.Clone());
        Assert.True(MaxAbsDiff(induced, p) < 1e-15);
    }

    [Fact]
    public void InducedDegenerateDraftStillExact()
    {
        double[] p = { 0.4, 0.3, 0.2, 0.1 };
        double[] q = { 1.0, 0.0, 0.0, 0.0 };
        double[] induced = Proof.InducedFromDists(p, q);
        Assert.True(MaxAbsDiff(induced, p) < 1e-15);
    }

    [Fact]
    public void MaxTotalVariationIsZeroAcrossContexts()
    {
        IReadOnlyList<IReadOnlyList<int>> prefixes = new IReadOnlyList<int>[]
        {
            Array.Empty<int>(),
            new[] { 0 },
            new[] { 1 },
            new[] { 2 },
            new[] { 2, 1 },
        };
        double tv = Proof.MaxTotalVariation(TargetModel(), DifferentDraft(), prefixes);
        Assert.True(tv < 1e-12, $"max total variation {tv} should be ~0");
    }

    [Fact]
    public void InducedNextTokenMatchesTargetNextDist()
    {
        MarkovModel target = TargetModel();
        MarkovModel draft = DifferentDraft();
        foreach (int[] prefix in new[] { Array.Empty<int>(), new[] { 0 }, new[] { 1 }, new[] { 2 } })
        {
            double[] induced = Proof.InducedNextTokenDistribution(target, draft, prefix);
            Assert.True(MaxAbsDiff(induced, target.NextDist(prefix)) < 1e-12);
        }
    }
}
