using System;
using System.Collections.Generic;
using SpecDecode;
using Xunit;

namespace SpecDecode.Tests;

public class ModelsTests
{
    [Fact]
    public void NormalizeZeroVectorIsUniform()
    {
        double[] outp = Distributions.Normalize(new double[] { 0, 0, 0, 0 });
        Assert.All(outp, v => Assert.Equal(0.25, v));
    }

    [Fact]
    public void NormalizeScalesToSumOne()
    {
        double[] outp = Distributions.Normalize(new double[] { 1, 3 });
        Assert.True(Math.Abs(outp[0] - 0.25) < 1e-15);
        Assert.True(Math.Abs(outp[1] - 0.75) < 1e-15);
    }

    [Fact]
    public void NormalizePreservesNormalized()
    {
        double[] input = { 0.2, 0.3, 0.5 };
        double[] outp = Distributions.Normalize(input);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(Math.Abs(outp[i] - input[i]) < 1e-15);
        }
    }

    [Fact]
    public void NormalizeNegativeTotalIsUniform()
    {
        double[] outp = Distributions.Normalize(new double[] { -1, -1 });
        Assert.Equal(0.5, outp[0]);
        Assert.Equal(0.5, outp[1]);
    }

    [Fact]
    public void MarkovNextDistUsesLastToken()
    {
        MarkovModel m = new(new[] { new[] { 0.1, 0.9 }, new[] { 0.7, 0.3 } }, 0);
        double[] got = m.NextDist(new[] { 0 });
        Assert.Equal(0.1, got[0]);
        Assert.Equal(0.9, got[1]);
        got = m.NextDist(new[] { 1 });
        Assert.Equal(0.7, got[0]);
        Assert.Equal(0.3, got[1]);
    }

    [Fact]
    public void MarkovEmptyPrefixUsesStartToken()
    {
        MarkovModel m = new(new[] { new[] { 0.1, 0.9 }, new[] { 0.7, 0.3 } }, 1);
        double[] got = m.NextDist(Array.Empty<int>());
        Assert.Equal(0.7, got[0]);
        Assert.Equal(0.3, got[1]);
    }

    [Fact]
    public void MarkovVocabSize()
    {
        MarkovModel m = new(new[] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } }, 0);
        Assert.Equal(2, m.VocabSize);
    }

    [Fact]
    public void MarkovNextDistReturnsCopy()
    {
        MarkovModel m = new(new[] { new[] { 0.1, 0.9 }, new[] { 0.7, 0.3 } }, 0);
        double[] got = m.NextDist(new[] { 0 });
        got[0] = 42.0;
        double[] again = m.NextDist(new[] { 0 });
        Assert.Equal(0.1, again[0]);
    }

    [Fact]
    public void MarkovRejectsNonSquare()
    {
        Assert.Throws<ArgumentException>(() => new MarkovModel(new[] { new[] { 0.5, 0.5 } }, 0));
    }

    [Fact]
    public void MarkovRejectsRowsNotSummingToOne()
    {
        Assert.Throws<ArgumentException>(() =>
            new MarkovModel(new[] { new[] { 0.5, 0.6 }, new[] { 0.5, 0.5 } }, 0));
    }

    [Fact]
    public void MarkovAcceptsTinyRowSumDrift()
    {
        MarkovModel m = new(new[] { new[] { 0.5 + 1e-6, 0.5 }, new[] { 0.2, 0.8 } }, 0);
        Assert.Equal(2, m.VocabSize);
    }

    [Fact]
    public void MarkovRejectsSmallButRealDrift()
    {
        Assert.Throws<ArgumentException>(() =>
            new MarkovModel(new[] { new[] { 0.5 + 1e-4, 0.5 }, new[] { 0.2, 0.8 } }, 0));
    }

    [Fact]
    public void SampleUsesInverseCdf()
    {
        double[] dist = { 0.2, 0.5, 0.3 };
        Assert.Equal(0, Distributions.Sample(dist, new ScriptRng(0.1)));
        Assert.Equal(1, Distributions.Sample(dist, new ScriptRng(0.5)));
        Assert.Equal(2, Distributions.Sample(dist, new ScriptRng(0.95)));
    }

    [Fact]
    public void SampleNeverReturnsZeroProbToken()
    {
        LcgRng rng = new(1);
        double[] dist = { 0.0, 1.0, 0.0 };
        for (int i = 0; i < 1000; i++)
        {
            Assert.Equal(1, Distributions.Sample(dist, rng));
        }
    }
}
