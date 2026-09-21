using System;
using System.Collections.Generic;

namespace SpecDecode;

/// <summary>
/// A source of uniform random doubles in the half-open interval [0, 1).
/// The speculative sampler is parameterised over this interface so that its
/// accept/reject logic is fully deterministic and testable.
/// </summary>
public interface IRng
{
    /// <summary>Returns the next uniform sample in [0, 1).</summary>
    double NextUniform();
}

/// <summary>
/// An autoregressive model: given a token prefix it yields the probability
/// distribution over the next token.
/// </summary>
public interface IModel
{
    /// <summary>Number of tokens in the vocabulary.</summary>
    int VocabSize { get; }

    /// <summary>
    /// Returns the next-token distribution for <paramref name="tokens"/>.
    /// The returned array is a fresh copy the caller may mutate.
    /// </summary>
    double[] NextDist(IReadOnlyList<int> tokens);
}

/// <summary>
/// A first-order Markov model backed by an explicit row-stochastic
/// transition matrix. This is the deterministic, dependency-free core used
/// by the exactness proof and the sampler tests.
/// </summary>
public sealed class MarkovModel : IModel
{
    // numpy.allclose default tolerance: atol + rtol against a target of 1.0.
    private const double RowSumTolerance = 1e-8 + 1e-5;

    private readonly double[][] _transition;
    private readonly int _startToken;

    /// <inheritdoc/>
    public int VocabSize { get; }

    /// <summary>
    /// Builds a model from a square transition matrix whose rows sum to 1.
    /// </summary>
    /// <param name="transition">Row-stochastic transition matrix.</param>
    /// <param name="startToken">Token assumed to precede an empty prefix.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square or a row does not sum to 1.
    /// </exception>
    public MarkovModel(double[][] transition, int startToken = 0)
    {
        if (transition is null)
        {
            throw new ArgumentNullException(nameof(transition));
        }

        int n = transition.Length;
        if (n == 0)
        {
            throw new ArgumentException("transition must be a non-empty square matrix");
        }

        _transition = new double[n][];
        for (int i = 0; i < n; i++)
        {
            double[] row = transition[i];
            if (row.Length != n)
            {
                throw new ArgumentException("transition must be a square matrix");
            }

            double sum = 0.0;
            foreach (double v in row)
            {
                sum += v;
            }

            if (Math.Abs(sum - 1.0) > RowSumTolerance)
            {
                throw new ArgumentException($"transition row {i} must sum to 1, got {sum}");
            }

            _transition[i] = (double[])row.Clone();
        }

        VocabSize = n;
        _startToken = startToken;
    }

    /// <inheritdoc/>
    public double[] NextDist(IReadOnlyList<int> tokens)
    {
        if (tokens is null)
        {
            throw new ArgumentNullException(nameof(tokens));
        }

        int last = tokens.Count > 0 ? tokens[tokens.Count - 1] : _startToken;
        return (double[])_transition[last].Clone();
    }
}

/// <summary>Distribution helpers shared by the sampler and the proof.</summary>
public static class Distributions
{
    /// <summary>
    /// Returns a probability distribution proportional to <paramref name="dist"/>.
    /// A vector whose entries sum to zero (or less) normalises to uniform.
    /// </summary>
    public static double[] Normalize(IReadOnlyList<double> dist)
    {
        if (dist is null)
        {
            throw new ArgumentNullException(nameof(dist));
        }

        int n = dist.Count;
        double total = 0.0;
        for (int i = 0; i < n; i++)
        {
            total += dist[i];
        }

        double[] outp = new double[n];
        if (total <= 0.0)
        {
            double u = 1.0 / n;
            for (int i = 0; i < n; i++)
            {
                outp[i] = u;
            }

            return outp;
        }

        for (int i = 0; i < n; i++)
        {
            outp[i] = dist[i] / total;
        }

        return outp;
    }

    /// <summary>
    /// Draws an index from <paramref name="dist"/> using one uniform sample
    /// and inverse-CDF lookup. The distribution is normalised first, so an
    /// index with zero probability is never returned.
    /// </summary>
    public static int Sample(IReadOnlyList<double> dist, IRng rng)
    {
        if (rng is null)
        {
            throw new ArgumentNullException(nameof(rng));
        }

        double[] nd = Normalize(dist);
        double u = rng.NextUniform();
        double cum = 0.0;
        for (int i = 0; i < nd.Length; i++)
        {
            cum += nd[i];
            if (u < cum)
            {
                return i;
            }
        }

        return nd.Length - 1;
    }
}
