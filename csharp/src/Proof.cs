using System;
using System.Collections.Generic;

namespace SpecDecode;

/// <summary>
/// The exactness proof, expressed constructively. Speculative decoding is
/// distribution-preserving: the token actually emitted for a given context is
/// distributed exactly according to the target model, regardless of the draft.
/// These helpers compute that induced distribution in closed form so tests can
/// assert it matches the target to floating-point tolerance.
/// </summary>
public static class Proof
{
    /// <summary>
    /// Closed-form distribution of the token emitted by one accept/reject
    /// round given target distribution <paramref name="p"/> and draft
    /// distribution <paramref name="q"/>. Equals <paramref name="p"/> exactly.
    /// </summary>
    public static double[] InducedFromDists(IReadOnlyList<double> p, IReadOnlyList<double> q)
    {
        if (p is null)
        {
            throw new ArgumentNullException(nameof(p));
        }

        if (q is null)
        {
            throw new ArgumentNullException(nameof(q));
        }

        int n = p.Count;
        double[] accepted = new double[n];
        double rejectionMass = 0.0;
        for (int i = 0; i < n; i++)
        {
            double a = Speculative.AcceptProbability(p[i], q[i]);
            accepted[i] = q[i] * a;
            rejectionMass += q[i] * (1.0 - a);
        }

        double[] r = Speculative.Residual(p, q);
        double[] outp = new double[n];
        for (int i = 0; i < n; i++)
        {
            outp[i] = accepted[i] + (rejectionMass * r[i]);
        }

        return outp;
    }

    /// <summary>
    /// Induced next-token distribution for a concrete target/draft pair and
    /// token <paramref name="prefix"/>.
    /// </summary>
    public static double[] InducedNextTokenDistribution(
        IModel target,
        IModel draft,
        IReadOnlyList<int> prefix)
    {
        return InducedFromDists(target.NextDist(prefix), draft.NextDist(prefix));
    }

    /// <summary>
    /// Maximum total-variation distance between the induced distribution and
    /// the target distribution across the supplied <paramref name="prefixes"/>.
    /// For a correct sampler this is zero up to rounding.
    /// </summary>
    public static double MaxTotalVariation(
        IModel target,
        IModel draft,
        IReadOnlyList<IReadOnlyList<int>> prefixes)
    {
        double worst = 0.0;
        foreach (IReadOnlyList<int> prefix in prefixes)
        {
            double[] induced = InducedNextTokenDistribution(target, draft, prefix);
            double[] p = target.NextDist(prefix);
            double tv = 0.0;
            for (int i = 0; i < p.Length; i++)
            {
                tv += Math.Abs(induced[i] - p[i]);
            }

            tv *= 0.5;
            if (tv > worst)
            {
                worst = tv;
            }
        }

        return worst;
    }
}
