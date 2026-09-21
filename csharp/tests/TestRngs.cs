using System.Collections.Generic;
using SpecDecode;

namespace SpecDecode.Tests;

/// <summary>Replays a fixed, cycling list of uniforms for exact scenarios.</summary>
internal sealed class ScriptRng : IRng
{
    private readonly IReadOnlyList<double> _values;
    private int _index;

    public ScriptRng(params double[] values)
    {
        _values = values;
    }

    public double NextUniform()
    {
        double v = _values[_index % _values.Count];
        _index++;
        return v;
    }
}

/// <summary>
/// A small 64-bit linear congruential generator. Deterministic and dependency
/// free; only used to drive statistical "many trials" tests.
/// </summary>
internal sealed class LcgRng : IRng
{
    private ulong _state;

    public LcgRng(ulong seed)
    {
        _state = seed;
    }

    public double NextUniform()
    {
        unchecked
        {
            _state = (_state * 6364136223846793005UL) + 1442695040888963407UL;
        }

        return (_state >> 11) * (1.0 / 9007199254740992.0);
    }
}
