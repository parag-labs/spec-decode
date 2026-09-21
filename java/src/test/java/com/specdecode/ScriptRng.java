package com.specdecode;

/** Replays a fixed, cycling list of uniforms for exact scenarios. */
final class ScriptRng implements Rng {
    private final double[] values;
    private int index;

    ScriptRng(double... values) {
        this.values = values;
    }

    @Override
    public double nextUniform() {
        double v = values[index % values.length];
        index++;
        return v;
    }
}
