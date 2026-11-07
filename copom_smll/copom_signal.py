"""COPOM signal: linear in the Selic change, bounded to [-1, 1]."""
import numpy as np

SCALE_BP = 100.0  # a 100 bp move (or larger) is a full-size signal


def selic_signal(delta_bp):
    """-delta_bp / 100, clipped to [-1, 1]. Cut -> positive (long small caps)."""
    return np.clip(-np.asarray(delta_bp, dtype=float) / SCALE_BP, -1.0, 1.0) + 0.0  # no -0.0
