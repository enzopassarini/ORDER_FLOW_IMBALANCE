"""COPOM signal: linear in the Selic change, deltas capped at +/-100 bp.

    capped_delta = clip(delta_bp, -100, +100)
    signal       = capped_delta / 100           -> in [-1, 1]

Same direction as the Selic move: hike -> positive, cut -> negative, hold -> 0.
E.g. -25 bp -> -0.25, -150 bp -> -1.00, +50 bp -> +0.50.
The trading position is the opposite sign (hike -> short small caps).
"""
import numpy as np

CAP_BP = 100.0


def selic_signal(delta_bp):
    capped = np.clip(np.asarray(delta_bp, dtype=float), -CAP_BP, CAP_BP)
    return capped / CAP_BP + 0.0  # + 0.0 turns -0.0 into 0.0
