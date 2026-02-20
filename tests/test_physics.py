"""
tests/test_physics.py
Unit tests for the physics functions in lorentz_utils.py.
Covers Galilean and Lorentz transforms, gamma, ramp_v, and the
spacetime-interval invariance that is the core physical claim of the animations.
"""

import sys
import os
import math
import pytest

# Allow importing from the project root regardless of where pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import lorentz_utils as U


# ── gamma ──────────────────────────────────────────────────────────────────

class TestGamma:
    def test_at_rest(self):
        assert U.gamma(0.0) == pytest.approx(1.0)

    def test_half_c(self):
        # γ(0.5) = 1/√(1−0.25) = 1/√0.75 ≈ 1.1547
        assert U.gamma(0.5) == pytest.approx(1.0 / math.sqrt(0.75), rel=1e-6)

    def test_high_v(self):
        # γ(0.99) ≈ 7.089
        assert U.gamma(0.99) == pytest.approx(1.0 / math.sqrt(1 - 0.99**2), rel=1e-6)

    def test_exceeds_unity_guard(self):
        # v >= 1 should return the sentinel large value, not blow up
        assert U.gamma(1.0) == pytest.approx(1e6)
        assert U.gamma(1.5) == pytest.approx(1e6)

    def test_positive(self):
        for v in [0.0, 0.1, 0.5, 0.9, 0.99]:
            assert U.gamma(v) >= 1.0


# ── ramp_v ─────────────────────────────────────────────────────────────────

class TestRampV:
    def test_pause_returns_zero(self):
        # During the pause window velocity should be 0
        for frame in range(U.PAUSE_FRAMES):
            assert U.ramp_v(frame) == 0.0

    def test_first_ramp_frame(self):
        # First frame after pause should be just above 0
        v = U.ramp_v(U.PAUSE_FRAMES)
        assert v == pytest.approx(0.0, abs=1e-4)

    def test_last_frame_approaches_vmax(self):
        v = U.ramp_v(U.TOTAL_FRAMES - 1)
        assert v == pytest.approx(0.99, rel=1e-3)

    def test_monotone(self):
        vs = [U.ramp_v(f) for f in range(U.TOTAL_FRAMES)]
        for a, b in zip(vs, vs[1:]):
            assert b >= a

    def test_custom_parameters(self):
        # Custom pause=10, n_ramp=100, v_max=0.5
        v = U.ramp_v(110, pause=10, n_ramp=100, v_max=0.5)
        assert v == pytest.approx(0.5, rel=1e-6)


# ── Galilean transform ─────────────────────────────────────────────────────

class TestGalileanTransform:
    """
    x' = x − v·t,   ct' = ct   (absolute time).
    """

    def test_at_rest(self):
        xp, ctp = U.galilean_transform(3.0, 2.0, v=0.0)
        assert xp  == pytest.approx(3.0)
        assert ctp == pytest.approx(2.0)

    def test_event_A(self):
        # (x, ct) = (1, 0): x' = 1 − v·0 = 1; ct' = 0
        xp, ctp = U.galilean_transform(1, 0, v=0.5)
        assert xp  == pytest.approx(1.0)
        assert ctp == pytest.approx(0.0)

    def test_event_B(self):
        # (x, ct) = (0, 1): x' = 0 − v·1 = −v; ct' = 1
        v = 0.6
        xp, ctp = U.galilean_transform(0, 1, v=v)
        assert xp  == pytest.approx(-v)
        assert ctp == pytest.approx(1.0)

    def test_absolute_time(self):
        # ct' must always equal ct regardless of v
        for v in [0.0, 0.3, 0.8, 0.99]:
            _, ctp = U.galilean_transform(5.0, 3.7, v=v)
            assert ctp == pytest.approx(3.7)

    def test_origin_maps_to_origin(self):
        xp, ctp = U.galilean_transform(0, 0, v=0.7)
        assert xp  == pytest.approx(0.0)
        assert ctp == pytest.approx(0.0)


# ── Lorentz transform ──────────────────────────────────────────────────────

class TestLorentzTransform:
    """
    x'  = γ(x − v·t),   ct' = γ(ct − v·x).
    """

    def test_at_rest(self):
        xp, ctp = U.lorentz_transform(3.0, 2.0, v=0.0)
        assert xp  == pytest.approx(3.0)
        assert ctp == pytest.approx(2.0)

    def test_event_A(self):
        # (x, ct) = (1, 0): x' = γ; ct' = −γv
        v = 0.6
        g = U.gamma(v)
        xp, ctp = U.lorentz_transform(1, 0, v=v)
        assert xp  == pytest.approx(g)
        assert ctp == pytest.approx(-g * v)

    def test_event_B(self):
        # (x, ct) = (0, 1): x' = −γv; ct' = γ
        v = 0.6
        g = U.gamma(v)
        xp, ctp = U.lorentz_transform(0, 1, v=v)
        assert xp  == pytest.approx(-g * v)
        assert ctp == pytest.approx(g)

    def test_origin_maps_to_origin(self):
        xp, ctp = U.lorentz_transform(0, 0, v=0.7)
        assert xp  == pytest.approx(0.0)
        assert ctp == pytest.approx(0.0)

    def test_reduces_to_galilean_at_low_v(self):
        # At small v, γ ≈ 1 and the transforms should be nearly identical.
        # Use x=0 so the Lorentz simultaneity correction −γv·x vanishes;
        # otherwise even small v gives a noticeable ct' difference when x is large.
        v = 0.001
        xg, ctg = U.galilean_transform(0.0, 2.0, v=v)
        xl, ctl = U.lorentz_transform(0.0, 2.0, v=v)
        assert xl  == pytest.approx(xg,  abs=1e-5)
        assert ctl == pytest.approx(ctg, abs=1e-5)


# ── Spacetime interval invariance ──────────────────────────────────────────

class TestSpacetimeIntervalInvariance:
    """
    The key physical claim: s² = Δx² − Δ(ct)² is invariant under Lorentz boosts.
    Galilean transforms do NOT preserve s².
    """

    def _interval(self, x, ct):
        return x**2 - ct**2

    @pytest.mark.parametrize("v", [0.0, 0.3, 0.6, 0.9, 0.99])
    def test_lorentz_preserves_interval_event_A(self, v):
        xp, ctp = U.lorentz_transform(1, 0, v=v)
        assert self._interval(xp, ctp) == pytest.approx(self._interval(1, 0), abs=1e-10)

    @pytest.mark.parametrize("v", [0.0, 0.3, 0.6, 0.9, 0.99])
    def test_lorentz_preserves_interval_event_B(self, v):
        xp, ctp = U.lorentz_transform(0, 1, v=v)
        assert self._interval(xp, ctp) == pytest.approx(self._interval(0, 1), abs=1e-10)

    @pytest.mark.parametrize("v", [0.3, 0.6, 0.9])
    def test_galilean_does_not_preserve_interval_event_B(self, v):
        # (x, ct) = (0, 1): s² = −1, but after Galilean boost s² ≠ −1
        xp, ctp = U.galilean_transform(0, 1, v=v)
        assert self._interval(xp, ctp) != pytest.approx(self._interval(0, 1), abs=1e-6)

    @pytest.mark.parametrize("v", [0.3, 0.6, 0.9])
    def test_galilean_preserves_event_A_interval(self, v):
        # Event A has ct = 0, so Galilean ct' = ct = 0 and x' = 1 always
        xp, ctp = U.galilean_transform(1, 0, v=v)
        assert self._interval(xp, ctp) == pytest.approx(self._interval(1, 0), abs=1e-10)


# ── Constants sanity checks ─────────────────────────────────────────────────

class TestConstants:
    def test_frame_count_consistent(self):
        assert U.TOTAL_FRAMES == U.PAUSE_FRAMES + U.RAMP_FRAMES

    def test_axis_limits_symmetric(self):
        assert U.XLIM[0] == -U.XLIM[1]
        assert U.YLIM[0] == -U.YLIM[1]

    def test_coord_values_within_axes(self):
        import numpy as np
        assert all(U.XLIM[0] < v < U.XLIM[1] for v in U.COORD_VALUES)
        assert all(U.YLIM[0] < v < U.YLIM[1] for v in U.COORD_VALUES)

    def test_range_arrays_span_beyond_axes(self):
        # X_RANGE and CT_RANGE must extend beyond axis limits for clipping to work
        import numpy as np
        assert U.X_RANGE.min()  < U.XLIM[0]
        assert U.X_RANGE.max()  > U.XLIM[1]
        assert U.CT_RANGE.min() < U.YLIM[0]
        assert U.CT_RANGE.max() > U.YLIM[1]
