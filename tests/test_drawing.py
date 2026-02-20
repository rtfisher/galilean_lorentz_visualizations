"""
tests/test_drawing.py
Smoke tests for the drawing helpers in lorentz_utils.py and the animate
functions in galilean.py / lorentz.py.

All tests run headless via the Agg backend (no display required).
"""

import sys
import os
import pytest

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import lorentz_utils as U


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture
def ax_spatial():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ── Spacetime drawing helpers ───────────────────────────────────────────────

class TestSpacetimeHelpers:
    def test_setup_spacetime_ax(self, ax):
        U.setup_spacetime_ax(ax, "Test Title")
        assert ax.get_title() == "Test Title"
        assert ax.get_xlabel() == "x'"
        assert ax.get_ylabel() == "ct'"

    def test_draw_Sprime_grid(self, ax):
        U.draw_Sprime_grid(ax)
        # Should produce lines without raising
        assert len(ax.lines) + len(ax.collections) > 0

    def test_draw_S_grid_galilean_zero_v(self, ax):
        U.setup_spacetime_ax(ax, "")
        U.draw_S_grid_galilean(ax, v=0.0)

    def test_draw_S_grid_galilean_nonzero_v(self, ax):
        U.setup_spacetime_ax(ax, "")
        U.draw_S_grid_galilean(ax, v=0.5)

    def test_draw_S_grid_lorentz_zero_v(self, ax):
        U.setup_spacetime_ax(ax, "")
        U.draw_S_grid_lorentz(ax, v=0.0, g=1.0)

    def test_draw_S_grid_lorentz_nonzero_v(self, ax):
        U.setup_spacetime_ax(ax, "")
        v = 0.6
        U.draw_S_grid_lorentz(ax, v=v, g=U.gamma(v))

    def test_draw_origin(self, ax):
        U.draw_origin(ax)

    def test_draw_origin_custom_label(self, ax):
        U.draw_origin(ax, label='O')

    def test_draw_spacetime_legend(self, ax):
        U.draw_spacetime_legend(ax)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2

    def test_draw_velocity_label_no_gamma(self, ax):
        U.draw_velocity_label(ax, v=0.5)

    def test_draw_velocity_label_with_gamma(self, ax):
        v = 0.6
        U.draw_velocity_label(ax, v=v, g=U.gamma(v))


# ── Spatial panel helper ────────────────────────────────────────────────────

class TestSpatialPanel:
    @pytest.mark.parametrize("v,g", [
        (0.0, 1.0),    # at rest, Galilean
        (0.5, 1.0),    # moving, Galilean (g=1 → no contraction)
        (0.0, 1.0),    # at rest, Lorentz
        (0.6, None),   # moving, Lorentz with computed gamma
        (0.99, None),  # high v
    ])
    def test_smoke(self, v, g, ax_spatial):
        if g is None:
            g = U.gamma(v)
        s_origin = -v * 1.0
        U.draw_spatial_panel(ax_spatial, v, g, s_origin, "Test Panel")

    def test_galilean_tick_spacing_is_one(self, ax_spatial):
        """With g=1.0 tick spacing should be 1 (no contraction)."""
        # Verify draw_spatial_panel runs without error at g=1
        U.draw_spatial_panel(ax_spatial, 0.5, 1.0, -0.5, "Galilean")

    def test_lorentz_tick_spacing_contracted(self, ax_spatial):
        """With g=gamma>1 tick spacing is 1/γ < 1."""
        v = 0.8
        g = U.gamma(v)
        U.draw_spatial_panel(ax_spatial, v, g, -v, "Lorentz")

    def test_custom_xlim(self, ax_spatial):
        U.draw_spatial_panel(ax_spatial, 0.3, 1.0, -0.3, "Custom",
                             sp_xlim=(-3, 3))


# ── Animate-frame smoke tests ──────────────────────────────────────────────

class TestGalileanAnimateFrame:
    """Call a single animate frame for each Galilean animation."""

    def test_single_panel_frame_zero(self):
        import galilean
        fig, ax = plt.subplots(figsize=(9, 9))

        def animate(frame):
            ax.clear()
            v = U.ramp_v(frame)
            U.setup_spacetime_ax(ax, "Galilean Transformation of Coordinates")
            U.draw_Sprime_grid(ax)
            U.draw_S_grid_galilean(ax, v)
            xAp, ctAp = U.galilean_transform(1, 0, v)
            xBp, ctBp = U.galilean_transform(0, 1, v)
            U.draw_origin(ax)
            U.draw_spacetime_legend(ax)
            U.draw_velocity_label(ax, v)
            return []

        animate(0)
        animate(U.PAUSE_FRAMES + 200)
        plt.close(fig)

    def test_two_panel_frame_zero(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

        def animate(frame):
            ax1.clear(); ax2.clear()
            v = U.ramp_v(frame)
            xAp, ctAp = U.galilean_transform(1, 0, v)
            xBp, ctBp = U.galilean_transform(0, 1, v)
            U.setup_spacetime_ax(ax1, "Spacetime Diagram (S' Frame)")
            U.draw_Sprime_grid(ax1)
            U.draw_S_grid_galilean(ax1, v)
            U.draw_origin(ax1)
            U.draw_spacetime_legend(ax1)
            U.draw_velocity_label(ax1, v)
            U.draw_spatial_panel(ax2, v, 1.0, -v, "Physical Space (S' Frame)",
                                 sp_xlim=(-3, 3))
            return []

        animate(0)
        animate(U.PAUSE_FRAMES + 200)
        plt.close(fig)


class TestLorentzAnimateFrame:
    """Call a single animate frame for the Lorentz three-panel animation."""

    def test_three_panel_frame_zero(self):
        fig = plt.figure(figsize=(16, 10))
        gs  = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.4, wspace=0.3)
        ax_st = fig.add_subplot(gs[:, 0])
        ax_A  = fig.add_subplot(gs[0, 1])
        ax_B  = fig.add_subplot(gs[1, 1])

        PAUSE = 108

        def animate(frame):
            ax_st.clear(); ax_A.clear(); ax_B.clear()
            ramp = max(0, frame - PAUSE)
            v = 0.99 * ramp / 1200
            g = U.gamma(v)
            xAp, ctAp = U.lorentz_transform(1, 0, v)
            xBp, ctBp = U.lorentz_transform(0, 1, v)
            U.setup_spacetime_ax(ax_st, "Spacetime Diagram (S' Frame)")
            U.draw_Sprime_grid(ax_st)
            U.draw_S_grid_lorentz(ax_st, v, g)
            U.draw_origin(ax_st, label='O')
            U.draw_spacetime_legend(ax_st)
            U.draw_velocity_label(ax_st, v, g)
            U.draw_spatial_panel(ax_A, v, g, -v * ctAp,
                                 f"Event A:  snapshot at ct' = {ctAp:.2f}")
            U.draw_spatial_panel(ax_B, v, g, -v * ctBp,
                                 f"Event B:  snapshot at ct' = {ctBp:.2f}")
            return []

        animate(0)
        animate(PAUSE + 600)
        plt.close(fig)
