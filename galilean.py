"""
galilean.py
Generates two MP4 animations of the Galilean transformation:

  galilean_animation.mp4          — single-panel spacetime diagram
  galilean_twopanel_animation.mp4 — spacetime diagram + physical-space panel

Events O, A, B are defined in the S frame and transformed to S' via:
    x' = x − v·t,   ct' = ct   (absolute time — simultaneity is preserved)

Run:
    python galilean.py
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lorentz_utils import (
    PAUSE_FRAMES, RAMP_FRAMES, TOTAL_FRAMES,
    COLOR_SP, COLOR_S,
    ramp_v, galilean_transform,
    setup_spacetime_ax, draw_Sprime_grid, draw_S_grid_galilean,
    draw_origin, draw_spacetime_legend, draw_velocity_label,
    draw_spatial_panel, save_mp4,
)

# ── Canonical events in S frame ────────────────────────────────────────────
xA, ctA = 1, 0   # Event A: purely spatial separation from O
xB, ctB = 0, 1   # Event B: purely temporal separation from O


# ─────────────────────────────────────────────────────────────────────────────
# 1. SINGLE-PANEL: Spacetime diagram only
# ─────────────────────────────────────────────────────────────────────────────

def run_single():
    fig, ax = plt.subplots(figsize=(9, 9))

    def animate(frame):
        ax.clear()
        v = ramp_v(frame)

        setup_spacetime_ax(ax, "Galilean Transformation of Coordinates")
        draw_Sprime_grid(ax)
        draw_S_grid_galilean(ax, v)

        xAp, ctAp = galilean_transform(xA, ctA, v)
        xBp, ctBp = galilean_transform(xB, ctB, v)

        draw_origin(ax)

        ax.plot(xAp, ctAp, 'ro', markersize=12, zorder=10)
        ax.annotate('A\n(x=1, ct=0)', xy=(xAp, ctAp),
                    xytext=(xAp + 0.5, ctAp - 1.5), fontsize=9, color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.plot(xBp, ctBp, 'go', markersize=12, zorder=10)
        ax.annotate('B\n(x=0, ct=1)', xy=(xBp, ctBp),
                    xytext=(xBp + 0.5, ctBp + 1.0), fontsize=9, color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.text(0.02, 0.12,
                f"O → A:  Δx = 1, Δct = 0  →  Δx' = {xAp:.2f}, Δct' = {ctAp:.2f}\n"
                f"O → B:  Δx = 0, Δct = 1  →  Δx' = {xBp:.2f}, Δct' = {ctBp:.2f}",
                transform=ax.transAxes, fontsize=10, va='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        draw_spacetime_legend(ax)
        draw_velocity_label(ax, v)   # no γ in Galilean
        return []

    anim = FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=50, blit=False)
    save_mp4(anim, 'galilean_animation.mp4')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TWO-PANEL: Spacetime diagram + physical-space view
# ─────────────────────────────────────────────────────────────────────────────

def run_two():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    def animate(frame):
        ax1.clear()
        ax2.clear()
        v = ramp_v(frame)

        xAp, ctAp = galilean_transform(xA, ctA, v)
        xBp, ctBp = galilean_transform(xB, ctB, v)

        # ── Left: spacetime diagram ────────────────────────────────────────
        setup_spacetime_ax(ax1, "Spacetime Diagram (S' Frame)")
        draw_Sprime_grid(ax1)
        draw_S_grid_galilean(ax1, v)

        draw_origin(ax1)

        ax1.plot(xAp, ctAp, 'ro', markersize=12, zorder=10)
        ax1.annotate('A\n(x=1, ct=0)', xy=(xAp, ctAp),
                     xytext=(xAp + 0.5, ctAp - 1.5), fontsize=9, color='darkred',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax1.plot(xBp, ctBp, 'go', markersize=12, zorder=10)
        ax1.annotate('B\n(x=0, ct=1)', xy=(xBp, ctBp),
                     xytext=(xBp + 0.5, ctBp + 1.0), fontsize=9, color='darkgreen',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax1.text(0.02, 0.12,
                 f"O → A:  Δx = 1, Δct = 0  →  Δx' = {xAp:.2f}, Δct' = {ctAp:.2f}\n"
                 f"O → B:  Δx = 0, Δct = 1  →  Δx' = {xBp:.2f}, Δct' = {ctBp:.2f}",
                 transform=ax1.transAxes, fontsize=10, va='bottom', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        draw_spacetime_legend(ax1)
        draw_velocity_label(ax1, v)

        # ── Right: physical-space snapshot ────────────────────────────────
        # In Galilean mechanics time is absolute, so a single snapshot suffices.
        # The S origin drifts to x' = −v when ct = 1 (the time of Event B).
        s_ox = -v
        draw_spatial_panel(ax2, v, 1.0, s_ox, "Physical Space (S' Frame)",
                           sp_xlim=(-3, 3))

        # Event A: always at x' = 1 (t = 0, frames coincide at t = 0)
        ax2.plot(1, 0, 'ro', markersize=14, zorder=10)
        ax2.annotate('A', xy=(1, 0), xytext=(1, 0.55),
                     fontsize=12, color='darkred', fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        # Event B: at x' = xBp = −v  (S origin has moved here by ct = 1)
        ax2.plot(xBp, 0, 'go', markersize=14, zorder=10)
        ax2.annotate('B', xy=(xBp, 0), xytext=(xBp, 0.55),
                     fontsize=12, color='darkgreen', fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        ax2.plot([], [], color=COLOR_SP, linewidth=2, label="S' frame (at rest)")
        ax2.plot([], [], color=COLOR_S,  linewidth=2, label="S frame (moving at −v)")
        ax2.legend(loc='upper left', fontsize=11)

        return []

    anim = FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=50, blit=False)
    save_mp4(anim, 'galilean_twopanel_animation.mp4')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_single()
    run_two()
