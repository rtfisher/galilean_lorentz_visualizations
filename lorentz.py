"""
lorentz.py
Generates one MP4 animation of the Lorentz transformation:

  lorentz_twopanel_animation.mp4

Three panels:
  Left         — Spacetime diagram (S coordinate grid in S' frame)
  Right top    — Physical-space snapshot for Event A (relativity of simultaneity)
  Right bottom — Physical-space snapshot for Event B (time dilation + drift)

Events O, A, B are defined in S and transformed via the direct Lorentz transform:
    x'  = γ(x − v·t),   ct' = γ(ct − v·x)

The velocity ramp is 3× slower than the Galilean scripts so the relativistic
effects (tilting grid, length contraction, time dilation) develop gradually.

Run:
    python lorentz.py
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lorentz_utils import (
    SP_XLIM, SP_YLIM,
    COLOR_S,
    gamma, lorentz_transform,
    setup_spacetime_ax, draw_Sprime_grid, draw_S_grid_lorentz,
    draw_origin, draw_spacetime_legend, draw_velocity_label,
    draw_spatial_panel, save_mp4,
)

# ── Timing: slower ramp so relativistic effects develop visibly ────────────
PAUSE_FRAMES = 108     # ~9 s at 12 fps
RAMP_FRAMES  = 1203
TOTAL_FRAMES = PAUSE_FRAMES + RAMP_FRAMES

# ── Canonical events in S frame ────────────────────────────────────────────
xA, ctA = 1, 0   # Event A: purely spatial separation from O
xB, ctB = 0, 1   # Event B: purely temporal separation from O


def run():
    # ── Layout: spacetime diagram (full height left) + two spatial panels ──
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.4, wspace=0.3)
    ax_st = fig.add_subplot(gs[:, 0])   # spacetime (full height)
    ax_A  = fig.add_subplot(gs[0, 1])   # Event A spatial (top right)
    ax_B  = fig.add_subplot(gs[1, 1])   # Event B spatial (bottom right)

    def animate(frame):
        ax_st.clear()
        ax_A.clear()
        ax_B.clear()

        ramp = max(0, frame - PAUSE_FRAMES)
        v = 0.99 * ramp / 1200
        g = gamma(v)

        xAp, ctAp = lorentz_transform(xA, ctA, v)
        xBp, ctBp = lorentz_transform(xB, ctB, v)

        # ── Left: spacetime diagram ────────────────────────────────────────
        setup_spacetime_ax(ax_st, "Spacetime Diagram (S' Frame)")
        draw_Sprime_grid(ax_st)
        draw_S_grid_lorentz(ax_st, v, g)

        s2A       = xA**2  - ctA**2        # spacetime interval in S
        s2A_prime = xAp**2 - ctAp**2       # spacetime interval in S'
        s2B       = xB**2  - ctB**2
        s2B_prime = xBp**2 - ctBp**2

        draw_origin(ax_st, label='O')

        ax_st.plot(xAp, ctAp, 'ro', markersize=10, zorder=10)
        ax_st.annotate('A', xy=(xAp, ctAp),
                       xytext=(xAp + 0.5, ctAp - 1.5),
                       fontsize=10, color='darkred', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax_st.plot(xBp, ctBp, 'go', markersize=10, zorder=10)
        ax_st.annotate('B', xy=(xBp, ctBp),
                       xytext=(xBp + 0.5, ctBp + 1.0),
                       fontsize=10, color='darkgreen', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax_st.text(0.02, 0.15,
                   f"O\u2192A: \u0394x=1, \u0394ct=0 \u2192 "
                   f"\u0394x'={xAp:.2f}, \u0394ct'={ctAp:.2f}\n"
                   f"  s\u00b2=\u0394x\u00b2\u2212\u0394(ct)\u00b2:  "
                   f"S:{s2A:.0f}   S':{s2A_prime:.1f}\n\n"
                   f"O\u2192B: \u0394x=0, \u0394ct=1 \u2192 "
                   f"\u0394x'={xBp:.2f}, \u0394ct'={ctBp:.2f}\n"
                   f"  s\u00b2=\u0394x\u00b2\u2212\u0394(ct)\u00b2:  "
                   f"S:{s2B:.0f}   S':{s2B_prime:.1f}",
                   transform=ax_st.transAxes, fontsize=9, va='bottom', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        draw_spacetime_legend(ax_st)
        draw_velocity_label(ax_st, v, g)

        # ── Right top: Event A spatial snapshot ───────────────────────────
        # At ct' = ctAp, the S origin is at x' = −v·ctAp  (= γv², slightly right of O')
        s_origin_A = -v * ctAp
        draw_spatial_panel(ax_A, v, g, s_origin_A,
                           f"Event A:  snapshot at ct' = {ctAp:.2f}")

        if SP_XLIM[0] < xAp < SP_XLIM[1]:
            ax_A.plot(xAp, 0, 'ro', markersize=14, zorder=10)
            ax_A.annotate('A', xy=(xAp, 0), xytext=(xAp, 0.6),
                          fontsize=12, color='darkred', fontweight='bold', ha='center',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        # Length-contraction bracket: distance from S origin to Event A = 1/γ
        if v > 0.05:
            bracket_ok = (SP_XLIM[0] + 0.2 < s_origin_A < SP_XLIM[1] - 0.2 and
                          SP_XLIM[0] + 0.2 < xAp       < SP_XLIM[1] - 0.2 and
                          abs(xAp - s_origin_A) > 0.1)
            if bracket_ok:
                y_b = -0.8
                ax_A.annotate('', xy=(xAp, y_b), xytext=(s_origin_A, y_b),
                              arrowprops=dict(arrowstyle='<->', color='darkred', lw=2))
                ax_A.text((xAp + s_origin_A) / 2, y_b - 0.35,
                          f"\u0394x' = 1/\u03b3 = {1/g:.2f}",
                          color='darkred', fontsize=10, ha='center', fontweight='bold')

        # ── Right bottom: Event B spatial snapshot ────────────────────────
        # At ct' = ctBp, the S origin has drifted to x' = −v·ctBp  (= −γv, left of O')
        s_origin_B = -v * ctBp
        draw_spatial_panel(ax_B, v, g, s_origin_B,
                           f"Event B:  snapshot at ct' = {ctBp:.2f}")

        if SP_XLIM[0] < xBp < SP_XLIM[1]:
            ax_B.plot(xBp, 0, 'go', markersize=14, zorder=10)
            ax_B.annotate('B', xy=(xBp, 0), xytext=(xBp, 0.6),
                          fontsize=12, color='darkgreen', fontweight='bold', ha='center',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        # Time-dilation annotation
        if v > 0.05:
            ax_B.text(0.98, 0.92,
                      f"S clock:  ct = 1\nS' clock: ct' = \u03b3 = {g:.2f}",
                      transform=ax_B.transAxes, fontsize=10, fontweight='bold',
                      ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

        return []

    anim = FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=50, blit=False)
    save_mp4(anim, 'lorentz_twopanel_animation.mp4')
    plt.close(fig)


if __name__ == '__main__':
    run()
