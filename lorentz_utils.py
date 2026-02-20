"""
lorentz_utils.py
Shared constants, physics functions, and drawing helpers for the
Galilean (galilean.py) and Lorentz (lorentz.py) animation scripts.
"""

import numpy as np
from matplotlib.animation import FFMpegWriter

# ── Output ─────────────────────────────────────────────────────────────────
OUTDIR = ('/Users/rfisher1/Library/CloudStorage/Dropbox/Teaching'
          '/_2026/_s26/_phy213/_code/_lorentz')
FPS = 12

# ── Default animation timing (Lorentz overrides these) ─────────────────────
PAUSE_FRAMES = 36        # ~3 s at 12 fps
RAMP_FRAMES  = 401
TOTAL_FRAMES = PAUSE_FRAMES + RAMP_FRAMES

# ── Spacetime diagram ──────────────────────────────────────────────────────
XLIM = (-10, 10)
YLIM = (-10, 10)
X_RANGE      = np.linspace(-15, 15, 200)
CT_RANGE     = np.linspace(-15, 15, 200)
COORD_VALUES = np.arange(-8, 10, 2)

# ── Spatial panel ──────────────────────────────────────────────────────────
SP_XLIM = (-3.5, 3.5)
SP_YLIM = (-2.5, 2.5)

# ── Colors ─────────────────────────────────────────────────────────────────
COLOR_SP = 'gray'   # S' frame (primed) — orthogonal background
COLOR_S  = 'blue'   # S frame (unprimed) — tilted / moving


# ── Physics ────────────────────────────────────────────────────────────────

def ramp_v(frame, pause=PAUSE_FRAMES, n_ramp=RAMP_FRAMES - 1, v_max=0.99):
    """Velocity for this animation frame: 0 during pause, then ramps to v_max."""
    return v_max * max(0, frame - pause) / n_ramp


def gamma(v):
    return 1.0 / np.sqrt(1.0 - v**2) if v < 1.0 else 1e6


def galilean_transform(x, ct, v):
    """(x', ct') under the Galilean transform:  x' = x − v·t,  ct' = ct."""
    return x - v * ct, float(ct)


def lorentz_transform(x, ct, v):
    """(x', ct') under the Lorentz transform:  x' = γ(x − v·t),  ct' = γ(ct − v·x)."""
    g = gamma(v)
    return g * (x - v * ct), g * (ct - v * x)


# ── Spacetime diagram helpers ───────────────────────────────────────────────

def setup_spacetime_ax(ax, title):
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x'", fontsize=12)
    ax.set_ylabel("ct'", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')


def draw_Sprime_grid(ax):
    """Gray orthogonal S' coordinate grid — identical in every spacetime panel."""
    for val in COORD_VALUES:
        ax.axhline(y=val, color=COLOR_SP, linewidth=0.8, alpha=0.5)
        ax.axvline(x=val, color=COLOR_SP, linewidth=0.8, alpha=0.5)
    ax.axhline(y=0, color=COLOR_SP, linewidth=2)
    ax.axvline(x=0, color=COLOR_SP, linewidth=2)


def draw_S_grid_galilean(ax, v):
    """Blue S-frame grid in S' coordinates, Galilean transform.

    t = const  →  horizontal lines (simultaneity preserved).
    x = const  →  slanted lines with slope −1/v.
    """
    for t_val in COORD_VALUES:
        ax.axhline(y=t_val, color=COLOR_S, linewidth=0.8, alpha=0.7)
    for x_val in COORD_VALUES:
        x_line = x_val - v * CT_RANGE
        mask = (x_line >= XLIM[0] - 2) & (x_line <= XLIM[1] + 2)
        ax.plot(x_line[mask], CT_RANGE[mask], color=COLOR_S, linewidth=0.8, alpha=0.7)
    # S axes (thicker)
    x_ct = -v * CT_RANGE
    vis  = (x_ct >= XLIM[0]) & (x_ct <= XLIM[1])
    ax.plot(x_ct[vis], CT_RANGE[vis], color=COLOR_S, linewidth=2.5)
    ax.axhline(y=0, color=COLOR_S, linewidth=2.5)


def draw_S_grid_lorentz(ax, v, g):
    """Blue S-frame grid in S' coordinates, Lorentz transform.

    t = const  →  lines of slope −v  (tilt grows with v).
    x = const  →  lines of slope −1/v (symmetric to t=const lines).
    """
    for t_val in COORD_VALUES:
        ct_line = -v * X_RANGE + t_val / g
        ax.plot(X_RANGE, ct_line, color=COLOR_S, linewidth=0.8, alpha=0.7)
    for x_val in COORD_VALUES:
        x_line = -v * CT_RANGE + x_val / g
        mask = (x_line >= XLIM[0] - 2) & (x_line <= XLIM[1] + 2)
        ax.plot(x_line[mask], CT_RANGE[mask], color=COLOR_S, linewidth=0.8, alpha=0.7)
    # S axes (thicker)
    x_ct = -v * CT_RANGE
    vis  = (x_ct >= XLIM[0]) & (x_ct <= XLIM[1])
    ax.plot(x_ct[vis], CT_RANGE[vis], color=COLOR_S, linewidth=2.5)
    ax.plot(X_RANGE, -v * X_RANGE,    color=COLOR_S, linewidth=2.5)


def draw_origin(ax, label='O (origin)'):
    ax.plot(0, 0, 'o', color='black', markersize=12, markerfacecolor='black', zorder=10)
    ax.annotate(label, xy=(0, 0), xytext=(0.5, 0.5), fontsize=10, fontweight='bold')


def draw_spacetime_legend(ax):
    ax.plot([], [], color=COLOR_SP, linewidth=2, label="S' coordinates (x', ct')")
    ax.plot([], [], color=COLOR_S,  linewidth=2, label="S coordinates (x, ct)")
    ax.legend(loc='upper left', fontsize=11)


def draw_velocity_label(ax, v, g=None):
    txt = f"v = {v:.2f}c" + (f"\nγ = {g:.2f}" if g is not None else "")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


# ── Spatial panel helper ────────────────────────────────────────────────────

def draw_spatial_panel(ax, v, g, s_origin, title,
                       sp_xlim=SP_XLIM, sp_ylim=SP_YLIM):
    """Physical-space panel showing S and S' coordinate axes at a given moment.

    Parameters
    ----------
    g        : Lorentz factor.  Pass g=1.0 for Galilean (no length contraction).
    s_origin : x' coordinate of the S frame origin in this snapshot.
    """
    ax.set_xlim(sp_xlim)
    ax.set_ylim(sp_ylim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Background grid
    for val in range(int(sp_xlim[0]), int(sp_xlim[1]) + 1):
        ax.axhline(y=val, color='lightgray', linewidth=0.5, alpha=0.4)
        ax.axvline(x=val, color='lightgray', linewidth=0.5, alpha=0.4)

    # S' axes (gray, stationary at origin)
    ax.axhline(y=0, color=COLOR_SP, linewidth=2, zorder=2)
    ax.axvline(x=0, color=COLOR_SP, linewidth=2, zorder=2)
    ax.text(sp_xlim[1] - 0.3, -0.3,        "x'", color=COLOR_SP, fontsize=13, fontweight='bold')
    ax.text(0.15, sp_ylim[1] - 0.3,        "y'", color=COLOR_SP, fontsize=13, fontweight='bold')

    # S' tick marks
    for tick in range(int(sp_xlim[0]), int(sp_xlim[1]) + 1):
        if tick != 0 and sp_xlim[0] + 0.2 < tick < sp_xlim[1] - 0.2:
            ax.plot([tick, tick], [-0.08, 0.08], color=COLOR_SP, linewidth=1.5, zorder=3)

    # S axes (blue, shifted to s_origin)
    if sp_xlim[0] < s_origin < sp_xlim[1]:
        ax.axvline(x=s_origin, color=COLOR_S, linewidth=2, alpha=0.8, zorder=2)
    # x-axis drawn slightly above y=0 for visibility when frames coincide
    ax.plot(sp_xlim, [0.03, 0.03], color=COLOR_S, linewidth=2, alpha=0.8, zorder=2)
    ax.text(sp_xlim[1] - 0.3, 0.2, "x", color=COLOR_S, fontsize=13, fontweight='bold')
    if sp_xlim[0] + 0.5 < s_origin < sp_xlim[1] - 0.5:
        ax.text(s_origin - 0.35, sp_ylim[1] - 0.3, "y",
                color=COLOR_S, fontsize=13, fontweight='bold')

    # S tick marks: spacing = 1/g  (1.0 for Galilean, 1/γ for Lorentz)
    tick_spacing = 1.0 / g
    for n in range(-8, 9):
        pos = s_origin + n * tick_spacing
        if sp_xlim[0] + 0.2 < pos < sp_xlim[1] - 0.2:
            ax.plot([pos, pos], [0.03 - 0.08, 0.03 + 0.08],
                    color=COLOR_S, linewidth=1.5, alpha=0.8, zorder=3)
            ax.text(pos, 0.25, str(n), color=COLOR_S, fontsize=8, ha='center')

    # Origin labels
    ax.text(0.12, -0.45, "O'", color=COLOR_SP, fontsize=11, fontweight='bold')
    if v > 0.05 and sp_xlim[0] + 0.3 < s_origin < sp_xlim[1] - 0.3 and abs(s_origin) > 0.3:
        ax.text(s_origin + 0.12, -0.45, "O", color=COLOR_S, fontsize=11, fontweight='bold')

    # Velocity arrow
    if v > 0.05:
        if g <= 1.001:
            # Galilean: arrow spans from O' to S origin, showing the displacement
            arrow_y = -1.2
            ax.annotate('', xy=(s_origin + 0.05, arrow_y), xytext=(-0.05, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
            ax.text(s_origin / 2, arrow_y - 0.35, f'v = {v:.2f}c',
                    color='purple', fontsize=12, ha='center', fontweight='bold')
        else:
            # Lorentz: fixed-length direction indicator
            arrow_y = sp_ylim[0] + 0.6
            ax.annotate('', xy=(-0.8, arrow_y), xytext=(0.8, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
            ax.text(0, arrow_y - 0.35, f'S moves at \u2212{v:.2f}c',
                    color='purple', fontsize=9, ha='center', fontweight='bold')


# ── Output ─────────────────────────────────────────────────────────────────

def save_mp4(anim, filename, fps=FPS):
    path = f'{OUTDIR}/{filename}'
    print(f"Saving {filename} ...")
    anim.save(path, writer=FFMpegWriter(fps=fps))
    print(f"  → {path}")
