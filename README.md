# Lorentz Transformation Visualizations

[![CI](https://github.com/rtfisher/galilean_lorentz_visualizations/actions/workflows/ci.yml/badge.svg)](https://github.com/rtfisher/galilean_lorentz_visualizations/actions/workflows/ci.yml)

Animated spacetime diagrams for PHY 213 demonstrating Galilean and Lorentz
coordinate transformations and the relativity of simultaneity.

---

## File structure

```
lorentz_utils.py          shared constants, physics functions, drawing helpers
galilean.py               generates both Galilean MP4 outputs
lorentz.py                generates the Lorentz MP4 output
tests/
  test_physics.py         unit tests: gamma, transforms, interval invariance
  test_drawing.py         smoke tests: drawing helpers and animate frames
.github/workflows/ci.yml  GitHub Actions CI (runs on every push and PR)
```

---

## Outputs

| Script | Output file | Description |
|---|---|---|
| `galilean.py` | `galilean_animation.mp4` | Single-panel spacetime diagram |
| `galilean.py` | `galilean_twopanel_animation.mp4` | Spacetime + physical-space view |
| `lorentz.py` | `lorentz_twopanel_animation.mp4` | Spacetime + two spatial snapshots (Events A & B) |

---

## Physics

Both scripts track three events defined in the S frame:

| Event | (x, ct) | Character |
|---|---|---|
| O | (0, 0) | Origin — same in all frames |
| A | (1, 0) | Purely spatial separation from O |
| B | (0, 1) | Purely temporal separation from O |

**Galilean** (`galilean.py`): `x′ = x − vt`,  `ct′ = ct`
Lines of simultaneity stay horizontal — time is absolute.

**Lorentz** (`lorentz.py`): `x′ = γ(x − vt)`,  `ct′ = γ(ct − vx)`
Lines of simultaneity tilt with slope v — simultaneity is relative.
The spacetime interval `s² = Δx² − Δ(ct)²` is shown to be frame-invariant.

---

## Requirements

- Python packages: `numpy`, `matplotlib` (available in the `npscipy` conda environment)
- `ffmpeg` for MP4 encoding

---

## Usage

```bash
conda activate npscipy

python galilean.py   # writes galilean_animation.mp4 and galilean_twopanel_animation.mp4
python lorentz.py    # writes lorentz_twopanel_animation.mp4
```

---

## Tests

```bash
MPLBACKEND=Agg conda run -n npscipy python -m pytest tests/ -v
```

The CI workflow runs the same tests automatically on every push and pull request.

---

## Color conventions

| Color | Frame |
|---|---|
| Gray | S′ frame (primed) — the rest frame of the diagrams |
| Blue | S frame (unprimed) — the boosted frame |
| Red dot | Event A |
| Green dot | Event B |
| Black dot | Origin O |
