"""
run_and_plot.py -- solve one mode by one method and plot the solution (the fields figure of Sec. II): the
real part of the ten retarded-field components psi^0 ... psi^9 of mode m for the circular orbit at r0, as
surfaces over (r*, theta), one panel per component, stacked in one column. (The style of Fig. 5 of
arXiv:2608.07934, whose m = 2 column it reproduces with the default settings.)

    python run_and_plot.py                         # the settings below: solve, save the field, plot
    python run_and_plot.py --m 3 --r0 7 --method direct
    python run_and_plot.py --replot                # redraw from the saved .npz, without solving

The retarded field inside the worldtube is the solved residual field plus the puncture
(retarded_field_components), and r* is shown up to RSTAR_MAX = 100M. The grid is the paper's for m
(settings.GRID; for m = 2: Delta r* = 0.28, Delta theta = pi/45, r* from about -10M to 102M at r0 = 6M).
The field is singular at the particle: values beyond each panel's range (set by the field outside the
worldtube) are cut off flat and drawn in grey.

Writes figures/fields_m<m>_r<r0>.pdf / .png (the surfaces rasterized at 300 dpi, everything else vector)
and the plotted field as figures/fields_m<m>_r<r0>.npz.
"""

import settings  # noqa: I001  first: sets the thread counts, makes the package importable

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ellipticgsf_neural import METHODS, build_problem, retarded_field_components, solve
from plot_style import INK, publication_style, save_figure

# ------------------------------------------------------------------ SETTINGS
M = 2 # m-mode
R0 = 6.0
METHOD = "neural"                 # any method of solve(): "direct", "ilu0", "ilu1", "neural"
RSTAR_MAX = 100.0                 # shown up to this r* (the paper's truncation)
OUTPUT_DIR = settings.FIGURES     # where the figure and its data go
FIGURE_WIDTH = 3.4                # inches: one column (APS / PRD)
PANEL_STEP = 0.59                 # inches from one component's panel to the next (the 3D boxes overlap
                                  #   their neighbours' empty axes space, not each other)
AXES_HEIGHT = 1.40                # inches of each panel's 3D axes, which sets the size of its box
BOX_LEFT = 0.05                   # left edge of the 3D axes, which are a figure wide (the box sits
                                  #   right of centre in them, clearing the Re psi labels)
BOX_ASPECT, BOX_ZOOM = (6.0, 1.23, 0.81), 1.93   # r*, theta, value proportions of each box, and its size
SHOW_PARAMETERS = True            # "m = 2, r0 = 6M" above the top panel (off: put it in the caption)

COMPONENT_COLOURS = [             # one hue per component psi^0 ... psi^9, around the colour wheel;
    "#d64545", "#e8743b", "#e3a21a",  # each panel is shaded from a dark to a light version of its hue
    "#8cbf26", "#2f9e44", "#1aa39a",
    "#2b9bd1", "#3b6fd6", "#7b55d6", "#c4489a",
]
CAP_COLOR = "#d9d9d6"             # surfaces cut off at the panel's range
MESH_COLOR = "#1b1b1a"
MESH_RSTAR = 5.0                  # mesh lines every 5M in r* ...
MESH_THETA = 5                    # ... and every 5th theta node (pi/9)
Z_RANGE = {}                      # component: (low, high) to override the automatic range, e.g. {9: (-0.4, 1.2)}
THETA_LABEL_AT = 1.5               # radians along the theta axis ...
THETA_LABEL_OFFSET = (-17, -9)     # ... and offset from it (points), past the tick labels
ELEV, AZIM = 16, -112              # camera (degrees): low, looking along theta, as in the paper
# ----------------------------------------------------------------------------


def field_data(problem, solution) -> dict:
    """Re psi^l of the retarded field on the grid, cut at RSTAR_MAX, as plain arrays."""
    g = problem.grid
    keep = g.rstar <= RSTAR_MAX + 1e-9
    return dict(m=problem.m, r0=problem.orbit.r0, rstar0=problem.orbit.rstar0, method=solution.method,
                iterations=solution.iterations or 0, residual=solution.relative_residual,
                rstar=g.rstar[keep], theta=g.theta,
                worldtube=np.array([[g.rstar[i] for i in g.worldtube_rstar_indices],
                                    [g.theta[j] for j in g.worldtube_theta_indices]]),
                re_psi=retarded_field_components(problem, solution.x)[:, keep, :].real)      # (10, Nr, Nth)


# ---------------------------------------------------------------------- drawing

def panel_range(z, rstar, theta, worldtube):
    """(low, high) of the field outside the worldtube (where the retarded field is singular), padded by 8%."""
    (r_lo, r_hi), (t_lo, t_hi) = worldtube
    R, T = np.meshgrid(rstar, theta, indexing="ij")
    away = (R < r_lo) | (R > r_hi) | (T < t_lo) | (T > t_hi)
    lo, hi = float(z[away].min()), float(z[away].max())
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def shade_range(colour):
    """(dark, light) RGB of a hue: the ends of its Lambert shading."""
    c = np.array(to_rgb(colour))
    return 0.30 * c, c + 0.12 * (1.0 - c)


def shaded_surface(ax, rstar, theta, z, lo, hi, colour):
    """The surface as one depth-sorted collection of facets and mesh segments (so hidden mesh lines stay
    hidden), Lambert-shaded in `colour`, with the cut-off parts flat and grey."""
    zc = np.clip(z, lo, hi)
    R, T = np.meshgrid(rstar, theta, indexing="ij")
    P = np.stack([R, T, zc], axis=-1)                                        # (Nr, Nth, 3)
    quads = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2).reshape(-1, 4, 3)

    # normals in the box's own proportions, where the light lives
    span = np.array([rstar[-1] - rstar[0], theta[-1] - theta[0], hi - lo])
    scale = np.asarray(ax.get_box_aspect()) / span
    q = quads * scale
    n = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n *= np.sign(n[:, 2:3] + 1e-12)                                         # upward
    light = np.array([-0.45, -0.55, 0.70])
    light /= np.linalg.norm(light)
    k = 0.18 + 0.82 * np.clip(n @ light, 0.0, 1.0) ** 1.2
    dark, bright = shade_range(colour)
    colours = dark + k[:, None] * (bright - dark)
    capped = np.all((quads[:, :, 2] >= hi - 1e-12) | (quads[:, :, 2] <= lo + 1e-12), axis=1) & \
        ((z[:-1, :-1] > hi) | (z[:-1, :-1] < lo)).ravel()
    colours[capped] = to_rgb(CAP_COLOR)
    face = np.column_stack([colours, np.ones(len(colours))])

    # mesh segments, nudged a hair toward the camera so the painter's sort puts them on their facets
    step_r = max(1, int(round(MESH_RSTAR / (rstar[1] - rstar[0]))))
    rows = list(range(_first_on(rstar, MESH_RSTAR), len(rstar), step_r))
    cols = list(range(0, len(theta), MESH_THETA))
    segs = [np.stack([P[i, j], P[i, j + 1]]) for i in rows for j in range(len(theta) - 1)]
    segs += [np.stack([P[i, j], P[i + 1, j]]) for j in cols for i in range(len(rstar) - 1)]
    segs = np.array(segs)
    e, a = np.radians(ELEV), np.radians(AZIM)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]) / scale
    segs = segs + 0.02 * toward

    polys = list(quads) + list(segs)
    fc = np.vstack([face, np.zeros((len(segs), 4))])
    ec = np.vstack([face, np.tile(np.r_[to_rgb(MESH_COLOR), 1.0], (len(segs), 1))])
    lw = np.r_[np.full(len(quads), 0.15), np.full(len(segs), 0.4)]          # facets: hide the seams
    coll = Poly3DCollection(polys, facecolors=fc, edgecolors=ec, linewidths=lw, rasterized=True,
                            clip_on=False)               # (with zoom, the box is larger than the axes)
    ax.add_collection3d(coll, autolim=False)


def _first_on(rstar, step):
    """Index of the first node at or after a whole multiple of `step` (0, 5, 10, ... in r*)."""
    return int(np.argmin(np.abs(rstar - step * np.ceil(rstar[0] / step))))


def style_axes(ax, rstar, lo, hi, l, last):
    ax.set_proj_type("ortho")
    ax.patch.set_visible(False)                     # the panels overlap: no background to hide the one above
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(rstar[0], rstar[-1])
    ax.set_ylim(0, np.pi)
    ax.set_zlim(lo, hi)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#b5b4ae")
        axis.pane.set_linewidth(0.5)
        axis.line.set_linewidth(0.6)
        axis.line.set_color(INK)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 1, 2, 3])
    ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=3, symmetric=False, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="both", which="major", labelsize=7, pad=0, length=2, width=0.5)
    ax.tick_params(axis="z", pad=-3)               # mplot3d sets labels well off a wide axes; pull them in
    ax.tick_params(axis="y", pad=-6)              # the theta labels next to their (foreshortened) axis
    if not last:
        ax.set_xticklabels([])
    else:
        pass                                        # r*/M: a figure text, below the tick labels
    if not last:                                    # theta and r* labelled on the bottom panel only
        ax.set_yticklabels([])
    else:
        theta_label(ax, rstar[0], lo)


def theta_label(ax, rstar_edge, z_edge):
    """theta, upright, beside the middle of the (short, slanted) theta axis -- between its 1 and 2 -- and
    just beyond its tick labels. (mplot3d's own axis label goes off the end of so short an axis.)"""
    x, y, _ = proj3d.proj_transform(rstar_edge, THETA_LABEL_AT, z_edge, ax.get_proj())
    ax.annotate(r"$\theta$", (x, y), xycoords="data", xytext=THETA_LABEL_OFFSET, textcoords="offset points",
                ha="center", va="center", fontsize=8.5, color=INK, annotation_clip=False)


def plot(data: dict, stem=None):
    publication_style()
    rstar, theta, re_psi = data["rstar"], data["theta"], data["re_psi"]
    n = re_psi.shape[0]
    top = 0.22 if SHOW_PARAMETERS else 0.05
    base = 0.36                                     # room for the r* tick labels and label
    height = top + n * PANEL_STEP + base
    fig = plt.figure(figsize=(FIGURE_WIDTH, height))
    for l in range(n):
        bottom = base + (n - 1 - l) * PANEL_STEP     # this component's slot, PANEL_STEP tall
        middle = bottom + 0.5 * PANEL_STEP
        ax = fig.add_axes((BOX_LEFT, (middle - 0.5 * AXES_HEIGHT) / height, 1.0, AXES_HEIGHT / height),
                          projection="3d", computed_zorder=False)
        ax.set_box_aspect(BOX_ASPECT, zoom=BOX_ZOOM)
        lo, hi = Z_RANGE.get(l) or panel_range(re_psi[l], rstar, theta, data["worldtube"])
        style_axes(ax, rstar, lo, hi, l, last=l == n - 1)
        shaded_surface(ax, rstar, theta, re_psi[l], lo, hi, COMPONENT_COLOURS[l % len(COMPONENT_COLOURS)])
        fig.text(0.0, middle / height, rf"$\mathrm{{Re}}\,\psi^{{{l}}}_{{m}}$",
                 rotation=90, ha="left", va="center", fontsize=8.5, color=INK)
    fig.text(0.53, 0.02 / height, r"$r_*/M$", ha="center", va="bottom", fontsize=8.5, color=INK)
    if SHOW_PARAMETERS:
        fig.text(0.55, 1 - 0.04 / height, rf"$m = {int(data['m'])}$, $r_0 = {float(data['r0']):g}M$",
                 ha="center", va="top", fontsize=9, color=INK)
    if stem is not None:
        with plt.rc_context({"savefig.bbox": "standard"}):  # the figure as laid out: the 3D axes overhang it
            save_figure(fig, stem)
        print(f"wrote {stem}.pdf, .png")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--r0", type=float, default=R0)
    ap.add_argument("--method", default=METHOD, choices=METHODS)
    ap.add_argument("--replot", action="store_true", help="redraw from the saved .npz instead of solving")
    args = ap.parse_args()
    stem = OUTPUT_DIR / f"fields_m{args.m}_r{args.r0:g}"
    if args.replot:
        data = dict(np.load(f"{stem}.npz"))
    else:
        problem = build_problem(args.m, args.r0, settings.GRID[args.m])
        print(problem.describe())
        solution = solve(problem, args.method, settings.GMRES, neural_precision=settings.NEURAL_PRECISION,
                         neural_compile=settings.NEURAL_COMPILE)
        print(solution.summary())
        data = field_data(problem, solution)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{stem}.npz", **data)
    plot(data, stem)


if __name__ == "__main__":
    main()
