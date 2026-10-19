"""
generalization.py -- GMRES relative residual against iteration on four problems outside the training
set (Sec. V B), each panel varying one thing:
    (a)  frequency:      m = 4, r0 = 11M  (Momega/m = 0.027; the training orbits stop at r0 = 10.4M)
    (b)  mode:           m = 8, r0 = 7.5M, with the network trained for m = 7 (no network is trained for m = 8)
    (c)  resolution:     m = 5, r0 = 7.5M, Delta r* and Delta theta both ~1/sqrt(2) of training: 2.0x the unknowns
    (d)  source vector:  m = 3, r0 = 7.5M, b of random numbers instead of the effective source
(r0 = 7.5M, Momega/m = 0.0487, lies midway between two training frequencies, 0.0475 and 0.05.)

    python generalization.py                  # solve all four panels and plot
    python generalization.py --panels b       # solve (b) again; the other panels come from their saved .npz
    python generalization.py --replot         # redraw from the saved .npz, without solving

Writes figures/generalization.pdf / .png, and each panel's residual histories as generalization_<letter>.npz (in
the form iterations.py reads). The methods, end markers, caps and colours are those of iterations.py:
no preconditioner and Jacobi (which do not converge) stop at the iteration count of the slowest of
ILU(0), ILU(1) and neural; an open end marker means not converged. One legend serves all four panels.
The solid marker in the upper right of each panel is the one that marks the case in training_cases.pdf
(set in PANELS).

Grids. "Training resolution" is the grid of settings.GRID for the mode. m = 8 has none, so (b) uses the
m = 7 grid (the one its network was trained on): only m changes. In (c) the requested spacings go from
Delta r* = 7/33, Delta theta = pi/45 to 7/47, pi/65 (14 instead of 10 nodes across the worldtube each way), so
that the worldtube still holds whole numbers of cells.

Random b. Entry k is a complex Gaussian number times the norm of row k of M, i.e. a standard random
right-hand side for the row-equilibrated system. The row norms of M span 1 (boundary rows) to ~4e5, and
an unscaled random b puts O(1) values where the operator is tiny: the solution is then ~1000 |b| and
not even the direct solve (SuperLU, double precision) gets the relative residual below ~3e-9, above the
1e-10 tolerance. Scaled, SuperLU reaches 3e-11. RANDOM_B_ROW_SCALED = False gives the unscaled b.
"""

import settings  # noqa: I001  first: sets the thread counts, makes the package importable

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import iterations
from ellipticgsf_neural import METHOD_NAMES, N_COMPONENTS, build_problem
from ellipticgsf_neural.problem import make_grid
from plot_style import method_handle, publication_style, save_figure

# ------------------------------------------------------------------ SETTINGS
PANELS = {   # letter: the problem, the title shown above the panel (r0 = 7.5M in (b)-(d): in the caption),
             # and the solid marker that stands for the case here (upper right) and in training_cases.py
    "a": dict(m=4, r0=11.0, marker="v", colour="#2a78d6",                       # blue triangle
              title=r"(a) extrapolated frequency: $r_0 = 11M$, $m = 4$"),
    "b": dict(m=8, r0=7.5, network=7, grid=settings.GRID[7], marker="p", colour="#e34948",   # red pentagon
              title=r"(b) extrapolated mode: $m = 8$ ($m = 7$ network)"),
    "c": dict(m=5, r0=7.5, grid=replace(settings.GRID[5], delta_rstar=7 / 47, delta_theta=math.pi / 65),
              marker="h", colour="#c98500",                                     # yellow hexagon
              title=r"(c) extrapolated resolution: {ratio:.1f}$\times$ unknowns, $m = 5$"),
    "d": dict(m=3, r0=7.5, random_b=True, marker="*", colour="#1baf7a",         # aqua star
              title=r"(d) extrapolated source: random $\mathbf{{b}}$, $m = 3$"),
}
METHODS = ("none", "ilu0", "ilu1", "jacobi", "neural")
MAX_ITERATIONS = 5000            # GMRES iterations at most, per run
SEED = 2026                      # for the random b of (d)
RANDOM_B_ROW_SCALED = True       # (d): scale entry k by the norm of row k of M (see above)
OUTPUT_DIR = settings.FIGURES     # where the figure and its data go
FIGURE_WIDTH = 7.0               # inches: 7.0 = full page width (APS / PRD)
SHOW_PARAMETERS = True           # the problem in each panel's title (off: just (a)...(d); say it in the caption)
# ----------------------------------------------------------------------------

# Layout, in inches. Each panel's axes are as tall as in iterations.py's figure.
PANEL_H = 0.77 * 0.5 * 3.4             # axes height
LEFT, RIGHT, MIDDLE = 0.72, 0.08, 0.22  # room for the two-line y label; the columns share the y axis
TOP, GAP, BOTTOM = 0.24, 0.72, 0.72     # titles; (c)/(d)'s titles + (a)/(b)'s x label; x label + legend


def case_marker(letter: str) -> dict:
    """Plot keywords for the solid marker of panel `letter` (the same in training_cases.py)."""
    marker = PANELS[letter]["marker"]
    size = {"*": 12, "v": 8}.get(marker, 9)          # points across; equal visual weight across shapes
    return dict(marker=marker, markersize=size, color=PANELS[letter]["colour"], markeredgecolor="white",
                markeredgewidth=0.4, linestyle="none")


def problem_for(spec: dict):
    """The linear system of one panel."""
    problem = build_problem(spec["m"], spec["r0"], spec.get("grid") or settings.GRID[spec["m"]])
    if spec.get("random_b"):
        rng = np.random.default_rng(SEED)
        b = rng.standard_normal(problem.n) + 1j * rng.standard_normal(problem.n)
        if RANDOM_B_ROW_SCALED:
            b *= np.sqrt(np.asarray(abs(problem.M).power(2).sum(axis=1)).ravel())
        problem = replace(problem, b=b)
    return problem


def solve_panel(letter: str, methods, max_iterations: int, stem: Path) -> None:
    spec = PANELS[letter]
    problem = problem_for(spec)
    print(f"({letter}) {problem.describe()}")
    network = None
    if "network" in spec:
        from ellipticgsf_neural import neural
        network = neural.shipped_network_inference(spec["network"], settings.NEURAL_PRECISION, settings.NEURAL_COMPILE)
    solutions = iterations.run(problem, methods, max_iterations, network=network)
    g = problem.grid
    training = (make_grid(problem.orbit, settings.GRID[problem.m]).size(N_COMPONENTS) if problem.m in settings.GRID
                else problem.n)
    iterations.save(stem, solutions, max_iterations, m=problem.m, r0=problem.orbit.r0,
                    network=spec.get("network", problem.m), unknowns=problem.n, training_unknowns=training,
                    n_rstar=g.n_rstar, n_theta=g.n_theta, delta_rstar=g.delta_rstar, delta_theta=g.delta_theta,
                    random_b=bool(spec.get("random_b")),
                    random_b_row_scaled=bool(spec.get("random_b")) and RANDOM_B_ROW_SCALED)


def plot(folder: Path, methods, max_iterations: int) -> Path:
    publication_style()
    height = TOP + 2 * PANEL_H + GAP + BOTTOM
    fig = plt.figure(figsize=(FIGURE_WIDTH, height))
    w = (FIGURE_WIDTH - LEFT - RIGHT - MIDDLE) / 2

    def axes(col, row, **kw):                         # row 0 at the top
        bottom = BOTTOM + (1 - row) * (PANEL_H + GAP)
        return fig.add_axes(((LEFT + col * (w + MIDDLE)) / FIGURE_WIDTH, bottom / height,
                             w / FIGURE_WIDTH, PANEL_H / height), **kw)

    first, converged_any, shown = None, set(), set()
    for k, letter in enumerate(PANELS):
        stem = folder / f"generalization_{letter}"
        histories, converged, cap = iterations.load(stem, methods, max_iterations)
        saved = np.load(f"{stem}.npz")
        ax = axes(k % 2, k // 2, sharey=first)
        first = first or ax
        iterations.draw(ax, histories, converged)
        if k % 2:                                     # the right column shares the left one's y axis
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        title = PANELS[letter]["title"] if SHOW_PARAMETERS else f"({letter})"
        title = title.format(ratio=int(saved["unknowns"]) / int(saved["training_unknowns"]))   # (c)
        ax.set_title(title, loc="left", fontsize=8.5, pad=4)
        ax.plot(0.955, 0.88, transform=ax.transAxes, clip_on=False, zorder=30,   # as in training_cases.pdf
                **case_marker(letter))
        shown |= set(histories)
        converged_any |= {mt for mt, c in converged.items() if c}

    legend = [method_handle(mt, METHOD_NAMES[mt], mt in converged_any) for mt in methods if mt in shown]
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), borderaxespad=0.0, columnspacing=1.4,
               bbox_to_anchor=((LEFT + (FIGURE_WIDTH - LEFT - RIGHT) / 2) / FIGURE_WIDTH, 0.04 / height))

    stem = folder / "generalization"
    save_figure(fig, stem)
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", nargs="+", default=list(PANELS), choices=list(PANELS),
                    help="panels to solve (the rest are read from their saved .npz)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=["none", "ilu0", "ilu1", "jacobi", "neural"])
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--replot", action="store_true", help="redraw from the saved .npz instead of solving")
    args = ap.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    if not args.replot:
        for letter in args.panels:
            solve_panel(letter, args.methods, args.max_iterations, folder / f"generalization_{letter}")
    stem = plot(folder, args.methods, args.max_iterations)
    print(f"wrote {stem}.pdf, .png")


if __name__ == "__main__":
    main()
