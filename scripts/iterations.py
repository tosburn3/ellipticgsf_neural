"""
iterations.py -- GMRES relative residual against iteration number for one m and r0 (Sec. V A), with no
preconditioner, ILU(0), ILU(1), Jacobi and the neural preconditioner. (Jacobi -- exact boundary strips and
the diagonal of the Schur complement M_S, App. A -- is here for comparison; it does not converge.)

    python iterations.py                 # the settings below
    python iterations.py --m 6 --r0 6.5
    python iterations.py --replot        # redraw from the saved .npz, without solving again
    python iterations.py --replot --max-iterations 2000

Writes figures/iterations_m<m>_r<r0>.pdf / .png and the residual histories as .npz. Every run stops at
MAX_ITERATIONS. The unpreconditioned and Jacobi runs do not converge on these problems; they run last
and are stopped at the iteration count of the slowest of ILU(0), ILU(1) and neural (CAPPED_MAX_ITERATIONS
overrides that). With --replot,
MAX_ITERATIONS cuts the saved histories, so a smaller value zooms in without solving again.

The figure is drawn at its printed size (FIGURE_WIDTH, one journal column by default), with the
colours, line styles and markers of timing.py: one marker at the end of each curve, open if
the run did not converge. Iteration 0 is the initial guess x = 0 (residual 1).
"""

import settings  # noqa: I001  first: sets the thread counts, makes the package importable

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

from ellipticgsf_neural import METHOD_NAMES, build_problem, solve
from plot_style import (ASPECT, INK_SECONDARY, MARKER_SIZE, METHOD_STYLE, REFERENCE_GRAY, method_handle,
                        publication_style, save_figure)

# ------------------------------------------------------------------ SETTINGS
M = 2 # m-mode
R0 = 6.0
METHODS = ("none", "ilu0", "ilu1", "jacobi", "neural")
MAX_ITERATIONS = 3000            # GMRES iterations at most, per run (settings.GMRES.max_iterations: 6000)
CAPPED_MAX_ITERATIONS = None          # None: stop "none" and "jacobi" at the slowest of ilu0 / ilu1 / neural
OUTPUT_DIR = settings.FIGURES     # where the figure and its data go
FIGURE_WIDTH = 3.4               # inches: 3.4 = one column (APS / PRD); 7.0 = full page width
SHOW_PARAMETERS = True           # write "m = ..., r0 = ..." inside the axes (off: put it in the caption)
# ----------------------------------------------------------------------------


def label(method: str, iterations: int, converged: bool, max_iterations: int) -> str:
    if converged:
        return f"{METHOD_NAMES[method]} ({iterations} its)"
    return f"{METHOD_NAMES[method]} ({'not converged' if iterations >= max_iterations else 'stopped'})"


def draw(ax, histories: dict, converged: dict) -> None:
    """The residual histories on the axes `ax`: each curve from iteration 0 (residual 1), one marker at its
    end (open: not converged), and the tolerance."""
    xmax = max(len(h) for h in histories.values())
    for k, (method, h) in enumerate(histories.items()):
        style = METHOD_STYLE[method]
        x = np.arange(len(h) + 1)
        y = np.concatenate(([1.0], h))
        ax.semilogy(x, y, color=style["color"], linestyle=style["linestyle"], zorder=2 + k)
        ax.plot(x[-1], y[-1], linestyle="none", marker=style["marker"], markersize=MARKER_SIZE,
                color=style["color"], markerfacecolor=style["color"] if converged[method] else "white",
                clip_on=False, zorder=20 - k)                     # open: not converged, as in timing.py

    ax.axhline(settings.GMRES.tolerance, color=REFERENCE_GRAY, linewidth=0.6, zorder=1)
    ax.annotate("tolerance", (0.0, settings.GMRES.tolerance), xycoords=("axes fraction", "data"), xytext=(3, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xlim(0, 1.02 * xmax)
    ax.set_ylim(settings.GMRES.tolerance / 4, 2.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=100.0, numticks=12))           # 10^0, 10^-2, ...
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=24))   # every decade
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("GMRES iteration")
    ax.set_ylabel("relative residual\n" r"$\|\mathbf{b} - \mathbf{M}\mathbf{x}_k\| \,/\, \|\mathbf{b}\|$")


def plot(histories: dict, converged: dict, max_iterations: int, m: int, r0: float, stem: Path) -> None:
    """histories[method]: relative residual after iterations 1, 2, ...; drawn from iteration 0 (residual 1)."""
    publication_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, ASPECT * FIGURE_WIDTH))
    draw(ax, histories, converged)
    legend = [method_handle(mt, label(mt, len(h), converged[mt], max_iterations), converged[mt])
              for mt, h in histories.items()]
    if SHOW_PARAMETERS:
        ax.text(0.97, 0.95, f"$m = {m}$, $r_0 = {r0:g}M$", transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5)
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, borderaxespad=0.0)
    save_figure(fig, stem)
    plt.close(fig)


def load(stem, methods, max_iterations: int):
    """The histories saved in <stem>.npz for `methods`, cut at max_iterations (a history longer than that
    counts as not converged). Returns (histories, converged, the cap used)."""
    saved = np.load(f"{stem}.npz")
    histories = {m: saved[f"history_{m}"] for m in methods if f"history_{m}" in saved}
    converged = {m: bool(saved[f"converged_{m}"]) if f"converged_{m}" in saved     # older files: infer
                 else bool(h[-1] <= saved["rtol"]) for m, h in histories.items()}
    cap = min(max_iterations, int(saved["max_iterations"]) if "max_iterations" in saved
              else settings.GMRES.max_iterations)                     # cannot extend what was not run
    for m, h in histories.items():
        if len(h) > cap:                                              # cut: not converged within the cap
            histories[m], converged[m] = h[:cap], False
    return histories, converged, cap


def run(problem, methods, max_iterations: int, network=None) -> dict:
    """Solve `problem` by each method; returns {method: Solution} in the order of `methods`. "none" and
    "jacobi" (which do not converge) run last and stop at the slowest of the rest (CAPPED_MAX_ITERATIONS
    overrides that). network: the network for "neural" (default: the shipped one for problem.m)."""
    if "neural" in methods:
        from ellipticgsf_neural import neural  # noqa: F401  fail now if PyTorch is missing, not after the other solves
    solutions = {}
    capped = ("none", "jacobi")
    for method in [m for m in methods if m not in capped] + [m for m in methods if m in capped]:
        maxiter = max_iterations
        if method in capped:
            done = [s.iterations for k, s in solutions.items() if k not in capped]
            maxiter = min(CAPPED_MAX_ITERATIONS or (max(done) if done else maxiter), maxiter)
        solutions[method] = solve(problem, method, settings.GMRES, max_iterations=maxiter, network=network,
                                  neural_precision=settings.NEURAL_PRECISION, neural_compile=settings.NEURAL_COMPILE)
        print(solutions[method].summary())
    return {m: solutions[m] for m in methods}


def save(stem, solutions: dict, max_iterations: int, **extra) -> None:
    """The residual histories as <stem>.npz, in the form load() reads (extra: anything else to keep)."""
    np.savez(f"{stem}.npz", **{f"history_{m}": s.history for m, s in solutions.items()},
             **{f"converged_{m}": s.converged for m, s in solutions.items()}, rtol=settings.GMRES.tolerance,
             max_iterations=max_iterations, **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--r0", type=float, default=R0)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=["none", "ilu0", "ilu1", "jacobi", "neural"])
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--replot", action="store_true", help="redraw from the saved .npz instead of solving")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"iterations_m{args.m}_r{args.r0:g}"

    if args.replot:
        histories, converged, cap = load(stem, args.methods, args.max_iterations)
        plot(histories, converged, cap, args.m, args.r0, stem)
        print(f"wrote {stem}.pdf, .png")
        return

    problem = build_problem(args.m, args.r0, settings.GRID[args.m])
    print(problem.describe())
    solutions = run(problem, args.methods, args.max_iterations)
    histories = {m: s.history for m, s in solutions.items()}
    converged = {m: s.converged for m, s in solutions.items()}
    plot(histories, converged, args.max_iterations, problem.m, problem.orbit.r0, stem)
    save(stem, solutions, args.max_iterations)
    print(f"wrote {stem}.pdf, .png, .npz")


if __name__ == "__main__":
    main()
