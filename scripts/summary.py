"""
summary.py -- the summary figure of Sec. V A for one m: three stacked panels, from data already saved
(no solving):
    (a)  total wall time against r0          from timing_m<m>.csv           (timing.py)
    (b)  peak memory against r0               from timing_m<m>.csv           (timing.py)
    (c)  GMRES residual against iteration     from iterations_m<m>_r<r0>.npz (iterations.py)
with one legend for all three.

    python summary.py                    # the settings below
    python summary.py --m 2 --r0 6

Run timing.py and iterations.py (for the same m, and R0 for panel c) first. Writes
figures/summary_m<m>.pdf / .png beside their data. Each panel is drawn exactly as in those scripts, by
their own draw() functions, at the same width and panel height.
"""

import settings  # noqa: I001  first: sets the thread counts, makes the package importable

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import iterations
import timing
from ellipticgsf_neural import METHOD_NAMES, METHODS
from plot_style import ASPECT, method_handle, not_converged_key, publication_style, save_figure

# ------------------------------------------------------------------ SETTINGS
M = 2 # m-mode
R0 = 6.0                         # the orbit of panel (c)
MAX_ITERATIONS = None            # panel (c): cut the saved histories here to zoom in (None: as saved)
DATA_DIR = settings.FIGURES      # where timing.py and iterations.py wrote their data; the figure goes here too
FIGURE_WIDTH = 3.4               # inches: 3.4 = one column (APS / PRD)
SHOW_PARAMETERS = True           # m in (a), r0 in (c) (off: put them in the caption)
# ----------------------------------------------------------------------------

# Layout, in inches. Each panel's axes are as tall as in the single-panel figures.
PANEL = 0.77 * ASPECT * FIGURE_WIDTH   # axes height (matplotlib's default 77% of a single figure)
LEFT, RIGHT = 0.72, 0.08               # room for two-line y labels / for the last tick label
GAP_SHARED = 0.10                      # between (a) and (b), which share the r0 axis
GAP_LABELLED = 0.50                    # above (c): (b)'s r0 tick labels and axis label
BOTTOM = 0.45                          # below (c): its tick labels and axis label
TOP = 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--r0", type=float, default=R0)
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--data", default=DATA_DIR)
    args = ap.parse_args()
    folder = Path(args.data)
    rows = timing.read_csv(folder / f"timing_m{args.m}.csv")
    histories, converged, _ = iterations.load(folder / f"iterations_m{args.m}_r{args.r0:g}",
                                              iterations.METHODS, args.max_iterations or 10**9)

    publication_style()
    height = TOP + 3 * PANEL + GAP_SHARED + GAP_LABELLED + BOTTOM
    fig = plt.figure(figsize=(FIGURE_WIDTH, height))
    width = (FIGURE_WIDTH - LEFT - RIGHT) / FIGURE_WIDTH

    def axes(bottom_in, **kw):                                  # an axes rectangle given in inches
        return fig.add_axes((LEFT / FIGURE_WIDTH, bottom_in / height, width, PANEL / height), **kw)

    ax_c = axes(BOTTOM)
    ax_b = axes(BOTTOM + PANEL + GAP_LABELLED)
    ax_a = axes(BOTTOM + 2 * PANEL + GAP_LABELLED + GAP_SHARED, sharex=ax_b)

    drawn_a, failed_a = timing.draw(ax_a, rows, "total_s")
    drawn_b, _ = timing.draw(ax_b, rows, "peak_mb", headroom=1.5)    # no label to make room for
    ax_a.set_xlabel("")
    ax_a.tick_params(labelbottom=False)                        # (a) and (b) share the r0 axis
    iterations.draw(ax_c, histories, converged)
    fig.align_ylabels([ax_a, ax_b, ax_c])

    if SHOW_PARAMETERS:
        ax_a.text(0.97, 0.93, f"$m = {args.m}$", transform=ax_a.transAxes,
                  ha="right", va="top", fontsize=8.5)
        ax_c.text(0.97, 0.93, f"$r_0 = {args.r0:g}M$", transform=ax_c.transAxes, ha="right", va="top",
                  fontsize=8.5)
    if failed_a:                                                # explain the open markers, in (a) only
        not_converged_key(ax_a, top=0.80 if SHOW_PARAMETERS else 0.97)
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):  # panel labels, in the left margin
        top = ax.get_position().y1
        fig.text(0.04 / FIGURE_WIDTH, top, f"({letter})", ha="left", va="top", fontsize=9)

    shown = set(drawn_a) | set(drawn_b) | set(histories)        # one legend for all three panels
    legend = [method_handle(k, METHOD_NAMES[k]) for k in METHODS if k in shown]
    fig.legend(handles=legend, loc="upper center", ncol=2, borderaxespad=0.0,
               bbox_to_anchor=(0.5 + (LEFT - RIGHT) / 2 / FIGURE_WIDTH, (BOTTOM - 0.42) / height))

    stem = folder / f"summary_m{args.m}"
    save_figure(fig, stem)
    print(f"wrote {stem}.pdf, .png")


if __name__ == "__main__":
    main()
