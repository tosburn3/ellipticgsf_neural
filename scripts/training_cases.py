"""
training_cases.py -- the orbits the neural preconditioners were trained on (Sec. IV B), one cross per (m, omega/m)
training case, with the orbits of the timing figures marked (large dashed hollow markers, one shape and
colour per figure), the four problems of the generalization figure (solid markers, one per panel), and r0
on the top axis.

    python training_cases.py

Writes figures/training_cases.pdf / .png. Needs no solves: the training orbits are those of
settings.TRAINING, the modes those of settings.GRID, and the timing orbits timing.py's R0_VALUES
(omega/m = r0^(-3/2)), the generalization problems and their solid markers generalization.py's PANELS (which
also draws each marker in its panel). Edit MARKED, and PANELS there, to change the markers, and MARKED and
GENERALIZATION_FIGURE for the figure numbers the legend refers to.

Colours: purple circles and green diamonds are clearly distinct from each other (also for colour-blind
readers), and from the method that shares each shape elsewhere (neural: blue circles; direct: violet
diamonds). The solid markers (blue, red, yellow, aqua) pass the dataviz normal-vision check (Delta E >= 15)
against each other and the dashed ones; they also differ in shape, which carries them for colour-blind
readers. They are drawn under the training crosses, which stay visible on top. (b)-(d) sit at r0 = 7.5M
(Momega/m = 0.0487), midway between the training columns at 0.0475 and 0.05.
"""

import settings  # noqa: I001  first: sets the thread counts, makes the package importable

import matplotlib.pyplot as plt
import numpy as np

from plot_style import ASPECT, INK, publication_style, save_figure
from generalization import PANELS, case_marker
from timing import R0_VALUES

TRAINING_FREQUENCIES = settings.TRAINING["training_frequencies"]       # M omega / m of the training orbits

# ------------------------------------------------------------------ SETTINGS
MARKED = [                          # (mode, marker, colour, figure the legend refers to): timing orbits at m
    (2, "o", "#9b3fbf", "Fig. 7"),  # purple circles
    (6, "D", "#008300", "Fig. 8"),  # green diamonds
]
GENERALIZATION_FIGURE = "Fig. 9"    # the generalization figure (one solid marker per panel, from its PANELS)
R0_LEFT = 11.5                      # the horizontal axis starts at this r0/M (no training data beyond 10.4)
M_TOP = 8.5                         # the vertical axis runs up to this m (training data up to 7)
OUTPUT = settings.FIGURES / "training_cases"   # (writes .pdf and .png)
FIGURE_WIDTH = 3.4                  # inches: 3.4 = one column (APS / PRD); 7.0 = full page width
# ----------------------------------------------------------------------------

TRAINING = dict(marker="x", markersize=4.5, markeredgewidth=0.9, color=INK, linestyle="none",
                label="training case")


def marked_style(marker, colour):
    """Large hollow markers with dashed outlines (a scatter, since plain markers cannot be dashed)."""
    size = 9 if marker == "o" else 8                # points across; diamonds look larger at equal size
    return dict(marker=marker, s=size ** 2, facecolors="none", edgecolors=colour, linewidths=1.0,
                linestyles=(0, (2.2, 1.4)))


def main():
    publication_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, ASPECT * FIGURE_WIDTH))
    modes = sorted(settings.GRID)
    w, m = np.meshgrid(TRAINING_FREQUENCIES, modes)
    ax.plot(w.ravel(), m.ravel(), clip_on=False, zorder=3, **TRAINING)

    extrapolated = []                                                # their own legend row
    for letter, spec in PANELS.items():                              # about twice the size of the crosses
        extrapolated += ax.plot(spec["r0"] ** -1.5, spec["m"], clip_on=False, zorder=2.5,
                                label=f"{GENERALIZATION_FIGURE}({letter})", **case_marker(letter))

    w_timing = np.asarray(R0_VALUES) ** -1.5                         # omega/m of each timing orbit
    for mode, marker, colour, figure in MARKED:
        ax.scatter(w_timing, np.full(w_timing.size, mode), clip_on=False, zorder=4,
                   label=f"{figure} ($m={mode}$)", **marked_style(marker, colour))

    ax.set_xlim(R0_LEFT ** -1.5, max(max(TRAINING_FREQUENCIES), w_timing.max()) + 0.002)
    ax.set_ylim(0.5, M_TOP)
    ax.set_yticks(range(1, int(M_TOP) + 1))
    ax.set_xlabel(r"$M\omega/m$")
    ax.set_ylabel(r"$m$")

    tiny = 1e-12                                    # matplotlib also probes these at 0
    top = ax.secondary_xaxis("top", functions=(lambda x: np.maximum(np.abs(x), tiny) ** (-2.0 / 3.0),  # w/m -> r0
                                               lambda r: np.maximum(np.abs(r), tiny) ** (-1.5)))       # r0 -> w/m
    r0_ticks = list(range(6, int(R0_LEFT) + 1))
    top.set_xticks(r0_ticks)
    top.set_xticklabels([str(r) for r in r0_ticks])
    top.set_xlabel(r"$r_0/M$")
    top.tick_params(which="minor", top=False)
    ax.tick_params(top=False)                       # the top edge carries the r0 axis instead

    handles, labels = ax.get_legend_handles_labels()
    first = [(h, l) for h, l in zip(handles, labels) if h not in extrapolated]
    box = ax.get_position()                         # centre the legend on the figure, not the axes (which
    centre = (0.5 - box.x0) / box.width             #   sit right of centre, past the m label)
    row1 = ax.legend(*zip(*first), loc="upper center", bbox_to_anchor=(centre, -0.33), ncol=len(first),
                     borderaxespad=0.0, columnspacing=0.9, handletextpad=0.35, handlelength=1.1)
    ax.add_artist(row1)                             # a second legend() call would replace it
    ax.legend(handles=extrapolated, loc="upper center", bbox_to_anchor=(centre, -0.47), ncol=len(extrapolated),
              borderaxespad=0.0, columnspacing=0.9, handletextpad=0.3, handlelength=1.0)
    out = OUTPUT
    save_figure(fig, out)
    print(f"wrote {out}.pdf, .png  ({w.size} training cases; timing orbits marked at m = "
          f"{', '.join(str(mode) for mode, *_ in MARKED)})")


if __name__ == "__main__":
    main()
