"""
plot_style.py -- the figure style shared by the scripts: one colour, marker and line style per method, and a serif style
that matches a LaTeX manuscript (Latin Modern / Computer Modern, no external LaTeX needed).

Figures are drawn at their printed size -- COLUMN_WIDTH or PAGE_WIDTH (APS / PRD) -- with
publication_style(), so the text in them matches the paper's, and saved with save_figure() as a vector
PDF (fonts embedded) and a 300 dpi PNG.

Colours: the neural preconditioner is blue; the baselines use pink, aqua, orange, brown and violet. The
six were checked as a set (all pairs) for colour-vision deficiency and normal vision; aqua is light on
white, so every method also has its own marker and line style and the figures always carry a legend.
"""

import logging

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logging.getLogger("fontTools").setLevel(logging.ERROR)          # quiet font-embedding chatter in PDFs
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

METHOD_STYLE = {
    "direct": dict(color="#4a3aa7", marker="D", linestyle="-"),      # violet
    "none":   dict(color="#b5306e", marker="X", linestyle=":"),      # pink
    "ilu0":   dict(color="#1baf7a", marker="s", linestyle="--"),     # aqua
    "ilu1":   dict(color="#eb6834", marker="^", linestyle="-."),     # orange
    "jacobi": dict(color="#8a5a00", marker="P", linestyle=(0, (3, 1, 1, 1, 1, 1))),   # brown, dash-dot-dot
    "neural": dict(color="#2a78d6", marker="o", linestyle="-"),      # blue
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MAGNITUDE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # light -> dark
WORLDTUBE_COLOR = "#eb6834"
REFERENCE_GRAY = "#8a8984"
COLUMN_WIDTH = 3.4               # inches: one column of a two-column journal (APS / PRD)
PAGE_WIDTH = 7.0                 # inches: the full page width
MARKER_SIZE = 4.5
ASPECT = 0.5                     # height / width of a single-panel figure (timing, memory, iterations)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,          # cmr10 has no minus sign at the usual code point
        "axes.formatter.use_mathtext": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.7,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def publication_style() -> None:
    """apply_style(), with font sizes and line weights for a figure drawn at its printed size."""
    apply_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.2,
        "lines.markersize": MARKER_SIZE,
        "lines.markeredgewidth": 0.9,
        "axes.grid": False,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.handlelength": 2.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
    })


def method_handle(method: str, label: str, filled: bool = True) -> Line2D:
    """A legend entry for a method: its line with its marker (open when `filled` is False)."""
    style = METHOD_STYLE[method]
    return Line2D([], [], color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
                  markerfacecolor=style["color"] if filled else "white", label=label)


def not_converged_key(ax, top: float = 0.97) -> None:
    """A one-entry key inside the axes, at the top right with its top at `top` (axes fraction): a hollow
    circle, "not converged". Added as its own artist, so the axes can still carry a legend."""
    from matplotlib.legend import Legend
    handle = Line2D([], [], linestyle="none", marker="o", color=INK_SECONDARY, markerfacecolor="white")
    key = Legend(ax, [handle], ["not converged"], loc="upper right", bbox_to_anchor=(0.97, top),
                 bbox_transform=ax.transAxes, frameon=False, handlelength=1.0, handletextpad=0.4,
                 borderaxespad=0.0, borderpad=0.0, fontsize=7.5)          # right edge aligned with the labels
    ax.add_artist(key)


def save_figure(fig, stem) -> None:
    """Write <stem>.pdf (vector, fonts embedded) and <stem>.png (300 dpi)."""
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", dpi=300)       # dpi: only for rasterized parts (colour maps)
                                               # (f-strings, not with_suffix: "r6.5" would lose its ".5")
    fig.savefig(f"{stem}.png", dpi=300)
