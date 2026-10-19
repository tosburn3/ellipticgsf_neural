"""
timing.py -- total wall time and peak memory against r0 for one m (Sec. V A): direct (SuperLU) and GMRES
with the ILU(0), ILU(1) and neural preconditioners. (GMRES without a preconditioner never converges on
these problems -- it only burns MAX_ITERATIONS iterations -- so it is left out; --methods none adds it back.)

    python timing.py                     # the settings below
    python timing.py --m 6 --methods direct ilu0 neural
    python timing.py --replot            # redraw from the saved .csv, without timing again

Writes figures/timing_m<m>.pdf / .png, figures/memory_m<m>.pdf / .png and the numbers as
figures/timing_m<m>.csv.

Every (r0, method) runs in a fresh Python process, so no measurement inherits memory or warm caches
from another. In that process:
  1. M and b are built (identical for every method, and left out of both measurements);
  2. the method runs once for 2 iterations, untimed, to compile numba's kernels and the network;
  3. the method runs: setup + solve.
  total wall time   setup + solve: the LU, or the strip LU + Schur complement + its factorization (+ the
                    network), and the iterations.
  peak memory       the largest resident memory during setup + solve minus the resident memory just
                    before it (sampled every 2 ms with psutil): what the method needs on top of M and b.
A solve that does not converge within MAX_ITERATIONS is drawn with an open marker. The figures are drawn at
their printed size (FIGURE_WIDTH, one journal column by default) in the style of iterations.py.
"""

import settings  # noqa: I001  first: sets the thread counts, makes the package importable

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from ellipticgsf_neural import METHOD_NAMES, METHODS

# ------------------------------------------------------------------ SETTINGS
M = 2 # m-mode
R0_VALUES = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
METHODS_TO_TIME = ["direct", "ilu0", "ilu1", "neural"]
MAX_ITERATIONS = 20000           # GMRES iterations at most, per run (settings.GMRES.max_iterations: 6000)
OUTPUT_DIR = settings.FIGURES     # where the figures and the .csv go
FIGURE_WIDTH = 3.4               # inches: 3.4 = one column (APS / PRD); 7.0 = full page width
SHOW_PARAMETERS = True           # write "m = ..." inside the axes (off: put it in the caption)
# ----------------------------------------------------------------------------


# ====================================================================== one measurement (worker process)

class PeakMemory:
    """Peak resident memory (MB) above the level at entry, sampled in a background thread."""

    def __init__(self, interval=0.002):
        import psutil
        self.interval, self.process = interval, psutil.Process()

    def __enter__(self):
        gc.collect()
        if sys.platform.startswith("linux"):           # hand freed heap back to the OS before the baseline
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        self.base = self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.mb = (self.peak - self.base) / 2**20


def measure(m: int, r0: float, method: str, max_iterations: int) -> dict:
    from ellipticgsf_neural import build_problem, solve
    problem = build_problem(m, r0, settings.GRID[m])
    options = dict(neural_precision=settings.NEURAL_PRECISION, neural_compile=settings.NEURAL_COMPILE)
    if method != "direct":       # warm-up, untimed: numba's kernels, and torch.compile of the network for
        solve(problem, method, settings.GMRES, max_iterations=2, **options)   # this graph size (static shapes)
    with PeakMemory() as mem:
        s = solve(problem, method, settings.GMRES, max_iterations=max_iterations, **options)   # (direct: ignored)
    print(f"  {s.summary()}, peak +{mem.mb:,.0f} MB", flush=True)
    import numba
    return dict(m=m, r0=r0, method=method, total_s=s.total_time, setup_s=s.setup_time, solve_s=s.solve_time,
                peak_mb=mem.mb, iterations=s.iterations or 0, converged=s.converged, residual=s.relative_residual,
                threads=numba.get_num_threads(), max_iterations=max_iterations)


# ====================================================================== the study

def run(m: int, r0: float, method: str, max_iterations: int) -> dict:
    """measure() in a fresh Python process."""
    proc = subprocess.run([sys.executable, __file__, "--worker", str(m), repr(r0), method, str(max_iterations)],
                          stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    lines = proc.stdout.splitlines()
    for line in lines:
        if not line.startswith("RESULT "):
            print(line)
    result = [line for line in lines if line.startswith("RESULT ")]
    if proc.returncode or not result:
        raise RuntimeError(f"the measurement of {method} at r0 = {r0} failed (exit code {proc.returncode})")
    return json.loads(result[-1][7:])


YLABELS = {"total_s": "total wall time (seconds)", "peak_mb": "peak memory (MB)"}


def draw(ax, rows, key, headroom=2.2):
    """One quantity (key: "total_s" or "peak_mb") against r0 on the axes `ax`, one curve per method; open
    markers: not converged. Returns the methods drawn and whether any point did not converge."""
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
    from plot_style import METHOD_STYLE
    methods = [k for k in METHODS if any(r["method"] == k for r in rows)]
    any_failed = False
    for k, method in enumerate(methods):
        mine = sorted((r for r in rows if r["method"] == method), key=lambda r: r["r0"])
        r0s = np.array([r["r0"] for r in mine])
        ys = np.array([r[key] for r in mine])
        ok = np.array([r["converged"] for r in mine])
        style = METHOD_STYLE[method]
        ax.plot(r0s, ys, color=style["color"], linestyle=style["linestyle"], zorder=2 + k)
        ax.plot(r0s[ok], ys[ok], linestyle="none", marker=style["marker"], color=style["color"],
                clip_on=False, zorder=10 + k)
        ax.plot(r0s[~ok], ys[~ok], linestyle="none", marker=style["marker"], color=style["color"],
                markerfacecolor="white", clip_on=False, zorder=10 + k)
        any_failed |= not ok.all()

    ax.set_yscale("log")
    low, high = min(r[key] for r in rows), max(r[key] for r in rows)
    bottom, top = low / 1.4, high * headroom            # headroom: room for a label at the top right
    ax.set_ylim(bottom, top)
    narrow = np.log10(top / bottom) < 1.2               # less than ~1.2 decades: label more ticks
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0) if narrow else (1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    r0_all = sorted({r["r0"] for r in rows})
    ax.set_xlim(r0_all[0] - 0.2, r0_all[-1] + 0.2)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel(r"$r_0/M$")
    ax.set_ylabel(YLABELS[key])
    return methods, any_failed


def plot(rows, m, out):
    """timing_m<m> and memory_m<m> (.pdf, .png): one curve per method against r0; open markers: not converged."""
    import matplotlib.pyplot as plt
    from plot_style import ASPECT, method_handle, not_converged_key, publication_style, save_figure
    publication_style()
    for key, stem in (("total_s", "timing"), ("peak_mb", "memory")):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, ASPECT * FIGURE_WIDTH))
        methods, any_failed = draw(ax, rows, key)
        legend = [method_handle(k, METHOD_NAMES[k]) for k in methods]
        if SHOW_PARAMETERS:
            ax.text(0.97, 0.95, f"$m = {m}$", transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5)
        if key == "total_s" and any_failed:                # explain the open markers, in the wall-time plot only
            not_converged_key(ax, top=0.82 if SHOW_PARAMETERS else 0.97)
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, borderaxespad=0.0)
        save_figure(fig, out / f"{stem}_m{m}")
        plt.close(fig)


def read_csv(path) -> list:
    """The rows timing.py wrote, with their types back."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("total_s", "setup_s", "solve_s", "peak_mb", "residual", "r0"):
            r[k] = float(r[k])
        for k in ("m", "iterations"):
            r[k] = int(r[k])
        r["converged"] = r["converged"] == "True"
        r["threads"] = int(r.get("threads") or settings.THREADS)    # older files have neither column
        r["max_iterations"] = int(r.get("max_iterations") or settings.GMRES.max_iterations)
    return rows


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--worker":
        m, r0, method, max_iterations = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
        print("RESULT " + json.dumps(measure(m, r0, method, max_iterations)), flush=True)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--r0", type=float, nargs="+", default=R0_VALUES)
    ap.add_argument("--methods", nargs="+", default=METHODS_TO_TIME,
                    choices=[k for k in METHODS if k != "jacobi"])     # Jacobi: iterations.py only
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--replot", action="store_true", help="redraw from the saved .csv instead of timing")
    args = ap.parse_args()
    out = Path(args.out)
    if args.replot:
        rows = [r for r in read_csv(out / f"timing_m{args.m}.csv") if r["method"] in args.methods]
        plot(rows, args.m, out)
        print(f"wrote {out}/timing_m{args.m}.pdf, memory_m{args.m}.pdf (+ .png)")
        return
    if "neural" in args.methods:
        from ellipticgsf_neural import neural  # noqa: F401  fail now if PyTorch is missing, not after the other solves
    out.mkdir(parents=True, exist_ok=True)

    from ellipticgsf_neural.problem import Orbit, make_grid
    rows = []
    for r0 in args.r0:
        g = make_grid(Orbit(0.0, r0), settings.GRID[args.m])
        print(f"m = {args.m}, r0 = {r0:g}: {g.n_rstar} x {g.n_theta} nodes, {10 * g.n_rstar * g.n_theta:,} unknowns",
              flush=True)
        for method in args.methods:
            rows.append(run(args.m, r0, method, args.max_iterations))
        with open(out / f"timing_m{args.m}.csv", "w", newline="") as f:      # rewritten after every r0
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    plot(rows, args.m, out)
    print(f"wrote {out}/timing_m{args.m}.pdf, memory_m{args.m}.pdf (+ .png) and timing_m{args.m}.csv")


if __name__ == "__main__":
    main()
