"""
settings.py -- the values used in the paper, shared by every script: the grid for each m, the GMRES
settings, the training recipe, and the machine settings (threads, network inference).

Every script imports this module FIRST, before numpy and the package:

    import settings                     # sets the thread counts; makes the package importable
    from ellipticgsf_neural import build_problem, solve
    problem = build_problem(2, 6.0, settings.GRID[2])
    solution = solve(problem, "neural", settings.GMRES, neural_compile=settings.NEURAL_COMPILE)

The package itself has no defaults for any of these: whatever calls it says which grid, which GMRES
settings and which training recipe it uses, and these are the paper's.
"""

import math
import os
import sys
from pathlib import Path

# ====================================================================== machine settings
# Thread counts, set in the environment BEFORE numpy, numba, OpenBLAS and PyTorch start their thread
# pools (which is why this module is imported first). ELLIPTICGSF_THREADS overrides the default of 8.
THREADS = int(os.environ.get("ELLIPTICGSF_THREADS", 8))       # compiled kernels (M x), BLAS and PyTorch
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(THREADS, os.cpu_count() or THREADS)))
os.environ.setdefault("OMP_WAIT_POLICY", "passive")          # keep numba's and BLAS's thread pools
os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "4")         #   from spinning against each other

# Network inference (Sec. IV B). torch.compile with freezing is ~2x faster per call, but the first call
# in a process compiles (~7 s with a warm cache), so it pays when the network runs more than once per
# process (timing.py compiles before the clock starts). It needs a 64-bit C++ compiler -- on Windows
# MSVC's cl.exe from an "x64 Native Tools Command Prompt for VS"; without one, or if compilation fails,
# the network runs uncompiled. ELLIPTICGSF_COMPILE=0 turns it off.
NEURAL_COMPILE = os.environ.get("ELLIPTICGSF_COMPILE", "1") != "0"
NEURAL_PRECISION = None          # None: bfloat16 where the CPU computes it natively, float32 elsewhere

# ====================================================================== paths
SCRIPTS = Path(__file__).resolve().parent
REPOSITORY = SCRIPTS.parent
FIGURES = REPOSITORY / "figures"                 # where the scripts write their figures and data
RUNS = REPOSITORY / "runs"                       # where train.py writes its runs
sys.path.insert(0, str(REPOSITORY))              # the package, without installing it

from ellipticgsf_neural import GMRESParameters, GridParameters  # noqa: E402  (after the thread counts)

# ====================================================================== the grids (Sec. II)
# Schwarzschild (a = 0). The radial spacing and the outer edge depend on m (about 400 radial nodes each,
# about 190,000 unknowns; r*_max = 7 k - 4 keeps a whole number of worldtube widths); the rest is shared.
# Each value is a request that Grid.worldtube snaps (see problem.GridParameters).
_RADIAL = {          # m: (Delta r*, r*_max)
    1: (1 / 3, 122.0),
    2: (7 / 25, 101.0),
    3: (7 / 29, 87.0),
    4: (7 / 31, 80.0),
    5: (7 / 33, 73.0),
    6: (7 / 37, 66.0),
    7: (7 / 39, 59.0),
}
GRID = {m: GridParameters(delta_rstar=delta_rstar, rstar_min=-4.0, rstar_max=rstar_max,
                          delta_theta=math.pi / 45,
                          worldtube_rstar_width=7.0, worldtube_theta_width=math.pi / 5)
        for m, (delta_rstar, rstar_max) in _RADIAL.items()}

# ====================================================================== GMRES (Sec. III B)
GMRES = GMRESParameters(
    restart_length=50,          # Krylov vectors per restart cycle ...
    deflation_vectors=10,       # ... of which this many "slow" ones are kept across restarts
    tolerance=1e-10,            # stop when ||b - M x|| / ||b|| <= 1e-10
    max_iterations=6000,        # cap on the total number of iterations
)

# ====================================================================== training (Sec. IV B)
# The keyword arguments of neural.TrainingParameters (train.py builds it; kept as a dict here so that
# importing the settings does not need PyTorch).
TRAINING = dict(
    training_frequencies=tuple(round(0.03 + 0.0025 * k, 6) for k in range(17)),   # M omega / m = 0.03 ... 0.07
    validation_frequencies=(6.0 ** -1.5, 0.04125, 0.05625),                       # r0 = 6, 8.376, 6.812
    loss_cycles=(1, 2, 4, 7, 11),
    learning_rate=3e-4,
    warmup_updates=8,
    gradient_clip=1.0,
    min_improvement=0.005,
    learning_rate_decay=0.5,
    patience=3,
    max_epochs=40,
    seed=0,
)
