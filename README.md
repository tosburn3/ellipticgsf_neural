# ellipticgsf_neural

Neural preconditioners for the elliptic m-mode equations of the gravitational self-force: the code
behind *Machine learning and high accuracy: Neural preconditioners for computational gravity*.

The ten tensor components of the metric perturbation of a circular orbit around a Schwarzschild black
hole satisfy, mode by mode, a coupled elliptic system in (r\*, θ) (the effective-source formulation of
arXiv:2608.07934). It is discretized by finite differences into a sparse, complex, non-Hermitian
system **M x = b**. That system is solved with GMRES with deflated restarts, preconditioned by exact
solves on the radial boundary strips and a factorization of the Schur complement **M**_S on the bulk.
The factorization is either textbook ILU(0) / ILU(1) or **P**_neural = (**L**₀ + δ**L**)(**U**₀ + δ**U**),
whose corrections are predicted by a graph neural network that reads **M**_S as a weighted adjacency
matrix. The network is trained by differentiating through GMRES restart cycles.

## Layout

```
ellipticgsf_neural/          the package
    problem.py               the orbit, the (r*, θ) grid, build_problem: M and b          Secs. II, III
    assemble.py              the sparse matrix M: coefficients, finite differences,       Sec. II C, App. A
                             boundary conditions (and the boundary-strip widths)
    source.py                the source vector b: puncture field, effective source,       Sec. II
                             worldtube jumps; the binding to the C library
    iterative.py             GMRES with deflated restarts; the Schur-complement           Sec. III, App. A
                             preconditioner; ILU(0) and ILU(1)
    neural.py                the GNN, P_neural, inference, the GMRES-cycle loss and       Sec. IV
                             its gradient, training
    solve.py                 solve(problem, method): direct, none, ilu0, ilu1, jacobi, neural
    models/                  the trained networks, gnn_m1.pt ... gnn_m7.pt
    effective_source/        Barry Wardell's effective-source C code (build it, see below)
scripts/
    settings.py              the values used in the paper: grids, GMRES, training; threads   <- edit here
    plot_style.py            the shared figure style
    run_and_plot.py          solve one mode and plot the ten field components             Sec. II (fields)
    iterations.py            residual against GMRES iteration, every method               Sec. V A
    timing.py                wall time and peak memory against r0                         Sec. V A
    summary.py               (a) wall time, (b) memory, (c) residual, from saved data      Sec. V A
    training_cases.py        the training orbits, with the timing and generalization cases Sec. IV B
    generalization.py        residual against iteration on four extrapolated problems      Sec. V B
    train.py                 train the network for one m                                   Sec. IV B
tests/                       python -m tests.run  (or pytest tests)
figures/                     where the scripts write figures and their data
```

The package has no settings of its own: every function takes the grid (`GridParameters`), the GMRES
settings (`GMRESParameters`) and the training recipe (`TrainingParameters`) it uses as arguments, and the
paper's values are in `scripts/settings.py`. Each script also has a SETTINGS block at the top for its
figure (which m, which r0, figure size, ...) and the same options on the command line (`--help`).

## Quick start

```bash
cd scripts
python run_and_plot.py                      # m = 2, r0 = 6M, neural: solve and plot the field
python iterations.py --m 2 --r0 6           # residual histories of every method
python iterations.py --replot               # redraw from the saved data in seconds
```

From Python (run from `scripts/`, or put `scripts/` on the path, so that `settings` is importable):

```python
import settings                             # first: thread counts; makes the package importable
from ellipticgsf_neural import build_problem, solve

problem = build_problem(m=2, r0=6.0, grid_parameters=settings.GRID[2])
for method in ("direct", "ilu0", "ilu1", "neural"):
    print(solve(problem, method, settings.GMRES).summary())
```

## Requirements

Python ≥ 3.10, numpy, scipy, numba, torch (for the neural preconditioner and training), matplotlib
(figures), psutil (memory in `timing.py`), threadpoolctl (optional; keeps SuperLU's BLAS
single-threaded). `pip install -e .` installs the package; the scripts also work without installing it.

Environment variables:

| variable | default | effect |
|---|---|---|
| `ELLIPTICGSF_THREADS` | 8 | threads for the compiled kernels, BLAS and PyTorch (set by `scripts/settings.py`) |
| `ELLIPTICGSF_COMPILE` | 1 | `0`: run the network without `torch.compile` |
| `ELLIPTICGSF_EFFSOURCE_LIB` | | path of the compiled effective-source library, if it is not in `ellipticgsf_neural/effective_source/` |

## Building the effective-source library

The source vector needs the effective-source C code in `ellipticgsf_neural/effective_source/`
compiled to a shared library in that folder (or pointed to by `ELLIPTICGSF_EFFSOURCE_LIB`). Compile
`kerr-circular.c`; it needs two GSL functions, which the bundled `gsl_ellint_shim.c` supplies without GSL.

```bash
# Linux / macOS, without GSL (the shim)
cd ellipticgsf_neural/effective_source
gcc -O3 -fPIC -shared -I . -o libeffsource.so kerr-circular.c gsl_ellint_shim.c -lm
# or with GSL
gcc -O3 -fPIC -shared -o libeffsource.so kerr-circular.c -lgsl -lgslcblas -lm
```

```bat
REM Windows, x64 Native Tools Command Prompt for VS, with the shim. /EXPORT is required: MSVC
REM exports nothing from a DLL unless named.
cd ellipticgsf_neural\effective_source
cl /nologo /LD /O2 /I . kerr-circular.c gsl_ellint_shim.c /Fe:effsource.dll ^
   /link /EXPORT:effsource_init /EXPORT:effsource_set_particle /EXPORT:effsource_hS_m /EXPORT:effsource_calc_m
```

The tests run without the library: a deterministic stand-in replaces it, except in
`tests/test_effective_source_library.py`, which skips itself when the library is missing.

## Notation: paper ↔ code

| paper | code |
|---|---|
| **M x** = **b** (Sec. III) | `problem.M`, `problem.b`, `solution.x` |
| ψ⁰ … ψ⁹, the ten tensor components | `N_COMPONENTS`, `field_components(problem, x)` (shape 10 × N_r\* × N_θ) |
| r\*, θ, Δr\*, Δθ | `grid.rstar`, `grid.theta`, `grid.delta_rstar`, `grid.delta_theta` |
| r₀, Ω, ω = mΩ | `orbit.r0`, `orbit.omega`, `orbit.mode_frequency(m)` |
| **x**_A, **x**_C, **x**₀ (App. A) | `HORIZON_STRIP_RADII` = 7, `OUTER_STRIP_RADII` = 5, `P.strip` |
| **x**_B (App. A) | `P.bulk` |
| **M**₀, **M**₀B, **M**_B0 (App. A) | `P.M_0_lu` (its LU), `P.M_0B`, `P.M_B0` |
| **M**_S (App. A; **M** in Sec. IV) | `P.M_S` |
| **P**_ILU(0) = **L**₀**U**₀, **P**_ILU(1) = **L**₁**U**₁ (Sec. III C) | `IncompleteLU(M_S, level=0 or 1)`, methods `"ilu0"`, `"ilu1"` |
| **P**_neural = (**L**₀ + δ**L**)(**U**₀ + δ**U**) (Sec. IV A) | `neural_preconditioner`, `neural_factor_values` |
| E⁽⁰⁾_ij, v⃗⁽⁰⁾_i | `gnn_inputs(P, problem)["E0"]`, `["v0"]` |
| Ψ⁽ⁿ⁾, Φ⁽ⁿ⁾, K | `MessagePassingGNN.psi[n]`, `.phi[n]`, `.K` |
| sign(i − j), f^tr_ij, N_row, N_col | `"sign_i_minus_j"`, `"f_tr"`, `inverse_N_row`, `inverse_N_col` |
| E⁽ᴷ⁾_ij | the network's `correction` (see the neural.py docstring for its scaling) |
| 𝓛, cycles c ∈ {1, 2, 4, 7, 11} (Sec. IV B) | `GMRESCycleLoss`, `TrainingParameters.loss_cycles` |
| GMRES with deflated restarts, 50 / 10 (Sec. III B) | `DeflatedRestartGMRES`, `GMRESParameters` |

ILU factors are stored in Doolittle form (unit lower triangle, pivots in the upper one), which the
triangular solves use; the paper writes them in Crout form (pivots on the diagonal of **L**₀, **U**₀ unit),
and `neural.py` converts between the two.

## Performance notes

* The Krylov basis and the factor values are complex64; every sum, residual and the solution are
  complex128, and every restart cycle starts from the true residual, so single precision costs
  efficiency, never accuracy.
* The network runs in bfloat16 where the CPU computes it natively (AVX512-BF16 or AMX; float32
  elsewhere), through `torch.compile` with freezing: about 1.2 s at m = 2 on an 8-thread AVX512-BF16 CPU
  (2.5 s uncompiled). The first compiled call in a process compiles (~7 s with a warm cache);
  `timing.py` does that before the clock starts. `torch.compile` needs a 64-bit C++ compiler (on Windows,
  MSVC's `cl.exe` from an x64 Native Tools Command Prompt); without one the network runs uncompiled.
  `neural.benchmark_inference(problem)` times every combination on your machine.

## Copyrighted source

`ellipticgsf_neural/effective_source/kerr-circular.c` and `effsource.h` are © Barry Wardell
