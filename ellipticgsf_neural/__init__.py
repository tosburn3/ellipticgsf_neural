"""
ellipticgsf_neural -- neural preconditioners for the elliptic m-mode equations of the gravitational
self-force of a circular orbit around a Schwarzschild black hole (the paper "Machine learning and high
accuracy: Neural preconditioners for computational gravity"; the elliptic formulation is that of
arXiv:2608.07934).

    from ellipticgsf_neural import GMRESParameters, GridParameters, build_problem, solve

    problem = build_problem(m=2, r0=6.0, grid_parameters=GridParameters(...))     # M x = b (Secs. II, III)
    solution = solve(problem, "neural", gmres=GMRESParameters(...))               # Secs. III, IV, App. A
    print(solution.summary())

The values used in the paper for the grid, GMRES and training are in scripts/settings.py, which also
sets the thread counts; the scripts in scripts/ make every figure.

THE MODULES
    problem      the orbit, the (r*, theta) grid, build_problem: M and b (Secs. II and III)
    assemble     the sparse matrix M: coefficients, finite differences, boundary conditions (Sec. II C)
    source       the source vector b: puncture field, effective source, worldtube jumps (Sec. II)
    iterative    GMRES with deflated restarts (Sec. III B), the boundary-strip Schur complement
                 preconditioner (App. A), ILU(0) / ILU(1) (Sec. III C)
    neural       the GNN, P_neural, inference, the GMRES-cycle loss and its gradient, training (Sec. IV)
    solve        solve(problem, method): direct, none, ilu0, ilu1, jacobi, neural
"""

# PyTorch first, before numpy, scipy and numba. On Windows they load their own copies of shared DLLs
# (Intel OpenMP, the MSVC runtime); if an older copy is already loaded when torch is imported, torch
# fails with "DLL load failed while importing _C: The specified procedure could not be found".
# Optional: everything except the neural preconditioner works without it.
try:
    import torch  # noqa: F401
except Exception as exc:                    # not installed, or broken: reported when "neural" is used
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from .assemble import HORIZON_STRIP_RADII, N_COMPONENTS, OUTER_STRIP_RADII, assemble
from .iterative import DeflatedRestartGMRES, GMRESParameters, SchurComplementPreconditioner
from .problem import (Grid, GridParameters, Orbit, Problem, build_problem, field_components, make_grid,
                      retarded_field_components)
from .solve import METHOD_NAMES, METHODS, Solution, solve
from .source import load_effective_source_library, source_vector

__all__ = ["N_COMPONENTS", "HORIZON_STRIP_RADII", "OUTER_STRIP_RADII", "assemble", "DeflatedRestartGMRES",
           "GMRESParameters", "SchurComplementPreconditioner", "Grid", "GridParameters", "Orbit", "Problem",
           "build_problem", "field_components", "make_grid", "retarded_field_components", "METHOD_NAMES",
           "METHODS", "Solution", "solve", "load_effective_source_library", "source_vector"]
