"""
Solve M x = b (problem.py) by one of the methods compared in the paper (Secs. III and V).

    solution = solve(problem, "neural", gmres=GMRESParameters(...))
    solution.x, solution.iterations, solution.relative_residual, solution.history, solution.setup_time, ...

    "direct"   SuperLU sparse LU (COLAMD ordering) of M (Sec. III A)
    "none"     GMRES with deflated restarts (Sec. III B), no preconditioner (does not converge on these
               problems; the baseline)
    "ilu0"     GMRES + the Schur-complement preconditioner of App. A with P_S = ILU(0), L0 U0 (Sec. III C)
    "ilu1"     the same with P_S = ILU(1), L1 U1
    "jacobi"   the same with P_S = diag(M_S), point Jacobi (for comparison; does not converge on these problems)
    "neural"   the same with P_S = P_neural = (L0 + dL)(U0 + dU) (Sec. IV, neural.py)

GMRES is right-preconditioned, so `history` is an estimate of the true relative residual ||b - M x|| / ||b||
at every iteration, comparable across methods. Setup time = everything after M and b are built (the
strip LU, the Schur complement, the factorization, the network); solve time = the iterations (or the
substitution).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse.linalg as spla

from .iterative import DeflatedRestartGMRES, GMRESParameters, IdentityPreconditioner, SchurComplementPreconditioner

__all__ = ["METHODS", "METHOD_NAMES", "Solution", "preconditioner", "solve"]

METHODS = ("direct", "none", "ilu0", "ilu1", "jacobi", "neural")
METHOD_NAMES = {"direct": "Direct (SuperLU)", "none": "No preconditioner", "ilu0": "ILU(0)", "ilu1": "ILU(1)",
                "jacobi": "Jacobi", "neural": "Neural"}


@dataclass
class Solution:
    method: str
    x: np.ndarray
    relative_residual: float                 # the true ||b - M x|| / ||b||
    converged: bool
    setup_time: float
    solve_time: float
    iterations: Optional[int] = None         # None for the direct solve
    history: Optional[np.ndarray] = None     # relative residual estimate per iteration
    factor_mb: float = 0.0                   # stored factors (SuperLU's L + U, or the preconditioner's)
    timings: dict = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        return self.setup_time + self.solve_time

    def summary(self) -> str:
        its = "" if self.iterations is None else f", {self.iterations} iterations"
        flag = "" if self.converged else " (NOT converged)"
        note = ""
        if "network" in self.timings:
            note = f" (network {self.timings['network']:.1f} s" + (
                ", including torch.compile: the first call for this graph size" if self.timings.get("network_compiled")
                else "") + ")"
        return (f"{METHOD_NAMES[self.method]:18s} setup {self.setup_time:7.2f} s, solve {self.solve_time:7.2f} s, "
                f"total {self.total_time:7.2f} s{its}, residual {self.relative_residual:.1e}{flag}, "
                f"factors {self.factor_mb:,.0f} MB{note}")


def _direct(problem):
    """SuperLU. Its BLAS calls are many and tiny, and are much slower with the multithreaded,
    passive-waiting BLAS used for GMRES, so BLAS is limited to one thread here."""
    t0 = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            lu = spla.splu(problem.M.tocsc())
    except ImportError:
        lu = spla.splu(problem.M.tocsc())
    t1 = time.perf_counter()
    x = lu.solve(problem.b)
    t2 = time.perf_counter()
    return x, t1 - t0, t2 - t1, (lu.L.nnz + lu.U.nnz) * 16 / 2**20


def preconditioner(problem, method: str, network=None, neural_precision=None, neural_compile=True):
    """The preconditioner P for an iterative method, and its setup time. For "neural": network is a
    MessagePassingGNN or GNNInference (default: the shipped network for problem.m), neural_precision a
    torch dtype (default: bfloat16 where the CPU supports it natively, else float32), neural_compile
    whether to use torch.compile."""
    t0 = time.perf_counter()
    if method == "none":
        P = IdentityPreconditioner()
    elif method in ("ilu0", "ilu1", "jacobi"):
        P = SchurComplementPreconditioner(problem, method)
    elif method == "neural":
        from .neural import neural_preconditioner      # imports torch only when needed
        P = neural_preconditioner(problem, network, neural_precision, neural_compile)
    else:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    return P, time.perf_counter() - t0


def solve(problem, method: str, gmres: GMRESParameters, *, max_iterations: Optional[int] = None,
          network=None, neural_precision=None, neural_compile=True) -> Solution:
    """Solve problem.M x = problem.b by `method` (see the module docstring). gmres: the GMRES settings
    (ignored by "direct"); max_iterations overrides gmres.max_iterations. network, neural_precision,
    neural_compile: see preconditioner()."""
    if method == "direct":
        x, t_setup, t_solve, mb = _direct(problem)
        res = float(np.linalg.norm(problem.M @ x - problem.b) / np.linalg.norm(problem.b))
        return Solution(method, x, res, True, t_setup, t_solve, factor_mb=mb)
    P, t_setup = preconditioner(problem, method, network, neural_precision, neural_compile)
    solver = DeflatedRestartGMRES(problem.M, P, gmres.restart_length, gmres.deflation_vectors, np.complex64)
    x, info = solver.solve(problem.b, gmres.tolerance,
                           gmres.max_iterations if max_iterations is None else max_iterations)
    solver.release()
    return Solution(method, x, info["relative_residual"], info["converged"], t_setup, info["time"],
                    info["iterations"], info["history"], getattr(P, "memory_mb", 0.0), getattr(P, "timings", {}))
