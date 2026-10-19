"""A small problem for the solver tests: the real operator M on a coarse grid, with the deterministic stub
of the effective-source library (stubs.py), so no compiled library is needed. The numbers are not physics,
but the algebra is real."""

from __future__ import annotations

import numpy as np

from ellipticgsf_neural import GMRESParameters, Grid, Orbit, Problem, assemble, source_vector

from .stubs import stub_puncture_field

#: The paper's GMRES settings (scripts/settings.py), spelled out: the package has no defaults.
GMRES = GMRESParameters(restart_length=50, deflation_vectors=10, tolerance=1e-10, max_iterations=6000)


def small_problem(m: int = 2) -> Problem:
    orbit = Orbit(0.0, 6.0)
    grid = Grid.worldtube(orbit, delta_rstar=0.5, delta_theta=np.pi / 18, rstar_inner=-6.0, rstar_outer=12.0)
    puncture = stub_puncture_field(orbit)
    M = assemble(m, orbit, grid).M.tocsr()
    M.sum_duplicates()
    M.eliminate_zeros()
    M.sort_indices()
    return Problem(m, orbit, grid, M, np.asarray(source_vector(m, puncture, grid), complex), puncture)
