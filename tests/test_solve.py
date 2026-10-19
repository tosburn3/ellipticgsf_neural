"""
GMRES with deflated restarts, the Schur-complement preconditioner of App. A, the incomplete LU
factorizations, and solve() (iterative.py, solve.py), on a small problem (common.py).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import spsolve_triangular

from ellipticgsf_neural import field_components, retarded_field_components
from ellipticgsf_neural.iterative import DeflatedRestartGMRES, SchurComplementPreconditioner, boundary_strip_mask
from ellipticgsf_neural.solve import METHODS, solve

from .common import GMRES, small_problem


def test_the_direct_solve_solves_the_system():
    s = solve(small_problem(), "direct", GMRES)
    assert s.relative_residual < 1e-9 and s.iterations is None and s.factor_mb > 0


def test_both_ilu_preconditioners_converge_to_the_direct_answer():
    p = small_problem()
    x = solve(p, "direct", GMRES).x
    for method in ("ilu0", "ilu1"):
        s = solve(p, method, dataclasses.replace(GMRES, tolerance=1e-11))
        assert s.converged and s.relative_residual <= 1e-11
        assert np.linalg.norm(s.x - x) / np.linalg.norm(x) < 1e-7
        assert s.history.size == s.iterations


def test_ilu1_needs_no_more_iterations_than_ilu0():
    p = small_problem()
    assert solve(p, "ilu1", GMRES).iterations <= solve(p, "ilu0", GMRES).iterations


def test_preconditioning_cuts_the_iteration_count():
    p = small_problem()
    assert solve(p, "ilu0", GMRES).iterations < solve(p, "none", GMRES, max_iterations=2000).iterations


def test_the_preconditioner_is_exact_with_an_exact_factorization_of_the_schur_complement():
    """App. A: with the exact LU of M_S in place of ILU, P = M."""
    p = small_problem()
    P = SchurComplementPreconditioner(p, "ilu0", factor_precision=np.complex128)
    lu = spla.splu(P.M_S.tocsc())
    v = np.random.default_rng(1).standard_normal(p.n) + 0j
    # P^-1 v by the block formulas of App. A, with the exact bulk solve
    y_0 = P.M_0_lu.solve(v[P.strip])
    x_B = lu.solve(v[P.bulk] - P.M_B0 @ y_0)
    x = np.empty(p.n, complex)
    x[P.bulk], x[P.strip] = x_B, y_0 - P.M_0_lu.solve(P.M_0B @ x_B)
    exact = spla.spsolve(p.M.tocsc(), v)          # (this small M is ill-conditioned: compare solutions)
    assert np.linalg.norm(x - exact) / np.linalg.norm(exact) < 1e-8


def test_the_schur_complement_is_the_textbook_one():
    """M_S = M_B - M_B0 M_0^-1 M_0B (App. A), checked on random vectors with M's own blocks."""
    p = small_problem()
    P = SchurComplementPreconditioner(p, "ilu0", factor_precision=np.complex128)
    s, b = P.strip, P.bulk
    M = p.M.tocsr()
    M_0, M_0B, M_B0, M_B = M[s][:, s].tocsc(), M[s][:, b], M[b][:, s], M[b][:, b]
    rng = np.random.default_rng(5)
    for _ in range(3):
        v = rng.standard_normal(b.size) + 1j * rng.standard_normal(b.size)
        want = M_B @ v - M_B0 @ spla.spsolve(M_0, M_0B @ v)
        assert np.linalg.norm(P.M_S @ v - want) <= 1e-9 * np.linalg.norm(want)


def test_the_compiled_apply_matches_the_block_formulas():
    p = small_problem()
    P = SchurComplementPreconditioner(p, "ilu1", factor_precision=np.complex128)
    v = np.random.default_rng(2).standard_normal(p.n) + 1j * np.random.default_rng(3).standard_normal(p.n)
    F = P.ilu
    y_0 = P.M_0_lu.solve(v[P.strip])
    r = v[P.bulk] - P.M_B0 @ y_0
    rows = np.repeat(np.arange(F.n), np.diff(F.indptr))
    lower = F.indices < rows
    L = sp.csr_matrix((F.values[lower], (rows[lower], F.indices[lower])), shape=(F.n, F.n)) + sp.identity(F.n)
    U = sp.csr_matrix((F.values[~lower], (rows[~lower], F.indices[~lower])), shape=(F.n, F.n))
    x_B = spsolve_triangular(U.tocsr(), spsolve_triangular(L.tocsr(), r, lower=True), lower=False)
    want = np.empty(p.n, complex)
    want[P.bulk], want[P.strip] = x_B, y_0 - P.M_0_lu.solve(P.M_0B @ x_B)
    assert np.linalg.norm(P.apply(v) - want) / np.linalg.norm(want) < 1e-9


def test_the_strips_are_the_first_7_and_last_5_radial_lines():
    p = small_problem()
    mask = boundary_strip_mask(p.grid).reshape(p.grid.n_theta, p.grid.n_rstar, 10)
    assert mask[:, :7].all() and mask[:, -5:].all() and not mask[:, 7:-5].any()


def test_field_components_matches_the_global_index_formula():
    p = small_problem()
    x = np.arange(p.n, dtype=complex)
    f = field_components(p, x)
    l, i, j = 3, 5, 7
    assert f[l, i, j] == l + 10 * (i + p.grid.n_rstar * j)


def test_the_retarded_field_adds_the_puncture_only_inside_the_worldtube():
    p = small_problem()
    x = solve(p, "direct", GMRES).x
    diff = retarded_field_components(p, x) - field_components(p, x)
    (imin, imax), (jmin, jmax) = p.grid.worldtube_rstar_indices, p.grid.worldtube_theta_indices
    inside = np.zeros(diff.shape[1:], bool)
    inside[imin:imax + 1, jmin:jmax + 1] = True
    assert np.all(diff[:, ~inside] == 0) and np.any(diff[:, inside] != 0)


def test_the_jacobi_factorization_is_the_diagonal_of_the_schur_complement():
    p = small_problem()
    P = SchurComplementPreconditioner(p, "jacobi", factor_precision=np.complex128)
    rng = np.random.default_rng(4)
    v = rng.standard_normal(p.n) + 1j * rng.standard_normal(p.n)
    y_0 = P.M_0_lu.solve(v[P.strip])
    x_B = (v[P.bulk] - P.M_B0 @ y_0) / P.M_S.diagonal()
    x = np.empty(p.n, complex)
    x[P.bulk], x[P.strip] = x_B, y_0 - P.M_0_lu.solve(P.M_0B @ x_B)
    assert np.allclose(P.apply(v), x, rtol=1e-9, atol=1e-9 * np.abs(x).max())
    assert P.ilu is None and P.memory_mb > 0
    s = solve(p, "jacobi", GMRES, max_iterations=50)      # runs end to end (it is not expected to converge)
    assert s.method == "jacobi" and s.history.size == s.iterations


def test_gmres_restarts_from_the_true_residual():
    p = small_problem()
    P = SchurComplementPreconditioner(p, "ilu0")
    x, info = DeflatedRestartGMRES(p.M, P, restart_length=20, deflation_vectors=5).solve(
        p.b, tolerance=1e-10, max_iterations=2000)
    assert info["converged"]
    assert np.isclose(np.linalg.norm(p.b - p.M @ x) / np.linalg.norm(p.b), info["relative_residual"], rtol=1e-6)


def test_gmres_reports_the_start_of_every_cycle():
    p = small_problem()
    P = SchurComplementPreconditioner(p, "ilu0")
    seen = []
    _, info = DeflatedRestartGMRES(p.M, P, restart_length=8, deflation_vectors=3).solve(
        p.b, tolerance=1e-10, max_iterations=400, cycle_start_callback=lambda c, r, C: seen.append((c, C is None)))
    assert [c for c, _ in seen] == list(range(1, len(seen) + 1)) and seen[0][1] and not seen[-1][1]
    assert len(seen) == info["cycles"] or len(seen) == info["cycles"] + 1


def test_an_unknown_method_is_refused():
    try:
        solve(small_problem(), "gauss-seidel", GMRES)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown method was accepted")
    assert METHODS == ("direct", "none", "ilu0", "ilu1", "jacobi", "neural")
