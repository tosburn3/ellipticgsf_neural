"""
The sparse matrix M (assemble.py): the coefficient matrices and boundary-condition stencils, the fast
assembler against the naive reference, and the structure of M.

The coefficient expressions are machine-emitted, so there is no point reading them; these tests check them
against a closed form where one is published, check that they are finite and correctly shaped across the
parameter range that gets used, and pin the structural facts the assembler relies on -- how many stencil
entries each boundary condition has and how far into the grid it reaches.

The load-bearing test is `test_fast_assembly_matches_the_naive_reference`:
`assemble` and `assemble_reference` are independent implementations of the
same equations, one vectorized and one written out entry by entry, so an
exact match means the index arithmetic and the batching are both right.
Everything else here pins a structural property that a silent bug could
otherwise hide behind.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ellipticgsf_neural.assemble import (
    HORIZON_STRIP_RADII,
    N_COMPONENTS,
    OUTER_STRIP_RADII,
    assemble,
    assemble_reference,
    first_rstar_derivative_coefficients,
    first_theta_derivative_coefficients,
    near_horizon_boundary_condition,
    non_derivative_coefficients,
    outer_boundary_condition,
    outer_boundary_condition_m0,
    polar_condition_classes,
)
from ellipticgsf_neural.problem import Grid, Orbit

from ._shim import get_pytest

pytest = get_pytest()

# ======================================================================
# The coefficient matrices and boundary-condition stencils
# ======================================================================


R = np.array([2.5, 4.0, 6.0, 10.0, 50.0])
THETAS = (0.05, 0.4, 1.0, np.pi / 2, 2.2, np.pi - 0.05)
SPINS = (0.0, 0.3, 0.9, -0.6)


# --------------------------------------------------------------------------
# The published Schwarzschild limit
# --------------------------------------------------------------------------


def schwarzschild_A(r: np.ndarray) -> np.ndarray:
    """
    A in the Schwarzschild limit:

        A = -(2 / r^2) diag(2, 2, 1, 1, 2, 1, 1, 0, 0, 0)

    The three trailing zeros are worth noticing -- those three equations
    carry no first radial derivative at all, so the block structure is not
    uniform across components.
    """
    diag = np.array([2, 2, 1, 1, 2, 1, 1, 0, 0, 0], dtype=float)
    out = np.zeros((N_COMPONENTS, N_COMPONENTS, len(r)), dtype=complex)
    for k in range(N_COMPONENTS):
        out[k, k, :] = -2.0 / r**2 * diag[k]
    return out


def test_A_reduces_to_the_schwarzschild_limit():
    """
    An independent check on the whole of first_rstar_derivative_coefficients: at zero spin it must
    collapse to a diagonal form with no m or omega dependence at all.
    """
    want = schwarzschild_A(R)
    for m in (0, 2, 5):
        for theta in (0.4, 1.0, 2.2):
            got = first_rstar_derivative_coefficients(m, 0.0, 0.13 * m, R, theta)
            assert got.shape == want.shape
            assert np.max(np.abs(got - want)) < 1e-10


# --------------------------------------------------------------------------
# Shape and finiteness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", list(SPINS))
def test_coefficients_are_finite_and_correctly_shaped(a):
    for m in (0, 1, 2, 7):
        for theta in THETAS:
            for func in (first_rstar_derivative_coefficients, first_theta_derivative_coefficients, non_derivative_coefficients):
                out = func(m, a, 0.068 * m, R, theta)
                assert out.shape == (N_COMPONENTS, N_COMPONENTS, len(R)), func.__name__
                assert np.all(np.isfinite(out)), f"{func.__name__} at a={a}, theta={theta}"


def test_coefficients_broadcast_over_r_rather_than_looping():
    """
    Assembly speed depends on a whole radial line coming back from one call,
    so a scalar-r result would be a performance regression, not just a shape
    difference.
    """
    line = non_derivative_coefficients(2, 0.3, 0.13, R, 1.0)
    for i, r in enumerate(R):
        one = non_derivative_coefficients(2, 0.3, 0.13, np.array([r]), 1.0)
        assert np.max(np.abs(line[:, :, i] - one[:, :, 0])) == 0.0


def test_each_matrix_depends_on_exactly_what_it_should():
    """
    Recorded because it is not obvious and it is structurally useful:

        first_rstar_derivative_coefficients   m and a, but NOT omega
        first_theta_derivative_coefficients   a only -- neither m nor omega
        non_derivative_coefficients   m, a and omega

    So B is the same array for every mode at fixed spin, and A changes with m
    only through terms carrying a factor of a. If any of this changes, the
    coefficients have changed.
    """
    expected = {
        "first_rstar_derivative_coefficients": (True, False, True),
        "first_theta_derivative_coefficients": (False, False, True),
        "non_derivative_coefficients": (True, True, True),
    }
    for func in (first_rstar_derivative_coefficients, first_theta_derivative_coefficients, non_derivative_coefficients):
        base = func(2, 0.3, 0.13, R, 1.0)
        depends = (
            not np.allclose(base, func(5, 0.3, 0.13, R, 1.0)),  # m
            not np.allclose(base, func(2, 0.3, 0.41, R, 1.0)),  # omega
            not np.allclose(base, func(2, 0.0, 0.13, R, 1.0)),  # a
        )
        assert depends == expected[func.__name__], (func.__name__, depends)


def test_the_blocks_are_sparse_and_C_is_the_densest():
    """
    The blocks' own sparsity is what gives the assembled matrix its ten-or-so
    nonzeros per row, which is the single most consequential structural fact
    about it, so the counts are pinned here.

    Out of 100 slots each: at zero spin 7, 18 and 53 for A, B and C; at
    nonzero spin 26, 26 and 84. Spin fills in couplings that vanish in
    Schwarzschild, which is why a Kerr matrix is denser than a Schwarzschild
    one at the same resolution.
    """
    expected = {0.0: (7, 18, 53), 0.3: (26, 26, 84), 0.9: (26, 26, 84)}
    for a, want in expected.items():
        got = tuple(
            int(np.count_nonzero(np.abs(func(2, a, 0.13, R, 1.0)[:, :, 0]) > 1e-14))
            for func in (first_rstar_derivative_coefficients, first_theta_derivative_coefficients, non_derivative_coefficients)
        )
        assert got == want, f"a={a}: {got} != {want}"
        assert max(got) < N_COMPONENTS * N_COMPONENTS, "the blocks should all still be sparse"


# --------------------------------------------------------------------------
# The boundary conditions
# --------------------------------------------------------------------------


def test_stencil_shapes_are_what_the_assembler_assumes():
    """
    Each entry is (row_field, row_offset, col_field, col_offset, col_dtheta,
    value), with None fields meaning "every field" and a per-field value list.
    """
    theta = np.linspace(0.2, np.pi - 0.2, 5)
    for func in (near_horizon_boundary_condition, outer_boundary_condition, outer_boundary_condition_m0):
        for row_field, row_off, col_field, col_off, col_dj, value in func(
            2, 0.3, 0.13, 2.5, theta, 0.05, 0.05
        ):
            assert (row_field is None) == (col_field is None)
            assert isinstance(row_off, int) and isinstance(col_off, int)
            assert col_dj in (-1, 0, 1)
            if row_field is None:
                assert len(value) == N_COMPONENTS
                for v in value:
                    assert np.shape(v) == theta.shape
            else:
                assert 0 <= row_field < N_COMPONENTS and 0 <= col_field < N_COMPONENTS
                assert np.shape(value) == theta.shape


def test_stencil_entry_counts_are_pinned():
    """
    A silent change in a boundary condition is exactly the kind of thing that
    should not slip through, so the counts are nailed down here.
    """
    theta = np.linspace(0.2, np.pi - 0.2, 5)
    assert len(near_horizon_boundary_condition(2, 0.0, 0.13, 2.5, theta, 0.05, 0.05)) == 43
    assert len(outer_boundary_condition(2, 0.0, 0.13, 2.5, theta, 0.05, 0.05)) == 5
    assert len(outer_boundary_condition_m0(0, 0.0, 0.0, 2.5, theta, 0.05, 0.05)) == 3


def test_stencil_depths_match_the_assemblers_grid_requirement():
    """
    `assemble` refuses grids with fewer than 7 radial nodes because of how far
    these reach; that number has to come from here.
    """
    theta = np.linspace(0.2, np.pi - 0.2, 5)
    inner = max(
        e[3] for e in near_horizon_boundary_condition(2, 0.0, 0.13, 2.5, theta, 0.05, 0.05)
    )
    outer = min(
        e[3] for e in outer_boundary_condition(2, 0.0, 0.13, 2.5, theta, 0.05, 0.05)
    )
    assert inner == 6, f"near-horizon condition now reaches {inner} nodes in"
    assert outer == -4, f"outgoing condition now reaches {-outer} nodes in"


def test_the_boundary_strips_are_exactly_the_radii_the_boundary_conditions_couple():
    """The strips x_A and x_C of App. A (iterative.boundary_strip_mask) are the radii the near-horizon and
    outer boundary conditions reach: offsets 0..6 and 0..-4 from the edges."""
    theta = np.linspace(0.2, np.pi - 0.2, 5)
    for m in (1, 2, 5):
        reach_in = max(max(e[1], e[3]) for e in near_horizon_boundary_condition(m, 0.0, 0.13, 2.5, theta, 0.05, 0.05))
        reach_out = min(min(e[1], e[3]) for e in outer_boundary_condition(m, 0.0, 0.13, 2.5, theta, 0.05, 0.05))
        assert HORIZON_STRIP_RADII == reach_in + 1
        assert OUTER_STRIP_RADII == -reach_out + 1


@pytest.mark.parametrize("a", list(SPINS))
def test_boundary_values_are_finite(a):
    theta = np.linspace(0.05, np.pi - 0.05, 41)
    for func in (near_horizon_boundary_condition, outer_boundary_condition):
        for row_field, _, _, _, _, value in func(2, a, 0.13, 2.2, theta, 0.05, 0.05):
            for v in (value if row_field is None else [value]):
                assert np.all(np.isfinite(v)), f"{func.__name__} at a={a}"


def test_the_m0_outer_condition_is_static_and_real():
    """
    At m = 0 the frequency vanishes, so the outgoing condition degenerates to
    a one-sided derivative with real weights.
    """
    theta = np.linspace(0.2, np.pi - 0.2, 5)
    for row_field, _, _, _, _, value in outer_boundary_condition_m0(0, 0.0, 0.0, 2.5, theta, 0.05, 0.05):
        for v in (value if row_field is None else [value]):
            assert np.max(np.abs(np.imag(v))) == 0.0


# ======================================================================
# Assembly
# ======================================================================


# Small enough for the naive assembler, but with room for the boundary
# stencils: the near-horizon condition reaches six nodes in and the outgoing
# condition four.
SMALL = dict(delta_rstar=0.6, delta_theta=np.pi / 9, rstar_min=-6.0, rstar_max=0.6)


def _case(m: int, a: float, r0: float = 6.0):
    return m, Orbit(a, r0), Grid.uniform(a, **SMALL)


# --------------------------------------------------------------------------
# Against the naive reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m,a", [(2, 0.0), (0, 0.0), (1, 0.6), (5, 0.9), (3, 0.3)])
def test_fast_assembly_matches_the_naive_reference(m, a):
    m, orbit, grid = _case(m, a)
    fast = assemble(m, orbit, grid).M
    slow = assemble_reference(m, orbit, grid)
    diff = (fast - slow).tocoo()
    scale = max(1.0, float(np.max(np.abs(slow.data))))
    assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-12 * scale


def test_fast_and_naive_share_the_sparsity_pattern():
    m, orbit, grid = _case(2, 0.0)
    fast = assemble(m, orbit, grid).M
    slow = assemble_reference(m, orbit, grid)
    fast.eliminate_zeros()
    slow.eliminate_zeros()
    assert fast.nnz == slow.nnz
    assert np.array_equal(fast.indptr, slow.indptr)
    assert np.array_equal(fast.indices, slow.indices)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_index_map_is_field_fastest_then_radial_then_polar():
    m, orbit, grid = _case(2, 0.0)
    system = assemble(m, orbit, grid)
    n, nr = system.n, grid.n_rstar
    assert system.index(0, 0, 0) == 0
    assert system.index(1, 0, 0) == 1
    assert system.index(0, 1, 0) == n
    assert system.index(0, 0, 1) == n * nr
    assert system.size == n * grid.n_rstar * grid.n_theta


def test_every_unknown_is_constrained():
    m, orbit, grid = _case(2, 0.3)
    A = assemble(m, orbit, grid).M
    counts = np.diff(A.indptr)
    assert np.all(counts > 0), f"{int((counts == 0).sum())} empty rows"


def test_interior_operators_do_not_leak_into_boundary_rows():
    """
    A pole row must be nothing but its own polar condition: three theta
    neighbours at fixed (field, radius), and no coupling to other fields.
    """
    m, orbit, grid = _case(5, 0.0)  # m >= 3: every field gets Dirichlet
    system = assemble(m, orbit, grid)
    A = system.M.tolil()
    for l, i in ((1, 4), (7, 2)):
        row = A.rows[system.index(l, i, 0)]
        assert set(row) <= {system.index(l, i, j) for j in (0, 1, 2)}


def test_the_matrix_is_sparse_in_the_way_the_stencils_imply():
    """About ten entries per row; a large change means a stencil changed."""
    m, orbit, grid = _case(2, 0.0)
    A = assemble(m, orbit, grid).M
    per_row = A.nnz / A.shape[0]
    assert 8.0 < per_row < 14.0, per_row


def test_polar_classes_switch_over_at_the_documented_mode_numbers():
    assert list(polar_condition_classes(0)) == [0, 0, 1, 1, 0, 1, 1, 1, 1, 1]
    assert list(polar_condition_classes(1)) == list(polar_condition_classes(-1))
    assert list(polar_condition_classes(2)) == list(polar_condition_classes(-2))
    assert list(polar_condition_classes(3)) == [1] * 10
    assert list(polar_condition_classes(9)) == [1] * 10


def test_dirichlet_rows_are_exactly_unit_rows():
    """A Dirichlet component at the pole should give a row with a single 1."""
    m, orbit, grid = _case(5, 0.0)
    system = assemble(m, orbit, grid)
    A = system.M
    row = A.getrow(system.index(3, 4, 0))
    assert row.nnz == 1
    assert abs(row.data[0] - 1.0) < 1e-15


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_grid_too_short_for_the_stencils_is_refused():
    orbit = Orbit(0.0, 6.0)
    grid = Grid.uniform(0.0, 1.0, np.pi / 9, -6.0, -1.0)  # 6 radial nodes
    try:
        assemble(2, orbit, grid)
    except (ValueError, IndexError) as exc:
        assert "radial" in str(exc)
        return
    raise AssertionError("expected a refusal for a grid shorter than the stencil")


def test_a_grid_built_for_a_different_spin_is_refused():
    grid = Grid.uniform(0.3, **SMALL)
    try:
        assemble(2, Orbit(0.0, 6.0), grid)
    except ValueError as exc:
        assert "spin" in str(exc)
        return
    raise AssertionError("expected a refusal for mismatched spins")


# --------------------------------------------------------------------------
# Solvability
# --------------------------------------------------------------------------


def test_the_system_solves():
    m, orbit, grid = _case(2, 0.0)
    A = assemble(m, orbit, grid).M
    rng = np.random.default_rng(0)
    b = rng.normal(size=A.shape[0]) + 1j * rng.normal(size=A.shape[0])
    x = spla.spsolve(A.tocsc(), b)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-8
