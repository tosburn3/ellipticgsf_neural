"""
The puncture field, the effective source, and the right-hand-side vector.

Run on the deterministic stubs, so none of this needs the compiled library.
That means nothing here checks the physics of the puncture field -- that is
the library's business -- but it does check every piece of assembly built on
top of it, which is where the mistakes that produce plausible-looking wrong
answers live: a sign on a jump term, a coefficient evaluated at the neighbour
instead of at the row, a halo row missed.

The load-bearing test is `test_source_vector_matches_the_naive_reference`,
for the same reason as in the assembler's suite.
"""

from __future__ import annotations

import numpy as np

from ellipticgsf_neural.problem import Grid, Orbit
from ellipticgsf_neural.source import source_vector_reference
from ellipticgsf_neural.source import PunctureField, source_vector
from ellipticgsf_neural.source import N_COMPONENTS, _source_matrix, azimuthal_shift

from ._shim import get_pytest
from .stubs import stub_h_s, stub_puncture_field, stub_s_eff

pytest = get_pytest()

WT = dict(delta_rstar=0.15, delta_theta=np.pi / 36, rstar_inner=-6.0, rstar_outer=30.0)


def _setup(m: int = 2, a: float = 0.0):
    orbit = Orbit(a, 6.0)
    return m, stub_puncture_field(orbit), Grid.worldtube(orbit, **WT)


# --------------------------------------------------------------------------
# The puncture field
# --------------------------------------------------------------------------


def test_azimuthal_shift_vanishes_for_schwarzschild():
    assert azimuthal_shift(0.0, 4.0) == 0.0
    assert azimuthal_shift(0.6, 4.0) != 0.0


def test_puncture_field_assembly_is_what_it_claims():
    """Check the glue explicitly: factor of r, the phase, the per-field scale."""
    from ellipticgsf_neural.source import _psi_scale

    m, puncture, _ = _setup()
    r, theta = 6.3, 1.0
    raw = stub_h_s(m, r, theta)
    want = _psi_scale(m, 0.0, r, theta) * (r * (raw[:10] + 1j * raw[10:]) / (2 * np.pi))
    assert np.max(np.abs(puncture.psi(m, r, theta) - want)) < 1e-14


def test_effective_source_carries_no_factor_of_r():
    m, puncture, _ = _setup()
    r, theta = 6.3, 1.0
    raw = stub_s_eff(m, r, theta)
    want = _source_matrix(m, 0.0, r, theta) @ ((raw[:10] + 1j * raw[10:]) / (2 * np.pi))
    assert np.max(np.abs(puncture.source(m, r, theta) - want)) < 1e-14


def test_the_source_transformation_is_diagonal_for_this_system():
    matrix = _source_matrix(2, 0.4, 5.0, 1.2)
    assert np.allclose(matrix, np.diag(np.diag(matrix)))


def test_a_wrong_length_evaluation_is_rejected():
    orbit = Orbit(0.0, 6.0)
    puncture = PunctureField(
        orbit=orbit, h_s=lambda m, r, th: np.zeros(7), s_eff=stub_s_eff
    )
    try:
        puncture.psi(2, 6.0, 1.0)
    except ValueError as exc:
        assert "20" in str(exc)
        return
    raise AssertionError("expected a refusal for a wrong-length evaluation")


def test_the_kerr_phase_has_unit_modulus_up_to_the_normalization():
    """The phase is exp(-i m Delta_phi) / 2pi, so |.| must be exactly 1/2pi."""
    orbit = Orbit(0.6, 8.0)
    puncture = stub_puncture_field(orbit)
    for r in (2.5, 6.0, 40.0):
        assert abs(abs(puncture._phase(3, r)) - 1.0 / (2 * np.pi)) < 1e-15


# --------------------------------------------------------------------------
# The right-hand side
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m,a", [(2, 0.0), (0, 0.0), (1, 0.0), (3, 0.6), (5, 0.9)])
def test_source_vector_matches_the_naive_reference(m, a):
    m, puncture, grid = _setup(m, a)
    fast = source_vector(m, puncture, grid)
    slow = source_vector_reference(m, puncture, grid)
    scale = max(1.0, float(np.max(np.abs(slow))))
    assert np.max(np.abs(fast - slow)) < 1e-12 * scale


def test_footprint_is_the_worldtube_plus_a_one_node_halo():
    """
    The nonzero pattern is a strong structural check. Interior nodes get the
    effective source; the two polar faces and the row just outside each get
    polar jump terms, at radial indices inside the worldtube only; likewise
    for the radial faces. The four corners of the halo get nothing, because
    neither loop reaches them.
    """
    m, puncture, grid = _setup()
    n, nr = N_COMPONENTS, grid.n_rstar
    vec = source_vector(m, puncture, grid)
    imin, imax = grid.worldtube_rstar_indices
    jmin, jmax = grid.worldtube_theta_indices

    expected = {
        (i, j)
        for j in range(jmin, jmax + 1)
        for i in range(imin, imax + 1)
    }
    for i in range(imin, imax + 1):
        expected.update({(i, jmin - 1), (i, jmax + 1)})
    for j in range(jmin, jmax + 1):
        expected.update({(imin - 1, j), (imax + 1, j)})

    touched = {(k // n % nr, k // n // nr) for k in np.flatnonzero(vec)}
    assert touched == expected, (
        f"footprint differs: {len(touched)} nodes touched, {len(expected)} expected; "
        f"unexpected {sorted(touched - expected)[:5]}, "
        f"missing {sorted(expected - touched)[:5]}"
    )
    assert np.count_nonzero(vec) == len(expected) * n

    for i, j in ((imin - 1, jmin - 1), (imax + 1, jmax + 1)):
        base = n * (i + nr * j)
        assert np.all(vec[base : base + n] == 0), "a halo corner was written to"


def test_the_vector_is_linear_in_the_point_evaluations():
    """
    Both the puncture field and the effective source are linear in the raw
    values, so the whole vector must be: scaling the evaluations by three has
    to scale the result by exactly three.
    """
    m, base, grid = _setup()
    scaled = PunctureField(
        orbit=base.orbit,
        h_s=lambda mm, r, th: 3.0 * stub_h_s(mm, r, th),
        s_eff=lambda mm, r, th: 3.0 * stub_s_eff(mm, r, th),
    )
    v1 = source_vector(m, base, grid)
    v3 = source_vector(m, scaled, grid)
    assert np.max(np.abs(v3 - 3.0 * v1)) < 1e-11 * max(1.0, float(np.max(np.abs(v1))))


def test_a_grid_without_a_worldtube_is_refused():
    orbit = Orbit(0.0, 6.0)
    grid = Grid.uniform(0.0, 0.5, np.pi / 9, -6.0, 6.0)
    try:
        source_vector(2, stub_puncture_field(orbit), grid)
    except ValueError as exc:
        assert "no worldtube" in str(exc)
        return
    raise AssertionError("expected a refusal for a grid with no worldtube")


def test_a_grid_built_for_a_different_spin_is_refused():
    orbit = Orbit(0.0, 6.0)
    other = Grid.worldtube(Orbit(0.6, 6.0), **WT)
    try:
        source_vector(2, stub_puncture_field(orbit), other)
    except ValueError as exc:
        assert "spin" in str(exc)
        return
    raise AssertionError("expected a refusal for mismatched spins")


def test_the_worldtube_never_lands_on_the_particle():
    """
    Every node the source vector touches -- the worldtube and its halo -- must
    be a point where the puncture field is finite, which means none of them
    may coincide with the particle.
    """
    orbit = Orbit(0.0, 6.0)
    grid = Grid.worldtube(orbit, **WT)
    imin, imax = grid.worldtube_rstar_indices
    jmin, jmax = grid.worldtube_theta_indices
    for i in range(imin - 1, imax + 2):
        for j in range(jmin - 1, jmax + 2):
            assert not (
                abs(grid.r[i] - orbit.r0) < 1e-10
                and abs(grid.theta[j] - 0.5 * np.pi) < 1e-10
            )
