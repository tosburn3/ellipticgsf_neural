"""
Coordinates, orbits and grids.

The tortoise coordinate is checked against its closed form where one exists
and against a round trip everywhere else; the grid constructors are checked
for the two properties the rest of the package depends on -- that the
worldtube faces land exactly on nodes, and that the particle does not.
"""

from __future__ import annotations

import numpy as np

from ellipticgsf_neural.problem import (
    Grid,
    Orbit,
    drstar_dr,
    horizon_radii,
    rstar_nodes,
    rstar_of_x,
    snap_worldtube,
    theta_nodes,
    x_of_rstar,
)

from ._shim import get_pytest

pytest = get_pytest()

WT = dict(delta_rstar=0.5, delta_theta=np.pi / 18, rstar_inner=-6.0, rstar_outer=8.0)


# --------------------------------------------------------------------------
# The tortoise coordinate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.0, 0.3, 0.9, -0.5])
def test_inversion_round_trips(a):
    rstar = np.linspace(-40.0, 400.0, 257)
    x = x_of_rstar(a, rstar)
    assert np.all(x > 0.0), "an iterate crossed the horizon"
    assert np.max(np.abs(rstar_of_x(a, x) - rstar)) < 1e-9


def test_schwarzschild_closed_form():
    r = np.array([2.5, 3.0, 6.0, 20.0, 1000.0])
    got = rstar_of_x(0.0, r - 2.0)
    want = r + 2.0 * np.log(r / 2.0 - 1.0)
    assert np.max(np.abs(got - want)) < 1e-12


def test_derivative_matches_finite_difference():
    a, x0, h = 0.4, 3.0, 1e-6
    num = (rstar_of_x(a, x0 + h) - rstar_of_x(a, x0 - h)) / (2 * h)
    assert abs(num - drstar_dr(a, x0)) < 1e-6


def test_inversion_survives_deep_near_horizon_values():
    """
    At r* = -60 the radius differs from r_+ by about 1e-14, which is the whole
    reason x is the working variable.
    """
    x = x_of_rstar(0.6, np.array([-60.0, -40.0, -20.0]))
    assert np.all(x > 0.0)
    assert np.max(np.abs(rstar_of_x(0.6, x) - np.array([-60.0, -40.0, -20.0]))) < 1e-8


def test_extremal_spin_is_refused():
    try:
        rstar_of_x(1.0, np.array([1.0]))
    except ValueError as exc:
        assert "extremal" in str(exc)
        return
    raise AssertionError("expected a refusal at |a| = 1")


# --------------------------------------------------------------------------
# Orbit
# --------------------------------------------------------------------------


def test_schwarzschild_frequency_is_keplerian():
    for r0 in (6.0, 10.0, 100.0):
        assert abs(Orbit(0.0, r0).omega - r0**-1.5) < 1e-15


def test_frequency_matches_the_standard_kerr_form():
    """Omega = 1 / (a + r0^(3/2)) with M = 1."""
    for a, r0 in ((0.0, 6.0), (0.6, 8.0), (-0.5, 12.0)):
        assert abs(Orbit(a, r0).omega - 1.0 / (a + r0**1.5)) < 1e-15


def test_mode_frequency_is_m_times_omega():
    orbit = Orbit(0.6, 8.0)
    for m in (0, 1, 5):
        assert orbit.mode_frequency(m) == m * orbit.omega


def test_orbit_rejects_impossible_parameters():
    for a, r0 in ((1.0, 6.0), (-1.2, 6.0), (0.0, -3.0)):
        try:
            Orbit(a, r0)
        except ValueError:
            continue
        raise AssertionError(f"expected a refusal for a={a}, r0={r0}")


def test_orbit_is_immutable():
    orbit = Orbit(0.0, 6.0)
    try:
        orbit.r0 = 7.0
    except Exception:
        return
    raise AssertionError("Orbit should be frozen")


# --------------------------------------------------------------------------
# Worldtube alignment
# --------------------------------------------------------------------------


def test_snapping_places_the_domain_in_whole_worldtube_widths():
    orbit = Orbit(0.0, 6.0)
    span = snap_worldtube(orbit, 1.0, -6.0, 8.0)
    assert span.inner < span.left < span.center < span.right < span.outer
    assert abs(span.diameter - 1.0) < 1e-12
    for edge in (span.outer - span.right, span.left - span.inner):
        assert abs(edge / 1.0 - round(edge / 1.0)) < 1e-12
    # Snapping moves edges outward, never inward.
    assert span.outer >= 8.0 and span.inner <= -6.0


def test_snapping_refuses_a_domain_that_misses_the_worldtube():
    orbit = Orbit(0.0, 6.0)
    try:
        # Both requested edges sit inside the worldtube.
        snap_worldtube(orbit, 1.0, orbit.rstar0 + 0.1, orbit.rstar0 - 0.1)
    except ValueError as exc:
        assert "worldtube" in str(exc)
        return
    raise AssertionError("expected a refusal for an inverted domain")


def test_worldtube_faces_land_on_nodes():
    orbit = Orbit(0.0, 6.0)
    span = snap_worldtube(orbit, 1.0, -6.0, 8.0)
    nodes, (imin, imax) = rstar_nodes(span, 0.5)
    assert abs(nodes[imin] - span.left) < 1e-10
    assert abs(nodes[imax] - span.right) < 1e-10


def test_even_node_counts_keep_the_particle_between_nodes():
    """
    The puncture field diverges at the particle, so no node may land on it.
    That is what the even point counts across the worldtube and the source
    region are for, in both directions.
    """
    orbit = Orbit(0.0, 6.0)
    grid = Grid.worldtube(orbit, **WT)
    imin, imax = grid.worldtube_rstar_indices
    jmin, jmax = grid.worldtube_theta_indices
    assert (imax - imin + 1) % 2 == 0, "odd node count across the worldtube"
    assert (jmax - jmin + 1) % 2 == 0, "odd node count across the source region"
    assert np.min(np.abs(grid.r - orbit.r0)) > 1e-6
    assert np.min(np.abs(grid.theta - 0.5 * np.pi)) > 1e-6


def test_source_region_is_centred_on_the_equator():
    theta, (jmin, jmax) = theta_nodes(np.pi / 18, 0.4)
    mid = 0.5 * (theta[jmin] + theta[jmax])
    assert abs(mid - 0.5 * np.pi) < 1e-12
    assert theta[0] == 0.0


def test_the_requested_spacing_is_quantized_not_honoured():
    """
    A surprise worth pinning: the node count across the worldtube is rounded
    to an even number, so `drstar` is a request. Two different requests can
    give the identical grid, and the realized spacing has to be read back off
    the grid rather than assumed.
    """
    span = snap_worldtube(Orbit(0.0, 6.0), 1.0, -6.0, 8.0)
    coarse, _ = rstar_nodes(span, 0.5)
    finer, _ = rstar_nodes(span, 0.25)
    assert len(coarse) == len(finer)
    assert np.max(np.abs(coarse - finer)) == 0.0

    # A genuinely finer request does refine it.
    much_finer, _ = rstar_nodes(span, 0.1)
    assert len(much_finer) > len(coarse)


def test_an_absurdly_coarse_spacing_degrades_to_the_minimum():
    """Two nodes across the worldtube is the floor, not an error here."""
    span = snap_worldtube(Orbit(0.0, 6.0), 1.0, -6.0, 8.0)
    nodes, (imin, imax) = rstar_nodes(span, 100.0)
    assert imax - imin == 1
    theta, (jmin, jmax) = theta_nodes(10.0, 0.4)
    assert jmax - jmin == 1


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------


def test_grid_stores_all_three_radial_parameterizations_consistently():
    grid = Grid.worldtube(
        Orbit(0.6, 8.0), delta_rstar=0.5, delta_theta=np.pi / 18,
        rstar_inner=-6.0, rstar_outer=30.0,
    )
    rp = horizon_radii(0.6)[0]
    assert np.max(np.abs(grid.r - (grid.x + rp))) == 0.0
    assert np.max(np.abs(rstar_of_x(0.6, grid.x) - grid.rstar)) < 1e-10


def test_grid_spacings_come_from_the_endpoints():
    grid = Grid.worldtube(Orbit(0.0, 6.0), **WT)
    assert abs(grid.delta_rstar - (grid.rstar[-1] - grid.rstar[0]) / (grid.n_rstar - 1)) == 0.0
    assert abs(grid.delta_theta - (grid.theta[-1] - grid.theta[0]) / (grid.n_theta - 1)) == 0.0
    # Uniform, so every gap equals the stored spacing.
    assert np.max(np.abs(np.diff(grid.rstar) - grid.delta_rstar)) < 1e-12
    assert np.max(np.abs(np.diff(grid.theta) - grid.delta_theta)) < 1e-12


def test_supplied_x_skips_the_inversion_and_must_match_in_length():
    grid = Grid.worldtube(Orbit(0.0, 6.0), **WT)
    again = Grid(0.0, grid.rstar, grid.theta, x=grid.x)
    assert np.max(np.abs(again.x - grid.x)) == 0.0
    try:
        Grid(0.0, grid.rstar, grid.theta, x=grid.x[:-1])
    except ValueError:
        return
    raise AssertionError("expected a refusal for an x of the wrong length")


def test_uniform_grid_has_no_worldtube_and_says_so():
    grid = Grid.uniform(0.0, 0.5, np.pi / 9, -6.0, 6.0)
    assert grid.worldtube_rstar_indices is None and grid.worldtube_theta_indices is None
    try:
        grid.require_worldtube()
    except ValueError as exc:
        assert "no worldtube" in str(exc)
        return
    raise AssertionError("expected a refusal from require_worldtube")


def test_require_worldtube_rejects_bounds_with_no_room_for_the_halo():
    grid = Grid.worldtube(Orbit(0.0, 6.0), **WT)
    grid.worldtube_rstar_indices = (0, 3)  # touching the inner edge
    try:
        grid.require_worldtube()
    except ValueError as exc:
        assert "jump halo" in str(exc)
        return
    raise AssertionError("expected a refusal for a worldtube against the edge")


def test_uniform_grid_covers_the_requested_range():
    grid = Grid.uniform(0.0, 0.5, np.pi / 9, -6.0, 6.0)
    assert grid.rstar[0] == -6.0
    assert grid.rstar[-1] <= 6.0 + 1e-12
    assert grid.theta[0] == 0.0
    assert grid.theta[-1] <= np.pi + 1e-12
    assert grid.size(10) == 10 * grid.n_rstar * grid.n_theta


# --------------------------------------------------------------------------
# GridParameters and make_grid
# --------------------------------------------------------------------------


def _parameters(**changes):
    from dataclasses import replace

    from ellipticgsf_neural.problem import GridParameters
    base = GridParameters(delta_rstar=7 / 33, rstar_min=-4.0, rstar_max=73.0, delta_theta=np.pi / 45,
                          worldtube_rstar_width=7.0, worldtube_theta_width=np.pi / 5)
    return replace(base, **changes)


def test_make_grid_snaps_to_whole_worldtube_widths():
    from ellipticgsf_neural.problem import make_grid
    orbit = Orbit(0.0, 8.5)
    grid = make_grid(orbit, _parameters())
    (i0, i1), (j0, j1) = grid.worldtube_rstar_indices, grid.worldtube_theta_indices
    assert np.isclose(grid.rstar[i1] - grid.rstar[i0], 7.0) and np.isclose(grid.theta[j1] - grid.theta[j0], np.pi / 5)
    assert np.isclose((grid.rstar[-1] - grid.rstar[i1]) / 7.0, round((grid.rstar[-1] - grid.rstar[i1]) / 7.0))
    assert grid.rstar[0] <= -4.0 and grid.rstar[-1] >= 73.0


def test_make_grid_follows_finer_requests():
    from ellipticgsf_neural.problem import make_grid
    orbit = Orbit(0.0, 8.5)
    base = make_grid(orbit, _parameters())
    fine = make_grid(orbit, _parameters(delta_rstar=7 / 47, delta_theta=np.pi / 65))
    assert fine.n_theta > base.n_theta and fine.n_rstar > base.n_rstar
    assert fine.worldtube_rstar_indices[1] > base.worldtube_rstar_indices[1]
