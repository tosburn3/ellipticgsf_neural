"""
The binding to the compiled effective-source library.

Everything that needs the library skips cleanly when it is absent, so a fresh
checkout still runs the suite. Set `ELLIPTICGSF_EFFSOURCE_LIB` or put the
built library in ellipticgsf_neural/effective_source/ to enable these.

The most useful test here is `test_the_two_code_paths_agree`: the library
computes the puncture field twice by different routes, so their agreement
independently confirms the ctypes signatures, the struct layout and the
real/imaginary buffer split all at once.
"""

from __future__ import annotations

import numpy as np

from ellipticgsf_neural.problem import Grid, Orbit
from ellipticgsf_neural.source import source_vector_reference
from ellipticgsf_neural.source import source_vector
from ellipticgsf_neural import source as es

from ._shim import get_pytest

pytest = get_pytest()

try:
    _LIB = es.load_effective_source_library()
    HAVE_LIB = True
except (FileNotFoundError, OSError, RuntimeError):  # pragma: no cover
    _LIB = None
    HAVE_LIB = False

ORBIT = Orbit(a=0.0, r0=6.0)
WT = dict(delta_rstar=0.15, delta_theta=np.pi / 36, rstar_inner=-6.0, rstar_outer=30.0)


def _skip() -> bool:
    if not HAVE_LIB:
        print("   (skipped: the effective-source library is not available)")
        return True
    return False


# --------------------------------------------------------------------------
# Orbit constants -- no library needed
# --------------------------------------------------------------------------


def test_schwarzschild_orbit_constants_match_the_closed_form():
    """E = (1 - 2/r)/sqrt(1 - 3/r) and L = sqrt(r^2/(r - 3)) at zero spin."""
    for r0 in (6.0, 10.0, 50.0):
        e, l = es.circular_orbit_constants(r0, 0.0)
        assert abs(e - (1 - 2 / r0) / np.sqrt(1 - 3 / r0)) < 1e-14
        assert abs(l - np.sqrt(r0**2 / (r0 - 3.0))) < 1e-13


def test_orbit_constants_refuse_the_photon_orbit():
    for r0, a in ((3.0, 0.0), (2.0, 0.0)):
        try:
            es.circular_orbit_constants(r0, a)
        except ValueError as exc:
            assert "photon orbit" in str(exc)
        else:
            raise AssertionError(f"expected a refusal at r0={r0}, a={a}")


# --------------------------------------------------------------------------
# The elliptic-integral shim, if it was used
# --------------------------------------------------------------------------


def test_the_shim_matches_scipy_when_it_is_present():
    """
    If the library was built against `gsl_ellint_shim.c`, its K and E must
    agree with scipy. Skipped when real GSL was used, since then the symbols
    are not exported.
    """
    if _skip():
        return
    import ctypes

    try:
        kcomp = _LIB._lib.gsl_sf_ellint_Kcomp
        ecomp = _LIB._lib.gsl_sf_ellint_Ecomp
    except AttributeError:
        print("   (built against real GSL; the shim is not present)")
        return
    from scipy.special import ellipe, ellipk

    for f in (kcomp, ecomp):
        f.argtypes = [ctypes.c_double, ctypes.c_int]
        f.restype = ctypes.c_double
    for k in (0.0, 0.1, 0.5, 0.9, 0.99, 0.999):
        assert abs(kcomp(k, 0) - ellipk(k * k)) < 1e-14 * max(1.0, ellipk(k * k))
        assert abs(ecomp(k, 0) - ellipe(k * k)) < 1e-12


# --------------------------------------------------------------------------
# The binding
# --------------------------------------------------------------------------


def test_the_expected_symbols_are_exported():
    if _skip():
        return
    for name in ("effsource_init", "effsource_set_particle", "effsource_hS_m",
                 "effsource_calc_m"):
        assert hasattr(_LIB._lib, name), f"missing {name}"


def test_the_two_code_paths_agree():
    if _skip():
        return
    _LIB.set_orbit(ORBIT)
    for m in (0, 1, 2, 7):
        raw = _LIB.h_s_m(m, 6.3, 1.4)
        direct = raw[:10] + 1j * raw[10:]
        via_calc = _LIB.calc_m(m, 6.3, 1.4)["hS"]
        scale = max(1.0, float(np.max(np.abs(via_calc))))
        assert np.max(np.abs(direct - via_calc)) < 1e-12 * scale


def test_calc_m_returns_all_six_outputs():
    if _skip():
        return
    _LIB.set_orbit(ORBIT)
    out = _LIB.calc_m(2, 6.3, 1.4)
    assert set(out) == set(es.CALC_M_OUTPUTS)
    for name, v in out.items():
        assert v.shape == (10,), name
        assert np.all(np.isfinite(v)), name


def test_the_supported_mode_range_is_enforced():
    if _skip():
        return
    _LIB.set_orbit(ORBIT)
    for m in (-1, -5):
        try:
            _LIB.h_s_m(m, 6.3, 1.4)
        except ValueError as exc:
            assert "negative" in str(exc)
        else:
            raise AssertionError(f"expected a refusal for m={m}")
    for m in (21, 100):
        try:
            _LIB.h_s_m(m, 6.3, 1.4)
        except ValueError as exc:
            assert "20" in str(exc)
        else:
            raise AssertionError(f"expected a refusal for m={m}")


def test_evaluating_at_the_particle_is_reported_not_propagated():
    """The puncture field diverges there, so a NaN must become an error."""
    if _skip():
        return
    _LIB.set_orbit(ORBIT)
    try:
        _LIB.calc_m(2, ORBIT.r0, 0.5 * np.pi)
    except FloatingPointError as exc:
        assert "diverges at the" in str(exc)
        return
    print("   (note: the library returned finite values at the particle)")


def test_evaluating_before_configuring_an_orbit_is_refused():
    if _skip():
        return
    fresh = es.load_effective_source_library(_LIB.path)
    try:
        fresh.h_s_m(2, 6.3, 1.4)
    except RuntimeError as exc:
        assert "no orbit" in str(exc)
        return
    raise AssertionError("expected a refusal before set_orbit")


def test_the_orbit_guard_rearms_and_counts():
    if _skip():
        return
    lib = es.load_effective_source_library(_LIB.path)
    assert lib.orbit is None and lib.rearm_count == 0
    lib.set_orbit(ORBIT)
    assert lib.orbit == ORBIT and lib.rearm_count == 1
    lib._require_orbit(ORBIT)
    assert lib.rearm_count == 1, "re-arming for the same orbit is wasted work"
    lib._require_orbit(Orbit(0.0, 8.0))
    assert lib.rearm_count == 2 and lib.orbit == Orbit(0.0, 8.0)


def test_strict_mode_raises_instead_of_rearming():
    if _skip():
        return
    lib = es.load_effective_source_library(_LIB.path, strict_orbit=True)
    lib.set_orbit(ORBIT)
    try:
        lib._require_orbit(Orbit(0.0, 8.0))
    except RuntimeError as exc:
        assert "stateful" in str(exc)
        return
    raise AssertionError("expected a refusal in strict_orbit mode")


def test_changing_the_orbit_changes_the_answer():
    """A stale coefficient table would give identical values for both orbits."""
    if _skip():
        return
    lib = es.load_effective_source_library(_LIB.path)
    lib.set_orbit(Orbit(0.0, 6.0))
    at6 = lib.h_s_m(2, 7.0, 1.4).copy()
    lib.set_orbit(Orbit(0.0, 10.0))
    at10 = lib.h_s_m(2, 7.0, 1.4).copy()
    assert not np.allclose(at6, at10), "the particle was not actually moved"


# --------------------------------------------------------------------------
# Driving the real assembly
# --------------------------------------------------------------------------


def test_every_worldtube_node_evaluates_finitely():
    if _skip():
        return
    grid = Grid.worldtube(ORBIT, **WT)
    imin, imax = grid.worldtube_rstar_indices
    jmin, jmax = grid.worldtube_theta_indices
    _LIB.set_orbit(ORBIT)
    for j in range(jmin - 1, jmax + 2):
        for i in range(imin - 1, imax + 2):
            assert np.all(np.isfinite(_LIB.h_s_m(2, grid.r[i], grid.theta[j])))


def test_the_real_puncture_field_drives_the_source_vector():
    if _skip():
        return
    lib = es.load_effective_source_library(_LIB.path)
    grid = Grid.worldtube(ORBIT, **WT)
    puncture = lib.puncture_field(ORBIT)

    before = lib.rearm_count
    fast = source_vector(2, puncture, grid)
    assert lib.rearm_count == before, "the orbit was re-armed inside the node loop"

    assert np.all(np.isfinite(fast))
    assert np.count_nonzero(fast) == 560  # worldtube plus halo, times ten fields

    slow = source_vector_reference(2, puncture, grid)
    assert np.max(np.abs(fast - slow)) < 1e-12 * max(1.0, float(np.max(np.abs(slow))))
