"""
The source vector b of M x = b (Sec. III): the puncture field, the effective source, and the
worldtube jump terms (the effective-source method, Sec. II), plus the binding to the compiled
effective-source library that evaluates them.

WHAT THE RIGHT-HAND SIDE IS
---------------------------
A point particle's field diverges at the particle, and what the self-force
needs is the finite residual left after the divergent piece is removed. The
effective-source approach does the removal on a finite region: a worldtube
around the particle.

Inside the worldtube the code solves for the *residual* field, with the
effective source on the right-hand side. Outside it solves for the full
retarded field, with nothing on the right-hand side. The two unknowns differ
by the puncture field, so wherever a finite-difference stencil straddles the
worldtube boundary it mixes them, and the mismatch has to be moved to the
right-hand side. That correction is the stencil weight times the puncture
field at the node on the other side of the face -- a jump term.

So the vector has three parts:

  1. the effective source at every node strictly inside the worldtube,
  2. polar jump terms on the worldtube's two theta faces and on the single
     row of nodes just outside each,
  3. radial jump terms on the two radial faces, likewise.

In a jump term the coefficient matrix is evaluated at the row's own node and
the puncture field at the neighbour across the face. Everything else in the
vector is zero, so its footprint is the worldtube plus a one-node halo.

WHERE THE NUMBERS COME FROM
---------------------------
`PunctureField` wraps two routines that return the raw puncture field and the
raw effective source at a point, as 2n floats (n real parts then n imaginary
parts). This module turns those raw values into the mode quantities the
assembly needs, applying the per-field scaling and the azimuthal phase.

The raw values themselves come from an external C library, which the second
half of this file binds -- see `load_effective_source_library`. Anything with
the same two-callable signature will do instead, which is how the tests drive
the whole assembly with no library present.
"""

from __future__ import annotations

import ctypes
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .assemble import first_rstar_derivative_coefficients, first_theta_derivative_coefficients
from .problem import N_COMPONENTS, Grid, Orbit, horizon_radii

__all__ = [
    # the puncture field and the right-hand side
    "PunctureField",
    "azimuthal_shift",
    "source_vector",
    "source_vector_reference",
    # the compiled library behind them
    "MAX_M",
    "Coordinate",
    "EffectiveSourceLibrary",
    "circular_orbit_constants",
    "find_effective_source_library",
    "load_effective_source_library",
]


# --------------------------------------------------------------------------
# Field-dependent factors
#
# Machine-emitted from the symbolic derivation, like the coefficients in assemble.py: the
# scaling that turns the raw puncture field into the mode variables, and the
# transformation applied to the raw effective source. Do not edit by hand.
# --------------------------------------------------------------------------


def _csc(x):
    return 1.0 / np.sin(x)


def _psi_scale(m, a, r, theta):
    """Per-field scaling of the puncture field. Scalar r, theta."""
    r = float(r)
    theta = float(theta)
    return np.array([1, (r**2-2*r+a**2)/r**2, 1/r,
    1/(r*np.sin(theta)), ((r**2-2*r+a**2)/r**2)**2, ((r**2-2*r+a**2)/r**2)/r, ((r**2-2*r+a**2)/r**2)/(r*np.sin(theta)), 1/r**2, 1/(r**2*np.sin(theta)), 1/(r*np.sin(theta))**2], dtype=complex)


def _source_matrix(m, a, r, theta):
    """Transformation applied to the raw effective source. Scalar r, theta."""
    r = float(r)
    theta = float(theta)
    return np.array([[-((r*(a**2 + (-2 + r)*r)*(r**2 + a**2*np.cos(theta)**2))/(a**2 + r**2)**2), 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -1/2*((a**2 + (-2 + r)*r)**2*(a**2 + 2*r**2 + a**2*np.cos(2*theta)))/(r*(a**2 + r**2)**2), 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, -(((a**2 + (-2 + r)*r)*(r**2 + a**2*np.cos(theta)**2))/(a**2 + r**2)**2), 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -(((a**2 + (-2 + r)*r)*(r**2 + a**2*np.cos(theta)**2)*_csc(theta))/(a**2 + r**2)**2), 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, -1/2*((a**2 + (-2 + r)*r)**3*(a**2 + 2*r**2 + a**2*np.cos(2*theta)))/(r**3*(a**2 + r**2)**2), 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1/2*((a**2 + (-2 + r)*r)**2*(a**2 + 2*r**2 + a**2*np.cos(2*theta)))/(r**2*(a**2 + r**2)**2), 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, -(((a**2 + (-2 + r)*r)**2*(r**2 + a**2*np.cos(theta)**2)*_csc(theta))/(r**2*(a**2 + r**2)**2)), 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -(((a**2 + (-2 + r)*r)*(r**2 + a**2*np.cos(theta)**2))/(r*(a**2 + r**2)**2)), 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, -(((a**2 + (-2 + r)*r)*(r**2 + a**2*np.cos(theta)**2)*_csc(theta))/(r*(a**2 + r**2)**2)), 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, -(((a**2 + (-2 + r)*r)*(r**2 + a**2*np.cos(theta)**2)*_csc(theta)**2)/(r*(a**2 + r**2)**2))]], dtype=complex)


# --------------------------------------------------------------------------
# The puncture field
# --------------------------------------------------------------------------


def azimuthal_shift(a: float, x) -> float:
    """
    The Kerr azimuthal twist accumulated on the way in from infinity,

        Delta_phi = a / (r_+ - r_-) * log( x / (r - r_-) ),   x = r - r_+

    which is what makes the corotating azimuthal coordinate regular on the
    horizon. Zero for a = 0.
    """
    if a == 0.0:
        return 0.0
    rp, rm = horizon_radii(a)
    r = np.asarray(x) + rp
    return a / (rp - rm) * np.log(np.asarray(x) / (r - rm))


@dataclass
class PunctureField:
    """
    The puncture field and effective source for one orbit.

    `h_s` and `s_eff` are the two point evaluators, each called as
    `f(m, r, theta)` and returning 2n floats: the n real parts followed by the
    n imaginary parts. `EffectiveSourceLibrary.puncture_field` below builds
    one of these from the compiled library; anything with that signature will
    do.

    Both diverge AT the particle, so neither may be called at exactly
    (r0, pi/2). Grids from `Grid.worldtube` straddle the particle rather than
    landing on it, which is what keeps that from happening.
    """

    orbit: Orbit
    h_s: Callable
    s_eff: Callable
    n: int = N_COMPONENTS

    def _phase(self, m: int, r: float) -> complex:
        rp = self.orbit.horizon
        return np.exp(-1j * m * azimuthal_shift(self.orbit.a, r - rp)) / (2.0 * np.pi)

    def _to_complex(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=float).ravel()
        if v.size != 2 * self.n:
            raise ValueError(
                f"a point evaluator returned {v.size} values; expected {2 * self.n} "
                f"({self.n} real parts followed by {self.n} imaginary parts)"
            )
        return v[: self.n] + 1j * v[self.n :]

    def psi(self, m: int, r: float, theta: float) -> np.ndarray:
        """The m-mode puncture field at (r, theta), as an n-vector."""
        raw = self._to_complex(self.h_s(m, float(r), float(theta)))
        return _psi_scale(m, self.orbit.a, r, theta) * (r * raw * self._phase(m, r))

    def source(self, m: int, r: float, theta: float) -> np.ndarray:
        """
        The m-mode effective source at (r, theta), as an n-vector.

        Same phase as `psi` but no factor of r, then the field-mixing
        transformation. That transformation is diagonal for this system, but
        it is applied as a full matrix product so a non-diagonal one would
        need no change here.
        """
        raw = self._to_complex(self.s_eff(m, float(r), float(theta)))
        return _source_matrix(m, self.orbit.a, r, theta) @ (raw * self._phase(m, r))


# --------------------------------------------------------------------------
# The right-hand side
# --------------------------------------------------------------------------


def source_vector(m: int, puncture: PunctureField, grid: Grid) -> np.ndarray:
    """
    Assemble the right-hand side for mode `m`.

    The grid must have been built by `Grid.worldtube`. The cost is dominated
    by the point evaluations inside `puncture`, one per node of the worldtube
    and its halo, so this is written as a plain loop: vectorizing the
    arithmetic around those calls would buy nothing until the calls
    themselves are batched.
    """
    orbit = puncture.orbit
    if grid.a != orbit.a:
        raise ValueError(
            f"grid was built for spin a = {grid.a} but the orbit has a = {orbit.a}"
        )
    (imin, imax), (jmin, jmax) = grid.require_worldtube()

    n = puncture.n
    nr, nth = grid.n_rstar, grid.n_theta
    a = orbit.a
    drs, dth = grid.delta_rstar, grid.delta_theta
    omega = orbit.mode_frequency(m)
    eye = np.eye(n)

    vec = np.zeros(n * nr * nth, dtype=complex)

    def node(i: int, j: int) -> int:
        """Start of the n contiguous entries at node (i, j); l is the fast index."""
        return n * (i + nr * j)

    def polar_coeff(r: float) -> float:
        return (r**2 - 2.0 * r + a * a) / (r * r + a * a) ** 2

    def q_rstar_at(r: float, th: float) -> np.ndarray:
        return first_rstar_derivative_coefficients(m, a, omega, np.array([r]), th)[:, :, 0]

    def q_theta_at(r: float, th: float) -> np.ndarray:
        return first_theta_derivative_coefficients(m, a, omega, np.array([r]), th)[:, :, 0]

    # 1. The effective source strictly inside the worldtube. This assigns
    #    rather than accumulates, so it has to come before the jump terms.
    for j in range(jmin, jmax + 1):
        th = grid.theta[j]
        for i in range(imin, imax + 1):
            k = node(i, j)
            vec[k : k + n] = puncture.source(m, grid.r[i], th)

    # 2. Polar jump terms. Each tuple is (row node, neighbour across the face,
    #    first-derivative weight, second-derivative weight, sign). The sign is
    #    positive on the inside rows, where the puncture field is being added
    #    back, and negative on the halo rows just outside, where it is being
    #    taken away.
    for i in range(imin, imax + 1):
        r = grid.r[i]
        cth = polar_coeff(r)
        for row_j, nbr_j, w1, w2, sign in (
            (jmin, jmin - 1, -1.0 / (2.0 * dth), 1.0 / dth**2, +1.0),
            (jmax, jmax + 1, +1.0 / (2.0 * dth), 1.0 / dth**2, +1.0),
            (jmin - 1, jmin, +1.0 / (2.0 * dth), 1.0 / dth**2, -1.0),
            (jmax + 1, jmax, -1.0 / (2.0 * dth), 1.0 / dth**2, -1.0),
        ):
            op = w1 * q_theta_at(r, grid.theta[row_j]) + w2 * cth * eye
            k = node(i, row_j)
            vec[k : k + n] += sign * (op @ puncture.psi(m, r, grid.theta[nbr_j]))

    # 3. Radial jump terms, the same pattern in the other direction. The
    #    radial second derivative has no coefficient, hence the bare identity.
    for j in range(jmin, jmax + 1):
        th = grid.theta[j]
        for row_i, nbr_i, w1, w2, sign in (
            (imin, imin - 1, -1.0 / (2.0 * drs), 1.0 / drs**2, +1.0),
            (imax, imax + 1, +1.0 / (2.0 * drs), 1.0 / drs**2, +1.0),
            (imin - 1, imin, +1.0 / (2.0 * drs), 1.0 / drs**2, -1.0),
            (imax + 1, imax, -1.0 / (2.0 * drs), 1.0 / drs**2, -1.0),
        ):
            op = w1 * q_rstar_at(grid.r[row_i], th) + w2 * eye
            k = node(row_i, j)
            vec[k : k + n] += sign * (op @ puncture.psi(m, grid.r[nbr_i], th))

    return vec


def source_vector_reference(
    m: int, puncture: PunctureField, grid: Grid
) -> np.ndarray:
    """
    Build b entry by entry. Far too slow for real grids; on a small grid it is
    an independent implementation to check `source_vector` against
    (tests/test_source.py). Keep it dumb: no vectorization, no shared helpers.

    The twelve blocks are spelled out separately rather than folded into
    loops, which is what makes an exact match rule out a mistake in that
    folding -- the signs and the which-node-gets-which-argument pairings are
    exactly where this assembly is easy to get wrong.
    """
    orbit = puncture.orbit
    (imin, imax), (jmin, jmax) = grid.require_worldtube()

    n = puncture.n
    nr, nth = grid.n_rstar, grid.n_theta
    a = orbit.a
    drs, dth = grid.delta_rstar, grid.delta_theta
    omega = orbit.mode_frequency(m)
    eye = np.eye(n)
    vec = np.zeros(n * nr * nth, dtype=complex)
    r, theta = grid.r, grid.theta

    def idx(l, i, j):
        return l + n * (i + nr * j)

    def q_theta(rr, th):
        return first_theta_derivative_coefficients(m, a, omega, np.array([rr]), th)[:, :, 0]

    def q_rstar(rr, th):
        return first_rstar_derivative_coefficients(m, a, omega, np.array([rr]), th)[:, :, 0]

    def ct(rr):
        return (rr**2 - 2.0 * rr + a * a) / (rr * rr + a * a) ** 2

    # 1. The effective source inside the worldtube.
    for j in range(jmin, jmax + 1):
        for i in range(imin, imax + 1):
            data = puncture.source(m, r[i], theta[j])
            for l in range(n):
                vec[idx(l, i, j)] = data[l]

    # 2. The four polar jump blocks.
    for i in range(imin, imax + 1):
        rr = r[i]

        data = (-1 / (2 * dth) * q_theta(rr, theta[jmin]) + 1 / dth**2 * ct(rr) * eye) @ \
            puncture.psi(m, rr, theta[jmin - 1])
        for l in range(n):
            vec[idx(l, i, jmin)] += data[l]

        data = (1 / (2 * dth) * q_theta(rr, theta[jmax]) + 1 / dth**2 * ct(rr) * eye) @ \
            puncture.psi(m, rr, theta[jmax + 1])
        for l in range(n):
            vec[idx(l, i, jmax)] += data[l]

        data = (1 / (2 * dth) * q_theta(rr, theta[jmin - 1]) + 1 / dth**2 * ct(rr) * eye) @ \
            puncture.psi(m, rr, theta[jmin])
        for l in range(n):
            vec[idx(l, i, jmin - 1)] -= data[l]

        data = (-1 / (2 * dth) * q_theta(rr, theta[jmax + 1]) + 1 / dth**2 * ct(rr) * eye) @ \
            puncture.psi(m, rr, theta[jmax])
        for l in range(n):
            vec[idx(l, i, jmax + 1)] -= data[l]

    # 3. The four radial jump blocks.
    for j in range(jmin, jmax + 1):
        th = theta[j]

        data = (-1 / (2 * drs) * q_rstar(r[imin], th) + 1 / drs**2 * eye) @ \
            puncture.psi(m, r[imin - 1], th)
        for l in range(n):
            vec[idx(l, imin, j)] += data[l]

        data = (1 / (2 * drs) * q_rstar(r[imax], th) + 1 / drs**2 * eye) @ \
            puncture.psi(m, r[imax + 1], th)
        for l in range(n):
            vec[idx(l, imax, j)] += data[l]

        data = (1 / (2 * drs) * q_rstar(r[imin - 1], th) + 1 / drs**2 * eye) @ \
            puncture.psi(m, r[imin], th)
        for l in range(n):
            vec[idx(l, imin - 1, j)] -= data[l]

        data = (-1 / (2 * drs) * q_rstar(r[imax + 1], th) + 1 / drs**2 * eye) @ \
            puncture.psi(m, r[imax], th)
        for l in range(n):
            vec[idx(l, imax + 1, j)] -= data[l]

    return vec


# --------------------------------------------------------------------------
# The compiled effective-source library
#
# The puncture field and the effective source are evaluated by an external C
# library (Barry Wardell's `GravitationalEffectiveSource`). Its two entry
# points are plain C with `double *` out-parameters, so ctypes is enough --
# no extension module, and no build step on the Python side:
#
#     effsource_hS_m     the m-mode puncture field
#     effsource_calc_m   the m-mode effective source, plus derivatives
#
# BUILDING IT
# -----------
# Compile `effective_source/kerr-circular.c`, NOT `kerr-circular.cxx`. The two have diverged:
# the `.c` exposes the library API above, while the `.cxx` rewrote the same
# functions to stream their results to an external front end and added a
# `WinMain`, so a build of it returns nothing usable through ctypes.
#
#     # Linux / macOS / WSL, with GSL installed
#     gcc -O3 -fPIC -shared -o libeffsource.so kerr-circular.c \
#         -lgsl -lgslcblas -lm
#
#     # Windows, via vcpkg
#     cl /LD /O2 /I C:/DEV/vcpkg/installed/x64-windows/include kerr-circular.c \
#        C:/DEV/vcpkg/installed/x64-windows/lib/gsl.lib /Fe:effsource.dll
#
# Only two GSL functions are used, the complete elliptic integrals K and E. If
# GSL is awkward to get hold of, `effective_source/gsl_ellint_shim.c` supplies
# both by the arithmetic-geometric mean; it agrees with GSL to near machine
# precision, but link against the real thing for published results.
#
# WHAT THE WRAPPER GUARDS
# -----------------------
# Three sharp edges in the library, each of which would otherwise give
# silently wrong numbers rather than an error:
#
# 1. `effsource_set_particle` is STATEFUL. It precomputes a coefficient table
#    for one orbit, so it must be re-called whenever r0 or a changes. Every
#    evaluation here checks the configured orbit first and re-arms if needed,
#    counting how often that happens -- re-arming inside a per-node loop would
#    be ruinously slow, so a high count is a signal rather than a detail.
#
# 2. `m > 20` is UNSUPPORTED. The internal tables are dimensioned for
#    m = 0..20; beyond that the C code prints a message and returns WITHOUT
#    WRITING the output arrays, leaving whatever was in the buffer. Checked
#    before the call, and the buffers are zeroed regardless.
#
# 3. Negative m would index those tables out of bounds, which is undefined
#    behaviour rather than an error. Also refused; get negative modes from the
#    conjugate symmetry instead.
#
# A fourth is physics, not a bug: the puncture field diverges at the particle,
# so evaluating at exactly (r0, pi/2) gives non-finite values. That is
# reported with the particle named as the first suspect. `Grid.worldtube`
# never puts a node there.
# --------------------------------------------------------------------------


#: Highest mode number the library's internal tables cover.
MAX_M = 20


#: The six (re, im) pairs `effsource_calc_m` fills, in argument order. Only
#: `src` is needed for the source vector, but the others cost nothing extra.
CALC_M_OUTPUTS = ("hS", "dhS_dr", "dhS_dth", "dhS_dph", "dhS_dt", "src")


class Coordinate(ctypes.Structure):
    """Mirrors `struct coordinate` in the library's header."""

    _fields_ = [
        ("t", ctypes.c_double),
        ("r", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("phi", ctypes.c_double),
    ]


def circular_orbit_constants(r0: float, a: float) -> tuple[float, float]:
    """
    Energy and angular momentum of a circular equatorial Kerr orbit, M = 1:

        v = 1/sqrt(r0)
        E = (1 - 2v^2 + a v^3) / sqrt(1 - 3v^2 + 2a v^3)
        L = r0 v (1 - 2a v^3 + a^2 v^4) / sqrt(1 - 3v^2 + 2a v^3)

    The library takes these as inputs rather than deriving them, so they are
    computed here to exactly the same form its own front end uses.
    """
    v = 1.0 / np.sqrt(r0)
    # Check the discriminant before rooting it, so an impossible orbit is an
    # error rather than a warning followed by a NaN.
    disc = 1.0 - 3.0 * v**2 + 2.0 * a * v**3
    if not np.isfinite(disc) or disc <= 0.0:
        raise ValueError(
            f"r0 = {r0} with a = {a} is at or inside the photon orbit "
            f"(1 - 3v^2 + 2a v^3 = {disc:.6g} <= 0), so no circular orbit exists there"
        )
    denom = np.sqrt(disc)
    energy = (1.0 - 2.0 * v**2 + a * v**3) / denom
    ang_mom = r0 * v * (1.0 - 2.0 * a * v**3 + a**2 * v**4) / denom
    return float(energy), float(ang_mom)


def find_effective_source_library(explicit: str | os.PathLike | None = None) -> Path:
    """
    Locate the compiled library.

    Search order: the explicit argument, then `$ELLIPTICGSF_EFFSOURCE_LIB`,
    then the usual file names in this package's folder, in its
    effective_source/ folder (where the C sources are), and in the working
    directory.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("ELLIPTICGSF_EFFSOURCE_LIB")
    if env:
        candidates.append(Path(env))

    names = ("libeffsource.so", "effsource.dll", "libeffsource.dylib", "effsource.so")
    package = Path(__file__).resolve().parent
    for directory in (package, package / "effective_source", Path.cwd()):
        candidates.extend(directory / name for name in names)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    looked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "could not find the compiled effective-source library.\nLooked in:\n  "
        + looked
        + "\n\nBuild it with, for example:\n"
        "  gcc -O3 -fPIC -shared -o libeffsource.so kerr-circular.c -lgsl -lgslcblas -lm\n"
        "then pass its path, or set ELLIPTICGSF_EFFSOURCE_LIB. See the comment "
        "above this function for the Windows recipe, and use kerr-circular.c, not the .cxx."
    )


class EffectiveSourceLibrary:
    """
    A loaded effective-source library, configured for one orbit at a time.

    Use `load()` for the common case. `rearm_count` reports how often the
    orbit had to be reconfigured; in a normal run it is 1.
    """

    def __init__(self, path: str | os.PathLike | None = None, *, strict_orbit: bool = False):
        self.path = find_effective_source_library(path)
        self._lib = self._open(self.path)
        self.strict_orbit = strict_orbit
        self._orbit: Orbit | None = None
        self.rearm_count = 0
        self._warned_rearm = False
        self._bind()

    def _open(self, path: Path) -> ctypes.CDLL:
        """
        Load the shared library, with the Windows dependency trap handled.

        Windows does not look beside a DLL for the DLLs it depends on, so a
        library built against a dynamic GSL fails to load with a message that
        blames the library itself rather than the missing dependency. Adding
        its own folder to the search path lets sibling DLLs -- `gsl.dll`,
        `gslcblas.dll` -- be found. The handle is kept for this object's
        lifetime so the entry stays in place.
        """
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            self._dll_directory = os.add_dll_directory(str(path.parent))
        try:
            return ctypes.CDLL(str(path))
        except OSError as exc:
            raise OSError(
                f"{path} exists but could not be loaded ({exc}).\n"
                "On Windows this usually means a DLL it depends on is missing "
                "rather than the library itself: a build against GSL needs "
                "gsl.dll and gslcblas.dll, which are found automatically only "
                "if they sit in the same folder as this library.\n"
                "Building with the bundled gsl_ellint_shim.c instead leaves the "
                "library with no dependencies at all."
            ) from None

    def _bind(self) -> None:
        dp = ctypes.POINTER(ctypes.c_double)
        cp = ctypes.POINTER(Coordinate)
        lib = self._lib

        try:
            lib.effsource_init.argtypes = [ctypes.c_double, ctypes.c_double]
            lib.effsource_init.restype = None

            lib.effsource_set_particle.argtypes = [cp, ctypes.c_double, ctypes.c_double,
                                                   ctypes.c_double]
            lib.effsource_set_particle.restype = None

            lib.effsource_hS_m.argtypes = [ctypes.c_int, cp, dp, dp]
            lib.effsource_hS_m.restype = None

            lib.effsource_calc_m.argtypes = [ctypes.c_int, cp] + [dp] * 12
            lib.effsource_calc_m.restype = None
        except AttributeError as exc:
            raise RuntimeError(
                f"{self.path} does not export the expected effective-source symbols "
                f"({exc}). Did you build kerr-circular.c rather than the .cxx?"
            ) from None

    # -- orbit state ------------------------------------------------------

    @property
    def orbit(self) -> Orbit | None:
        """The orbit currently configured, or None."""
        return self._orbit

    def set_orbit(self, orbit: Orbit) -> None:
        """
        Configure the library for one circular orbit.

        The order matters: `effsource_init` must run first, because it sets
        the spin that the rest of the setup reads.
        """
        energy, ang_mom = circular_orbit_constants(orbit.r0, orbit.a)
        self._lib.effsource_init(ctypes.c_double(1.0), ctypes.c_double(orbit.a))
        particle = Coordinate(t=0.0, r=float(orbit.r0), theta=0.5 * np.pi, phi=0.0)
        self._lib.effsource_set_particle(
            ctypes.byref(particle),
            ctypes.c_double(energy),
            ctypes.c_double(ang_mom),
            ctypes.c_double(0.0),
        )
        self._orbit = orbit
        self.rearm_count += 1
        if self.rearm_count > 8 and not self._warned_rearm:
            self._warned_rearm = True
            warnings.warn(
                f"the effective-source orbit has been reconfigured "
                f"{self.rearm_count} times. effsource_set_particle rebuilds the "
                "whole coefficient table, so interleaving orbits is very slow; "
                "finish one orbit before starting the next.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _require_orbit(self, orbit: Orbit) -> None:
        if self._orbit == orbit:
            return
        if self._orbit is not None and self.strict_orbit:
            raise RuntimeError(
                f"library is configured for {self._orbit} but {orbit} was requested. "
                "effsource_set_particle is stateful; call set_orbit explicitly, or "
                "construct with strict_orbit=False to re-arm automatically."
            )
        self.set_orbit(orbit)

    def particle_distance(self, r: float, theta: float) -> float:
        """
        Coordinate distance from an evaluation point to the particle.

        Useful for confirming a grid keeps clear of the singularity; the
        worldtube grids are built so this is never zero.
        """
        if self._orbit is None:
            raise RuntimeError("no orbit configured; call set_orbit(orbit) first")
        r0 = self._orbit.r0
        return float(np.hypot(r - r0, r0 * (theta - 0.5 * np.pi)))

    def _check_finite(self, values: np.ndarray, m: int, r: float, theta: float) -> None:
        if np.all(np.isfinite(values)):
            return
        dist = self.particle_distance(r, theta)
        raise FloatingPointError(
            f"the effective source returned non-finite values at m = {m}, "
            f"r = {r!r}, theta = {theta!r}. The puncture field diverges at the "
            f"particle, which sits at r = {self._orbit.r0}, theta = pi/2; this "
            f"point is {dist:.3e} away in coordinate distance. Worldtube grids "
            "straddle the particle rather than landing on it, so if this fires "
            "the grid did not come from Grid.worldtube."
        )

    @staticmethod
    def _check_mode(m: int) -> int:
        m = int(m)
        if m < 0:
            raise ValueError(
                f"m = {m}: the library's tables are indexed from 0, so a negative "
                "mode would read out of bounds. Use the conjugate symmetry instead."
            )
        if m > MAX_M:
            raise ValueError(
                f"m = {m} exceeds the {MAX_M} modes the library supports (its tables "
                "are dimensioned for m = 0..20). It would print a message and leave "
                "the output arrays untouched."
            )
        return m

    # -- the two routines -------------------------------------------------

    def h_s_m(self, m: int, r: float, theta: float) -> np.ndarray:
        """
        `effsource_hS_m`: the m-mode puncture field.

        Returns 2n floats, n real parts then n imaginary -- the layout
        `PunctureField` expects.
        """
        m = self._check_mode(m)
        if self._orbit is None:
            raise RuntimeError("no orbit configured; call set_orbit(orbit) first")
        out = np.zeros(2 * N_COMPONENTS, dtype=np.float64)
        x = Coordinate(t=0.0, r=float(r), theta=float(theta), phi=0.0)
        dp = ctypes.POINTER(ctypes.c_double)
        buf = out.ctypes.data_as(dp)
        self._lib.effsource_hS_m(
            m, ctypes.byref(x), buf, ctypes.cast(ctypes.addressof(buf.contents) + N_COMPONENTS * 8, dp)
        )
        self._check_finite(out, m, r, theta)
        return out

    def calc_m(self, m: int, r: float, theta: float) -> dict[str, np.ndarray]:
        """
        `effsource_calc_m`: all six outputs, as complex n-vectors.

        The source vector needs only `src`, but the puncture field and its
        four first derivatives come out of the same call, so they are handed
        back too.
        """
        m = self._check_mode(m)
        if self._orbit is None:
            raise RuntimeError("no orbit configured; call set_orbit(orbit) first")
        buffers = [np.zeros(N_COMPONENTS, dtype=np.float64) for _ in range(12)]
        dp = ctypes.POINTER(ctypes.c_double)
        args = [b.ctypes.data_as(dp) for b in buffers]
        x = Coordinate(t=0.0, r=float(r), theta=float(theta), phi=0.0)
        self._lib.effsource_calc_m(m, ctypes.byref(x), *args)
        out = {
            name: buffers[2 * i] + 1j * buffers[2 * i + 1]
            for i, name in enumerate(CALC_M_OUTPUTS)
        }
        self._check_finite(out["src"], m, r, theta)
        return out

    def src_m(self, m: int, r: float, theta: float) -> np.ndarray:
        """`effsource_calc_m`, keeping only `src`, in the 2n-real layout."""
        src = self.calc_m(m, r, theta)["src"]
        return np.concatenate([src.real, src.imag])

    # -- the object source_vector wants -----------------------------------

    def puncture_field(self, orbit: Orbit) -> PunctureField:
        """
        A `PunctureField` bound to one orbit, ready for `source_vector`.

        Each evaluation re-checks the configured orbit, so this stays correct
        even if something else touches the library in between.
        """
        self.set_orbit(orbit)

        def h_s(m, r, theta):
            self._require_orbit(orbit)
            return self.h_s_m(m, r, theta)

        def s_eff(m, r, theta):
            self._require_orbit(orbit)
            return self.src_m(m, r, theta)

        return PunctureField(orbit=orbit, h_s=h_s, s_eff=s_eff, n=N_COMPONENTS)


def load_effective_source_library(
    path: str | os.PathLike | None = None, **kwargs
) -> EffectiveSourceLibrary:
    """
    Load the compiled library, searching the usual places.

    See `find_effective_source_library` for the search order, and the comment
    block above for how to build it.
    """
    return EffectiveSourceLibrary(path, **kwargs)
