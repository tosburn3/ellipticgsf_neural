"""
The problem: one m-mode of the metric perturbation of a circular orbit around a Schwarzschild black
hole, discretized on an (r*, theta) grid (Sec. II), and the sparse linear system M x = b it becomes
(Sec. III).

    problem = build_problem(m=2, r0=6.0, grid_parameters=GridParameters(...))
    problem.M, problem.b                   # the system of Sec. III
    problem.grid, problem.orbit            # the discretization and the orbit (Sec. II)

THE UNKNOWNS
------------
x holds the ten tensor components psi^0 ... psi^9 of the field at every grid node, ordered with the
component fastest, then r*, then theta:

    x[unknown_index(l, i, j)] = psi^l at (r*_i, theta_j),    unknown_index = l + 10 (i + N_r* j)

(Here i and j are grid indices. In Sec. IV the same letters are matrix indices: row and column of M.)
`field_components` unpacks x into a (10, N_r*, N_theta) array.

WHAT IS IN HERE
---------------
    Orbit, Grid                     the orbit and the worldtube-aligned grid (Sec. II), and the tortoise
                                    coordinate r*(r) behind them
    GridParameters                  the requested spacings and extents of a grid (the values used in the
                                    paper are in scripts/settings.py)
    Problem, build_problem          M and b for one (m, r0)
    field_components,
    retarded_field_components       x as fields on the grid

Geometric units, with the black hole mass M = 1 (BLACK_HOLE_MASS). The coordinate code is written for a
Kerr black hole of spin a; this project uses a = 0 throughout.

WHY r* AND WHY x = r - r_+
--------------------------
The radial direction is discretized uniformly in the tortoise coordinate r* rather than in r, because
r* resolves the near-horizon region, where the field oscillates on a scale that shrinks with r - r_+,
without wasting points in the far field. Internally the shifted variable x = r - r_+ does the
arithmetic: near the horizon r* -> -infinity logarithmically in x, and working in x keeps that
cancellation-prone region well conditioned (at r* = -50 the radius differs from r_+ by about 1e-11, so
forming r - r_+ by subtraction would throw away most of the significant digits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.special import lambertw

__all__ = [
    "BLACK_HOLE_MASS",
    "N_COMPONENTS",
    "unknown_index",
    "Orbit",
    "Grid",
    "GridParameters",
    "Problem",
    "build_problem",
    "make_grid",
    "field_components",
    "retarded_field_components",
    "WorldtubeSpan",
    "horizon_radii",
    "kerr_delta",
    "drstar_dr",
    "rstar_of_x",
    "x_of_rstar",
    "snap_worldtube",
    "rstar_nodes",
    "theta_nodes",
]

#: The black hole mass. Fixed: the coefficient expressions are written for M = 1.
BLACK_HOLE_MASS = 1.0

#: The ten tensor components psi^0 ... psi^9 of the metric perturbation at every grid node.
N_COMPONENTS = 10


def unknown_index(l: int, i: int, j: int, n_rstar: int) -> int:
    """Position in x (a row/column of M) of component l at grid node (r*_i, theta_j)."""
    return l + N_COMPONENTS * (i + n_rstar * j)


# --------------------------------------------------------------------------
# The tortoise coordinate
# --------------------------------------------------------------------------


def horizon_radii(a: float) -> tuple[float, float]:
    """Outer and inner horizon radii r_+ , r_- for spin a."""
    s = np.sqrt(1.0 - a * a)
    return 1.0 + s, 1.0 - s


def kerr_delta(a: float, x):
    """Delta = r^2 - 2r + a^2, written as x (r - r_-) to keep it exact near r_+."""
    rp, rm = horizon_radii(a)
    r = np.asarray(x) + rp
    return np.asarray(x) * (r - rm)


def drstar_dr(a: float, x):
    """dr*/dr = (r^2 + a^2) / Delta."""
    rp, _ = horizon_radii(a)
    r = np.asarray(x) + rp
    return (r * r + a * a) / kerr_delta(a, x)


def rstar_of_x(a: float, x):
    """
    The tortoise coordinate, as a function of x = r - r_+:

        r* = r + 2/(r_+ - r_-) * ( r_+ log(x/2) - r_- log((r - r_-)/2) )

    For a = 0 this collapses to the familiar r* = r + 2 log(r/2 - 1).
    """
    rp, rm = horizon_radii(a)
    x = np.asarray(x, dtype=float)
    r = x + rp
    if rp == rm:  # extremal: the two log terms merge and this form breaks down
        raise ValueError("extremal Kerr (|a| = 1) is not supported")
    term = rp * np.log(x / 2.0)
    if rm != 0.0:
        term = term - rm * np.log((r - rm) / 2.0)
    return r + 2.0 / (rp - rm) * term


def _initial_guess(a: float, rstar: np.ndarray) -> np.ndarray:
    """
    Starting point for the r* -> x inversion, by regime.

    Each branch inverts the dominant behaviour of r*(x) analytically, which
    puts the guess close enough that Newton converges in a few steps
    everywhere, including tens of e-foldings inside the horizon scale.
    """
    rp, _ = horizon_radii(a)
    rstar = np.asarray(rstar, dtype=float)
    guess = np.empty_like(rstar)

    near = rstar < -2.0
    mid = (~near) & (rstar < 1000.0)
    far = rstar >= 1000.0

    # Near the horizon r* is dominated by r_+ log(x), so invert that.
    if np.any(near):
        guess[near] = (
            2.0
            * (1.0 - a * a) ** (1.0 / rp - 0.5)
            * np.exp((a * a - rp + (rp - 1.0) * rstar[near]) / rp)
        )
    # In between, the Schwarzschild form r + 2 log(r/2 - 1) inverts exactly
    # through the Lambert W function.
    if np.any(mid):
        guess[mid] = 2.0 * np.real(lambertw(np.exp((rstar[mid] - rp) / 2.0)))
    # Far away, r* ~ r.
    if np.any(far):
        guess[far] = rstar[far]

    return np.maximum(guess, 1e-14)


def x_of_rstar(a: float, rstar, tol: float = 1e-13, max_iter: int = 100) -> np.ndarray:
    """
    Invert r*(x) for x = r - r_+, on a whole array at once.

    Damped Newton. dr*/dr is known in closed form so convergence is
    quadratic, and the step is clipped so an iterate can never cross the
    horizon into x <= 0, where r*(x) is undefined.
    """
    rstar = np.atleast_1d(np.asarray(rstar, dtype=float))
    x = _initial_guess(a, rstar)

    for _ in range(max_iter):
        f = rstar_of_x(a, x) - rstar
        fp = drstar_dr(a, x)  # d(r*)/dx == d(r*)/dr, since dr/dx = 1
        step = f / fp
        step = np.where(step >= x, 0.5 * x, step)
        x_new = x - step
        converged = np.all(np.abs(x_new - x) < tol * np.maximum(1.0, np.abs(x_new)))
        x = x_new
        if converged:
            break

    resid = np.max(np.abs(rstar_of_x(a, x) - rstar))
    if not np.isfinite(resid) or resid > 1e-8 * max(1.0, np.max(np.abs(rstar))):
        raise RuntimeError(f"r* inversion failed to converge (max residual {resid:.3e})")
    return x


# --------------------------------------------------------------------------
# The orbit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Orbit:
    """
    A circular equatorial orbit of radius r0 around a Kerr hole of spin a.

    Frozen, because the coefficient matrices, the boundary conditions and the
    effective-source library are all configured from it; letting it change
    underneath them is the one bug this type exists to prevent.
    """

    a: float
    r0: float

    def __post_init__(self) -> None:
        if not abs(self.a) < 1.0:
            raise ValueError(f"spin must satisfy |a| < 1, got a = {self.a}")
        if self.r0 <= 0.0:
            raise ValueError(f"orbital radius must be positive, got r0 = {self.r0}")

    @property
    def omega(self) -> float:
        """Orbital angular frequency, Omega = v^3 / (1 + a v^3) with v = 1/sqrt(r0)."""
        v = 1.0 / math.sqrt(self.r0)
        return v**3 / (1.0 + self.a * v**3)

    def mode_frequency(self, m: int) -> float:
        """The frequency of the m-mode, omega = m Omega."""
        return m * self.omega

    @property
    def horizon(self) -> float:
        """Outer horizon radius r_+."""
        return horizon_radii(self.a)[0]

    @property
    def rstar0(self) -> float:
        """The particle's tortoise coordinate."""
        return float(rstar_of_x(self.a, np.array([self.r0 - self.horizon]))[0])


# --------------------------------------------------------------------------
# Worldtube-aligned spacing
#
# An effective-source calculation needs the worldtube faces to fall exactly on
# grid lines, so the requested spacings are nudged until they do. The point
# counts across the worldtube and across the polar source region are also
# forced to be EVEN, which puts the particle strictly between nodes in both
# directions -- necessary, because the puncture field diverges at the particle
# and any grid point landing on it would evaluate to infinity.
# --------------------------------------------------------------------------


class WorldtubeSpan(NamedTuple):
    """Tortoise coordinates of the domain edges and the worldtube faces."""

    inner: float   #: inner edge of the computational domain
    left: float    #: inner face of the worldtube
    center: float  #: the particle
    right: float   #: outer face of the worldtube
    outer: float   #: outer edge of the computational domain

    @property
    def diameter(self) -> float:
        return self.right - self.left


def snap_worldtube(
    orbit: Orbit, diameter: float, rstar_inner: float, rstar_outer: float
) -> WorldtubeSpan:
    """
    Place the worldtube around the particle and snap the domain edges outward
    to whole multiples of its width.

    `rstar_inner` and `rstar_outer` are requested positions for the edges of
    the domain; each is moved away from the particle until it sits a whole
    number of worldtube widths from the nearer face.
    """
    center = orbit.rstar0
    left = center - 0.5 * diameter
    right = center + 0.5 * diameter
    outer = right + diameter * math.ceil((rstar_outer - right) / diameter)
    inner = left - diameter * math.ceil((left - rstar_inner) / diameter)

    # The snapping arithmetic will cheerfully return a domain that does not
    # contain the worldtube if the requested edges are on the wrong side of
    # the particle. Left alone, that surfaces much later as an out-of-range
    # index, so it is caught here where the cause is visible.
    if not (inner < left < right < outer):
        raise ValueError(
            f"the snapped domain does not contain the worldtube: "
            f"inner={inner:.4f} < left={left:.4f} < right={right:.4f} "
            f"< outer={outer:.4f} is violated. The particle sits at "
            f"r* = {center:.4f}, so rstar_inner must be well below it and "
            f"rstar_outer well above it."
        )
    return WorldtubeSpan(inner, left, center, right, outer)


def rstar_nodes(span: WorldtubeSpan, delta_rstar: float) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Uniform r* nodes across the domain, plus the worldtube's index range.

    The spacing is adjusted from `delta_rstar` so that an even number of nodes
    spans the worldtube and both faces are nodes. Returns the node array and
    the inclusive (first, last) radial index of the worldtube.

    Note that `delta_rstar` is a request, not a setting: it is quantized, so two
    nearby values can produce the identical grid. Read the realized spacing
    off `Grid.delta_rstar` rather than assuming it.
    """
    diameter = span.diameter
    # Round the node count across the worldtube up or down to the nearest even
    # number, never below two. A coarse request therefore lands on a two-node
    # worldtube rather than being rejected here; the resulting grid may then be
    # too short for the boundary stencils, which `assemble` reports.
    count = max(2, 2 * round((1.0 + diameter / delta_rstar) / 2.0))
    spacing = diameter / (count - 1)
    total = round((span.outer - span.inner) / spacing) + 1
    nodes = span.inner + spacing * np.arange(total)
    bounds = (
        round((span.left - span.inner) / spacing),
        round((span.right - span.inner) / spacing),
    )
    return nodes, bounds


def theta_nodes(delta_theta: float, theta_size: float) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Uniform theta nodes from 0 to pi, plus the source region's index range.

    `theta_size` is the requested angular width of the source region; like the
    worldtube width it is snapped so that a whole odd number of such regions
    tiles [0, pi] with one centred on the equator, and then the spacing is
    adjusted so an even number of nodes spans it.

    The last node is `round(pi / spacing) * spacing`, which need not be
    exactly pi; the assembler always takes its delta_theta from the endpoints
    rather than from the request, so that costs nothing.
    """
    count_in_pi = round(np.pi / theta_size)
    if count_in_pi % 2 == 1:
        regions = float(count_in_pi)
    else:
        # An even tiling would straddle the equator with a region boundary
        # instead of centring one there; step to the nearest odd count.
        regions = 2.0 * (round(np.pi / (2.0 * theta_size)) + 0.5)
    size = np.pi / regions

    count = max(2, 2 * round((1.0 + size / delta_theta) / 2.0))
    spacing = size / (count - 1)
    total = round(np.pi / spacing) + 1
    nodes = spacing * np.arange(total)
    bounds = (
        round((np.pi / 2 - size / 2) / spacing),
        round((np.pi / 2 + size / 2) / spacing),
    )
    return nodes, bounds


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


def _inclusive_range(lo: float, hi: float, step: float) -> np.ndarray:
    """
    Nodes from `lo` towards `hi` in steps of `step`, including `hi` only when
    it is hit to within rounding.

    Built from a count rather than by accumulation, so the node positions do
    not drift over thousands of points.
    """
    count = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(count)


class Grid:
    """
    The (r*, theta) mesh a mode is solved on.

    Holds the radial nodes in all three parameterizations that get used -- r*,
    x = r - r_+ and r -- so nothing downstream has to re-invert the tortoise
    coordinate or re-derive a radius.

    `worldtube_rstar_indices` and `worldtube_theta_indices` are inclusive (first, last) index pairs
    bounding the worldtube, or None on a grid that was not built for an
    effective-source calculation. `Grid.worldtube` is the constructor that
    sets them.

    The stored spacings are computed from the endpoints, not from whatever was
    requested, because that is what the finite differences actually see.
    """

    __slots__ = (
        "a", "rstar", "x", "r", "delta_rstar",
        "theta", "delta_theta", "worldtube_rstar_indices", "worldtube_theta_indices",
    )

    def __init__(
        self,
        a: float,
        rstar: np.ndarray,
        theta: np.ndarray,
        *,
        x: np.ndarray | None = None,
        worldtube_rstar_indices: tuple[int, int] | None = None,
        worldtube_theta_indices: tuple[int, int] | None = None,
    ):
        self.a = float(a)
        self.rstar = np.asarray(rstar, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        if self.rstar.ndim != 1 or self.theta.ndim != 1:
            raise ValueError("rstar and theta must be one-dimensional")
        if len(self.rstar) < 2 or len(self.theta) < 2:
            raise ValueError("a grid needs at least two nodes in each direction")

        # x may be supplied to reuse values computed elsewhere -- reloading a
        # saved run, say -- which skips the inversion.
        self.x = x_of_rstar(self.a, self.rstar) if x is None else np.asarray(x, dtype=float)
        if len(self.x) != len(self.rstar):
            raise ValueError("x and rstar have different lengths")
        self.r = self.x + horizon_radii(self.a)[0]

        self.delta_rstar = (self.rstar[-1] - self.rstar[0]) / (len(self.rstar) - 1)
        self.delta_theta = (self.theta[-1] - self.theta[0]) / (len(self.theta) - 1)
        self.worldtube_rstar_indices = worldtube_rstar_indices
        self.worldtube_theta_indices = worldtube_theta_indices

    # -- constructors -----------------------------------------------------

    @classmethod
    def uniform(
        cls, a: float, delta_rstar: float, delta_theta: float, rstar_min: float, rstar_max: float
    ) -> "Grid":
        """
        A plain grid with no worldtube: r* from `rstar_min` to `rstar_max` and
        theta from 0 to pi, at the requested spacings.

        Enough for the operator on its own -- benchmarks, sparsity studies,
        preconditioner experiments -- but not for a source vector.
        """
        return cls(
            a,
            _inclusive_range(rstar_min, rstar_max, delta_rstar),
            _inclusive_range(0.0, np.pi, delta_theta),
        )

    @classmethod
    def worldtube(
        cls,
        orbit: Orbit,
        *,
        delta_rstar: float,
        delta_theta: float,
        rstar_inner: float,
        rstar_outer: float,
        diameter: float = 1.0,
        theta_size: float = 0.4,
    ) -> "Grid":
        """
        A grid aligned to a worldtube around the particle.

        `diameter` is the worldtube's width in r* and `theta_size` the angular
        width of the source region; both are snapped as described above, so
        the grid that comes back may not have exactly the spacings requested.
        """
        span = snap_worldtube(orbit, diameter, rstar_inner, rstar_outer)
        rstar, wt_i = rstar_nodes(span, delta_rstar)
        theta, wt_j = theta_nodes(delta_theta, theta_size)
        return cls(orbit.a, rstar, theta, worldtube_rstar_indices=wt_i, worldtube_theta_indices=wt_j)

    # -- shape ------------------------------------------------------------

    @property
    def n_rstar(self) -> int:
        return len(self.rstar)

    @property
    def n_theta(self) -> int:
        return len(self.theta)

    def size(self, n_components: int) -> int:
        """Number of unknowns for `n_components` field components per node."""
        return n_components * self.n_rstar * self.n_theta

    def require_worldtube(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        The worldtube bounds, refusing clearly if this grid has none or if it
        leaves no room for the jump halo one cell outside each face.
        """
        if self.worldtube_rstar_indices is None or self.worldtube_theta_indices is None:
            raise ValueError(
                "this grid has no worldtube; build it with Grid.worldtube(...) "
                "if you need a source vector"
            )
        imin, imax = self.worldtube_rstar_indices
        jmin, jmax = self.worldtube_theta_indices
        if not (1 <= imin <= imax <= self.n_rstar - 2):
            raise ValueError(
                f"radial worldtube bounds {self.worldtube_rstar_indices} leave no room for the "
                f"jump halo in {self.n_rstar} radial nodes"
            )
        if not (1 <= jmin <= jmax <= self.n_theta - 2):
            raise ValueError(
                f"polar worldtube bounds {self.worldtube_theta_indices} leave no room for the "
                f"jump halo in {self.n_theta} polar nodes"
            )
        return self.worldtube_rstar_indices, self.worldtube_theta_indices

    def __repr__(self) -> str:
        wt = "" if self.worldtube_rstar_indices is None else (
            f", worldtube i={self.worldtube_rstar_indices} j={self.worldtube_theta_indices}"
        )
        return (
            f"Grid(a={self.a}, {self.n_rstar} x {self.n_theta}, "
            f"r* in [{self.rstar[0]:.3f}, {self.rstar[-1]:.3f}]{wt})"
        )


# --------------------------------------------------------------------------
# The linear system M x = b of one mode (Sec. III)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridParameters:
    """
    The requested grid for one mode. Each value is a request: `Grid.worldtube` snaps the spacings so
    that whole numbers of cells span the worldtube and the domain edges sit whole numbers of worldtube
    widths from it, so read the realized values off the grid. The values used in the paper are in
    scripts/settings.py.
    """

    delta_rstar: float              #: radial spacing Delta r*
    rstar_min: float                #: inner edge of the domain (snapped outward)
    rstar_max: float                #: outer edge of the domain (snapped outward)
    delta_theta: float              #: polar spacing Delta theta
    worldtube_rstar_width: float    #: worldtube width in r*
    worldtube_theta_width: float    #: worldtube width in theta


@dataclass
class Problem:
    """M x = b for mode m of the circular orbit `orbit` on `grid` (Sec. III)."""

    m: int
    orbit: Orbit
    grid: Grid
    M: sp.csr_matrix          #: the sparse matrix (sorted CSR, no explicit zeros)
    b: np.ndarray             #: the source vector (complex128)
    puncture: object          #: the puncture field (source.PunctureField), to recover the retarded field

    @property
    def n(self) -> int:
        """Number of unknowns."""
        return self.M.shape[0]

    def describe(self) -> str:
        g = self.grid
        return (f"m = {self.m}, r0 = {self.orbit.r0:g}: {g.n_rstar} x {g.n_theta} nodes, {self.n:,} unknowns, "
                f"nnz(M) = {self.M.nnz:,}, Delta r* = {g.delta_rstar:.4f}, "
                f"r* in [{g.rstar[0]:.2f}, {g.rstar[-1]:.2f}]")


def make_grid(orbit: Orbit, grid_parameters: GridParameters) -> Grid:
    """The worldtube-aligned grid requested by `grid_parameters` (snapped; see GridParameters)."""
    p = grid_parameters
    return Grid.worldtube(orbit, delta_rstar=p.delta_rstar, delta_theta=p.delta_theta,
                          rstar_inner=p.rstar_min, rstar_outer=p.rstar_max,
                          diameter=p.worldtube_rstar_width, theta_size=p.worldtube_theta_width)


def build_problem(m: int, r0: float, grid_parameters: GridParameters, effective_source_library=None) -> Problem:
    """The sparse matrix M and the effective-source vector b for mode m of the circular orbit at r0
    (a = 0), on the grid requested by `grid_parameters`."""
    # Imported here: assemble.py and source.py import Grid and Orbit from this module.
    from .assemble import assemble
    from .source import load_effective_source_library, source_vector

    orbit = Orbit(a=0.0, r0=float(r0))
    grid = make_grid(orbit, grid_parameters)
    M = assemble(m, orbit, grid).M.tocsr()
    M.sum_duplicates()
    M.eliminate_zeros()
    M.sort_indices()
    puncture = (effective_source_library or load_effective_source_library()).puncture_field(orbit)
    b = np.asarray(source_vector(m, puncture, grid), np.complex128)
    return Problem(m, orbit, grid, M, b, puncture)


def field_components(problem: Problem, x: np.ndarray) -> np.ndarray:
    """x as the ten components on the grid: an (10, N_r*, N_theta) array. Inside the worldtube this is the
    residual field that was solved for (see retarded_field_components)."""
    g = problem.grid
    return np.asarray(x).reshape(g.n_theta, g.n_rstar, N_COMPONENTS).transpose(2, 1, 0)


def retarded_field_components(problem: Problem, x: np.ndarray) -> np.ndarray:
    """The retarded field: the solved field with the puncture field added back inside the worldtube."""
    g = problem.grid
    out = field_components(problem, x).copy()
    (imin, imax), (jmin, jmax) = g.worldtube_rstar_indices, g.worldtube_theta_indices
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            out[:, i, j] += problem.puncture.psi(problem.m, float(g.r[i]), float(g.theta[j]))
    return out
