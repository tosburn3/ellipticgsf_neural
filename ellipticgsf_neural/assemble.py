"""
The sparse matrix M of one m-mode: the coefficients of the elliptic system, its finite-difference
discretization, and the boundary conditions (Secs. II C and III; boundary strips in App. A).

THE SYSTEM
----------
The ten tensor components psi = (psi^0, ..., psi^9) of the m-mode live at every node of the (r*, theta)
grid and satisfy

    [ d2/dr*^2 + (Delta / (r^2 + a^2)^2) d2/dtheta^2 + Q_r* d/dr* + Q_theta d/dtheta + Q_0 ] psi  =  S

with Q_r*, Q_theta and Q_0 the 10 x 10 coefficient matrices (functions of r and theta for a given m,
spin a and frequency omega = m Omega) and S the effective source (source.py). All derivatives are
second-order centred differences, so each interior row has a five-point radial-and-polar stencil on its
own component plus on-node and first-derivative couplings to the other nine.

M is complex and non-Hermitian: omega enters the coefficients, and the radiation conditions at the two
radial edges are one-sided and complex.

ORDERING
--------
    row/column = unknown_index(l, i, j) = l + 10 (i + N_r* j)        (problem.py)

The component index l is fastest, then the radial index i, then the polar index j: consecutive unknowns
are the ten components at one node, which makes the coupling blocks contiguous and M as banded as it can
be.

BOUNDARY LAYOUT
---------------
Every row is constrained exactly once, and the two kinds of condition do not overlap:

  * The interior coefficients are identically zero on all four edges, so an interior stencil never
    writes into a boundary row.
  * Rows at j = 0 and j = N_theta - 1 (the poles) carry the polar condition (Dirichlet or Neumann per
    component), for every i including the four corners.
  * Rows at i = 0 and i = N_r* - 1 carry the radial conditions, for j = 1 .. N_theta - 2. The
    near-horizon condition involves the fifth r* derivative and couples the seven radii nearest the
    horizon; the outer condition involves the third and couples the five outermost radii. Those radii
    are the boundary strips x_A and x_C of App. A (HORIZON_STRIP_RADII, OUTER_STRIP_RADII).

WHAT IS IN HERE
---------------
    first_rstar_derivative_coefficients    Q_r*     \\
    first_theta_derivative_coefficients    Q_theta   |  machine-emitted from the symbolic derivation of
    non_derivative_coefficients            Q_0       |  the field equations: EDIT NOTHING IN THESE BY
    near_horizon_boundary_condition                  |  HAND (a change belongs upstream in the
    outer_boundary_condition(_m0)                   /   derivation, which then re-emits them)
    assemble                M, fast (vectorized COO batches; a 1.7-million-unknown mode in ~4 s)
    assemble_reference      M, slow and obviously correct, to check `assemble` against

Each coefficient function takes a scalar theta and a vector of r values and returns an array of shape
(10, 10, len(r)). The boundary-condition functions return lists of stencil entries

    (row_component, row_offset, col_component, col_offset, col_dtheta, value)

with 0-based component indices (or None for a rule that applies to every component, in which case
`value` is a list of 10 values), radial offsets measured from the edge in question (from index 0 inwards
at the horizon, from N_r* - 1 inwards at the outer edge), the column's polar offset relative to the row,
and `value` broadcasting against the vector of theta values passed in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as _dc_field

import numpy as np
import scipy.sparse as sp

from .problem import N_COMPONENTS, Grid, Orbit, kerr_delta, unknown_index

__all__ = [
    "N_COMPONENTS",
    "HORIZON_STRIP_RADII",
    "OUTER_STRIP_RADII",
    "first_rstar_derivative_coefficients",
    "first_theta_derivative_coefficients",
    "non_derivative_coefficients",
    "near_horizon_boundary_condition",
    "outer_boundary_condition",
    "outer_boundary_condition_m0",
    "AssembledSystem",
    "assemble",
    "assemble_reference",
    "polar_condition_classes",
]

#: Radii coupled by the near-horizon boundary condition (offsets 0..6): the strip x_A of App. A.
HORIZON_STRIP_RADII = 7
#: Radii coupled by the outer boundary condition (offsets 0..-4): the strip x_C of App. A.
OUTER_STRIP_RADII = 5




# ======================================================================
# The coefficient matrices and boundary-condition stencils (machine-emitted)
# ======================================================================

# --------------------------------------------------------------------------
# Trigonometric shorthands used by the emitted expressions
# --------------------------------------------------------------------------


def _cot(x):
    return np.cos(x) / np.sin(x)


def _csc(x):
    return 1.0 / np.sin(x)


def _stack(rows, shape):
    """Broadcast every entry to the r-grid shape and stack to (n, n, len(r))."""
    n = len(rows)
    out = np.empty((n, len(rows[0]), *shape), dtype=complex)
    for i, row in enumerate(rows):
        for j, entry in enumerate(row):
            out[i, j] = np.broadcast_to(np.asarray(entry, dtype=complex), shape)
    return out


# --------------------------------------------------------------------------
# Second-order finite-difference weights, by offset from the row's own node.
# `_fwd` are the one-sided forms the boundary conditions need.
# --------------------------------------------------------------------------


def _d1_up(h):
    """First derivative, central, weight of the node one step up."""
    return 1.0 / (2 * h)


def _d1_down(h):
    """First derivative, central, weight of the node one step down."""
    return -1.0 / (2 * h)


def _d1_fwd0(h):
    """First derivative, forward three-point form: weights (-3, 4, -1) / 2h."""
    return -3.0 / (2 * h)


def _d1_fwd1(h):
    return 2.0 / h


def _d1_fwd2(h):
    return -1.0 / (2 * h)


# --------------------------------------------------------------------------
# The coefficient matrices
# --------------------------------------------------------------------------


def first_rstar_derivative_coefficients(m, a, omega, r, theta):
    """Q_r*, the coefficient matrix of d psi / dr*. Shape (10, 10, len(r))."""
    r = np.asarray(r, dtype=float)
    theta = float(theta)
    _rows = [[(-2*a*(a**3-1j*a**2*m*r+a*(-2+r)*r-1j*m*r**3))/(r*(a**2+r**2)**2)-(32*r**2)/(a**2+2*r**2+a**2*np.cos(2*theta))**2+8/(a**2+2*r**2+a**2*np.cos(2*theta)),0*r,0*r,(8*a*r*(a**2-2*r**2+a**2*np.cos(2*theta))*np.sin(theta))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,0*r,0*r,0*r,0*r,0*r],[0*r,(2*(a**4+1j*a**3*m*r+1j*a*m*r**3+(-3+r)*r**3+a**2*r*(-1+2*r)))/(r*(a**2+r**2)**2)-(16*r**2)/(a**2+2*r**2+a**2*np.cos(2*theta))**2-(4*(a**2*(-1+r)+(-3+r)*r**2))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),(-2*a**2*(a**2+(-2+r)*r)*np.sin(2*theta))/(r*(a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r,0*r,(4*a*r*(a**2-2*r**2+a**2*np.cos(2*theta))*np.sin(theta))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,0*r,0*r],[0*r,(2*a**2*r*np.sin(2*theta))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),(2*(1j*a**3*m+a**2*r+(-2+1j*a*m)*r**2+r**3))/(a**2+r**2)**2-(16*r**2)/(a**2+2*r**2+a**2*np.cos(2*theta))**2-(4*(a**2*(-1+r)+(-3+r)*r**2))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r,0*r,0*r,0*r,(4*a*r*(a**2-2*r**2+a**2*np.cos(2*theta))*np.sin(theta))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r],[(2*a*(r**2*(a**2+3*r**2)+a**2*(-a+r)*(a+r)*np.cos(theta)**2)*np.sin(theta))/(r*(a**2+r**2)*(r**2+a**2*np.cos(theta)**2)**2),0*r,0*r,(2*(a**2+1j*a**3*m-r**2+1j*a*m*r**2))/(a**2+r**2)**2,0*r,0*r,0*r,0*r,0*r,(4*a*r*(a**2-2*r**2+a**2*np.cos(2*theta))*np.sin(theta))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**2)],[0*r,0*r,0*r,0*r,(2*(1j*a**3*m+(3*a**4)/r+1j*a*m*r**2+2*(-3+r)*r**2+a**2*(-4+5*r)-(4*r*((-1j)*a+r)*(1j*a+r)*(a**2+(-2+r)*r))/(a**2+2*r**2+a**2*np.cos(2*theta))))/(a**2+r**2)**2,(-4*a**2*(a**2+(-2+r)*r)*np.sin(2*theta))/(r*(a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r,0*r,0*r],[0*r,0*r,0*r,0*r,(2*a**2*r*np.sin(2*theta))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),(2*(1j*a**3*m+(2*a**4)/r+1j*a*m*r**2+r**2*(-5+2*r)+a**2*(-3+4*r)-(4*r*((-1j)*a+r)*(1j*a+r)*(a**2+(-2+r)*r))/(a**2+2*r**2+a**2*np.cos(2*theta))))/(a**2+r**2)**2,0*r,(-2*a**2*(a**2+(-2+r)*r)*np.sin(2*theta))/(r*(a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r],[0*r,(-4*a*(a**4-3*a**2*r**2-6*r**4+a**2*(a-r)*(a+r)*np.cos(2*theta))*np.sin(theta))/(r*(a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,0*r,0*r,0*r,(4*a**4+(2*1j)*a**3*m*r+(2*1j)*a*m*r**3+2*(-4+r)*r**3+2*a**2*r*(-2+3*r))/(r*(a**2+r**2)**2)+(16*r**2)/(a**2+2*r**2+a**2*np.cos(2*theta))**2-(4*((-1+r)*r**2+a**2*(1+r)))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,-((a**2*(a**2+(-2+r)*r)*np.sin(2*theta))/(r*(a**2+r**2)*(r**2+a**2*np.cos(theta)**2))),0*r],[0*r,0*r,0*r,0*r,0*r,(4*a**2*r*np.sin(2*theta))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,(2*a*(a**5+1j*a**4*m*r+a**3*(-2+r)*r+(3*1j)*a**2*m*r**3+(2*1j)*m*r**5+a*(a**4+1j*a**3*m*r+1j*a*m*r**3+2*(-2+r)*r**3+a**2*r*(-2+3*r))*np.cos(2*theta)))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r],[0*r,0*r,-((a*(a**4-3*a**2*r**2-6*r**4+a**2*(a-r)*(a+r)*np.cos(2*theta))*np.sin(theta))/(r*(a**2+r**2)*(r**2+a**2*np.cos(theta)**2)**2)),0*r,0*r,0*r,(a**2*r*np.sin(2*theta))/((a**2+r**2)*(r**2+a**2*np.cos(theta)**2)),0*r,(2*(a**4+1j*a**3*m*r+1j*a*m*r**3+(-3+r)*r**3+a**2*r*(-1+2*r)))/(r*(a**2+r**2)**2)+(16*r**2)/(a**2+2*r**2+a**2*np.cos(2*theta))**2-(4*((-1+r)*r**2+a**2*(1+r)))/((a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r],[0*r,0*r,0*r,(4*a*(r**2*(a**2+3*r**2)+a**2*(-a+r)*(a+r)*np.cos(theta)**2)*np.sin(theta))/(r*(a**2+r**2)*(r**2+a**2*np.cos(theta)**2)**2),0*r,0*r,0*r,0*r,0*r,(2*(a**4+1j*a**3*m*r+a**2*r**2+(-2+1j*a*m)*r**3))/(r*(a**2+r**2)**2)+(32*r**2)/(a**2+2*r**2+a**2*np.cos(2*theta))**2-8/(a**2+2*r**2+a**2*np.cos(2*theta))]]
    return _stack(_rows, r.shape)


def first_theta_derivative_coefficients(m, a, omega, r, theta):
    """Q_theta, the coefficient matrix of d psi / dtheta. Shape (10, 10, len(r))."""
    r = np.asarray(r, dtype=float)
    theta = float(theta)
    _rows = [[((a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)*(8*(a**2+r**2)**2*_cot(theta)-2*a**2*(3*a**2+4*(-4+r)*r+a**2*np.cos(2*theta))*np.sin(2*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),0*r,0*r,(32*a*r**2*(a**2+(-2+r)*r)*np.cos(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,0*r,0*r,0*r,0*r,0*r],[0*r,((a**2+(-2+r)*r)*(8*(a**2+r**2)**2*_cot(theta)-2*a**2*(a**2-8*r)*np.sin(2*theta)+a**4*np.sin(4*theta)))/(2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(-4*(a**2+(-2+r)*r)**2)/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r,0*r,(16*a*r**2*(a**2+(-2+r)*r)*np.cos(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,0*r,0*r],[0*r,(2*r**2*(a**2+(-2+r)*r))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)),((a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)*(8*(a**2+r**2)**2*_cot(theta)-2*a**2*(a**2-8*r)*np.sin(2*theta)+a**4*np.sin(4*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),0*r,0*r,0*r,0*r,0*r,(16*a*r**2*(a**2+(-2+r)*r)*np.cos(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r],[(-4*a**3*(a**2+(-2+r)*r)*np.cos(theta)*np.sin(theta)**2)/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**2),0*r,0*r,((a**2+(-2+r)*r)*_cot(theta))/(a**2+r**2)**2,0*r,0*r,0*r,0*r,0*r,(16*a*r**2*(a**2+(-2+r)*r)*np.cos(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2)],[0*r,0*r,0*r,0*r,((a**2+(-2+r)*r)*(5*a**2+2*r**2-3*a**2*np.cos(2*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(-8*(a**2+(-2+r)*r)**2)/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r,0*r,0*r],[0*r,0*r,0*r,0*r,(4*r**2*(a**2+(-2+r)*r))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),((a**2+(-2+r)*r)*(5*a**2+2*r**2-3*a**2*np.cos(2*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,(-4*(a**2+(-2+r)*r)**2)/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,0*r],[0*r,(-4*a**3*(a**2+(-2+r)*r)*np.cos(theta)*np.sin(theta)**2)/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**2),0*r,0*r,0*r,0*r,((a**2+(-2+r)*r)*(5*a**4+16*a**2*(-1+r)*r+8*r**4+4*a**2*(a**2+4*r)*np.cos(2*theta)-a**4*np.cos(4*theta))*_cot(theta))/(2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,(-2*(a**2+(-2+r)*r)**2)/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)),0*r],[0*r,0*r,0*r,0*r,0*r,(4*r**2*(a**2+(-2+r)*r))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)),0*r,-(((a**2+(-2+r)*r)*(-5*a**2-2*r**2+3*a**2*np.cos(2*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta)))),0*r,0*r],[0*r,0*r,(-4*a**3*(a**2+(-2+r)*r)*np.cos(theta)*np.sin(theta)**2)/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**2),0*r,0*r,0*r,(2*r**2*(a**2+(-2+r)*r))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)),0*r,((a**2+(-2+r)*r)*(5*a**4+16*a**2*(-1+r)*r+8*r**4+4*a**2*(a**2+4*r)*np.cos(2*theta)-a**4*np.cos(4*theta))*_cot(theta))/(2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r],[0*r,0*r,0*r,(-8*a**3*(a**2+(-2+r)*r)*np.cos(theta)*np.sin(theta)**2)/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**2),0*r,0*r,0*r,0*r,0*r,((a**2+(-2+r)*r)*(3*a**4+8*a**2*(-4+r)*r+8*r**4+4*a**2*(a**2+2*r*(4+r))*np.cos(2*theta)+a**4*np.cos(4*theta))*_cot(theta))/(2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2)]]
    return _stack(_rows, r.shape)


def non_derivative_coefficients(m, a, omega, r, theta):
    """Q_0, the coefficient matrix of psi itself. Shape (10, 10, len(r))."""
    r = np.asarray(r, dtype=float)
    theta = float(theta)
    _rows = [[((4*(a**2-r-1j*a*m*r)*(a**2+(-2+r)*r))/r**2-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+a**2*(a**2+(-2+r)*r)*omega**2*np.cos(2*theta)+(16*(96*r**5*(a**2+r**2)+8*r**3*(a**2*(-5+r)+(-11+r)*r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))+(a**4+4*a**4*r+2*a**2*(-5-(2*1j)*a*m)*r**2+4*(5+2*a**2)*r**3+(-11-(4*1j)*a*m)*r**4+4*r**5)*(a**2+2*r**2+a**2*np.cos(2*theta))**2-((a**2-1j*a*m*r+(-2+r)*r)*(a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**3)/r))/(a**2+2*r**2+a**2*np.cos(2*theta))**4-2*m**2*(a**2+(-2+r)*r)*_csc(theta)**2)/(2*(a**2+r**2)**2),((8*1j)*r**2*(-(a*m)+(a**2+r**2)*omega)*(a**2-2*r**2+a**2*np.cos(2*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),((-16*1j)*a*r**2*(a**2+(-2+r)*r)*(2*m-a*omega+a*omega*np.cos(2*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(16*a*(r**2+a**2*np.cos(theta)**2)*(4*r**2*(a**2+(-2+r)*r)*np.cos(theta)*(a**2+2*r**2+a**2*np.cos(2*theta))**2*_cot(theta)+(a**2+(-2+r)*r)*(a**2-2*r**2+a**2*np.cos(2*theta))*(a**2+2*r**2+a**2*np.cos(2*theta))**2*np.sin(theta)+(192*r**6-16*r**4*(5+2*r)*(a**2+2*r**2+a**2*np.cos(2*theta))+2*r*(a**2+2*a**2*r+(3-(2*1j)*a*m)*r**2+2*r**3)*(a**2+2*r**2+a**2*np.cos(2*theta))**2-(a**2-1j*a*m*r+(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))**3)*np.sin(theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5),(-8*r**4*(r**2+a**2*np.cos(theta)**2)*(a**4*(3-27*r)-4*a**2*(-10+r)*r**2+8*r**4*(-3+2*r)+4*a**2*(a**2*(1-6*r)+(10-7*r)*r**2)*np.cos(2*theta)+a**4*(1+3*r)*np.cos(4*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5),(-48*a**2*r**3*(a**2+(-2+r)*r)*(a**2-6*r**2+a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**4),0*r,(16*r**3*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)*(-9*a**4+4*(-2+r)*r**3+2*a**2*r*(10+r)-2*a**2*(3*a**2+5*(-2+r)*r)*np.cos(2*theta)+3*a**4*np.cos(4*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5),0*r,(8*r**3*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)*((2*(a**3-2*a*r**2+a**3*np.cos(2*theta))**2)/(r*(a**2+(-2+r)*r))+32*a**2*r*_cot(theta)**2-((a**2+2*(-2+r)*r+a**2*np.cos(2*theta))**2*(3*a**2-2*r**2+3*a**2*np.cos(2*theta))*_csc(theta)**2)/(a**2+(-2+r)*r))*np.sin(theta)**2)/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5)],[((2*1j)*(-(a*m*(a**4-3*a**2*r**2+2*(4-3*r)*r**3))+(a**6+a**2*(4-3*r)*r**3-2*r**6)*omega+a**2*(a*m*(-a**2+r**2)+(a**4+2*a**2*r**2+(-4+r)*r**3)*omega)*np.cos(2*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(((4*1j)*a*m*(a**2-r))/r-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+a**2*(a**2+(-2+r)*r)*omega**2*np.cos(2*theta)+(8*(-48*r**3*(a**2+(-2+r)*r)*(a**2+r**2)+4*r*(4*a**4-1j*a**3*m*r-1j*a*m*r**3+8*(-2+r)*r**3+4*a**2*r*(-2+3*r))*(a**2+2*r**2+a**2*np.cos(2*theta))-((a**4*(-1+r)+1j*a**3*m*(-1+r)*r+1j*a*m*(-3+r)*r**3+r**3*(-6+r+r**2)+2*a**2*(r+r**3))*(a**2+2*r**2+a**2*np.cos(2*theta))**2)/r))/(a**2+2*r**2+a**2*np.cos(2*theta))**3-2*m**2*(a**2+(-2+r)*r)*_csc(theta)**2)/(2*(a**2+r**2)**2),-1/2*((a**2+(-2+r)*r)*(2*(1j*a**7*m+(4*1j)*a**5*m*r**2+(8*1j)*a**3*m*r**4+32*a**2*(-3+r)*r**4+16*(-2+r)*r**6+a**6*(8+6*r)+2*a**4*r**2*(-34+11*r))+a**2*(1j*a**5*m+16*a**4*r-(16*1j)*a*m*r**4+48*a**2*r**2*(2+r)+32*r**4*(2+r))*np.cos(2*theta)+2*a**4*((-1j)*a**3*m+2*a**2*(-4+r)-(4*1j)*a*m*r**2+2*(-6+r)*r**2)*np.cos(4*theta)-1j*a**7*m*np.cos(6*theta))*_cot(theta))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),((-4*1j)*m*r*(a**2+(-2+r)*r)**2*_csc(theta)+1j*a*((4*a*m*r**2*(-3+2*r)+a**3*m*(-1+5*r)+a**4*omega-5*a**2*r**2*omega+4*(4-3*r)*r**3*omega)*np.sin(theta)+a**2*(a*m*(-1+r)+(a-r)*(a+r)*omega)*np.sin(3*theta)))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),((4*1j)*r**2*(-(a*m)+(a**2+r**2)*omega)*(a**2-2*r**2+a**2*np.cos(2*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),((-8*1j)*a*r**2*(a**2+(-2+r)*r)*(2*m-a*omega+a*omega*np.cos(2*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(a*(32*r**2*(a**2+(-2+r)*r)*(a**2+r**2)*_csc(theta)+2*(2*a**6+1j*a**5*m*r+2*a**4*(-2+r)*r-48*a**2*r**4-(8*1j)*a*m*r**5-48*(-2+r)*r**5)*np.sin(theta)+a**2*(6*a**4+(3*1j)*a**3*m*r+32*(-2+r)*r**3+2*a**2*r*(-6+19*r))*np.sin(3*theta)+a**4*(2*a**2+1j*a*m*r+2*(-2+r)*r)*np.sin(5*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),0*r,(-32*a**3*(a**2+(-2+r)*r)*np.cos(theta)*(a**2-4*r**2+a**2*np.cos(2*theta))*np.sin(theta)**2)/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),0*r],[((4*1j)*a**2*(-(a*m)+(a**2+r**2)*omega)*np.sin(2*theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(a**2*((5*a**6+(5*1j)*a**5*m*r+(16*1j)*a**3*m*r**3+(16*1j)*a*m*r**5+16*(-2+r)*r**4*(10+r)+a**4*r*(-26+21*r)+16*a**2*r**2*(2+r*(7+2*r)))*np.sin(2*theta)+4*a**2*(a**4+1j*a**3*m*r+2*(-2+r)*(-1+r)*r**2+(2*1j)*a*m*r**3+a**2*r*(-4+3*r))*np.sin(4*theta)+a**4*(a**2+1j*a*m*r+(-2+r)*r)*np.sin(6*theta)))/(2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),(4*(a**2+(-2+r)*r)-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+a**2*(a**2+(-2+r)*r)*omega**2*np.cos(2*theta)+(8*(48*r**3*(a**2+(-2+r)*r)*(a**2+r**2)-4*r*(4*a**4+1j*a**3*m*r+1j*a*m*r**3+7*(-2+r)*r**3+a**2*r*(-8+11*r))*(a**2+2*r**2+a**2*np.cos(2*theta))+((-1j)*a**3*m*(-1+r)-a**2*(-4+r)*r-1j*a*m*(-3+r)*r**2-(-4+r)*(-2+r)*r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**2))/(a**2+2*r**2+a**2*np.cos(2*theta))**3-2*(1+m**2)*(a**2+(-2+r)*r)*_csc(theta)**2)/(2*(a**2+r**2)**2),((-8*1j)*(r**2+a**2*np.cos(theta)**2)*_csc(theta)*(m*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(2*theta))*_cot(theta)+a**3*(a*m-2*r*omega)*np.cos(theta)*np.sin(theta)**3))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,((4*1j)*r**2*(-(a*m)+(a**2+r**2)*omega)*(a**2-2*r**2+a**2*np.cos(2*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(-16*a*r**2*(a**2+(-2+r)*r)*np.cos(theta)*(a**2-10*r**2+a**2*np.cos(2*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),((-8*1j)*a*r**2*(a**2+(-2+r)*r)*(2*m-a*omega+a*omega*np.cos(2*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(a*(64*r**2*(a**2+(-2+r)*r)*(a**2+r**2)*_csc(theta)+2*(a**6+1j*a**5*m*r+16*a**2*(7-4*r)*r**3-(8*1j)*a*m*r**5-8*(-2+r)*r**5-a**4*r*(2+55*r))*np.sin(theta)+3*a**2*(a**4+1j*a**3*m*r-16*(-2+r)*r**3-a**2*r*(2+15*r))*np.sin(3*theta)+a**4*(a**2+1j*a*m*r+(-2+r)*r)*np.sin(5*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),0*r],[(4*a*(-192*r**6*(a**2+r**2)-16*r**4*(r**2*(-11+4*r)+a**2*(-5+4*r))*(a**2+2*r**2+a**2*np.cos(2*theta))-2*r*(a**4+2*a**4*r+2*a**2*(-7-1j*a*m)*r**2+4*(5+a**2)*r**3+(-15-(2*1j)*a*m)*r**4+2*r**5)*(a**2+2*r**2+a**2*np.cos(2*theta))**2+(a-r)*(a+r)*(a**2-1j*a*m*r+(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))**3)*np.sin(theta))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**4),((1j/8)*r*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(8*(-2+r)*r**4+4*a**2*r**2*(-3+2*r)+a**4*(1+3*r))+a*(-a**4+5*a**2*r**2+12*r**4)*omega+4*a*r*(a*m*(a**2+r*(-1+2*r))-r*(a**2+3*r**2)*omega)*np.cos(2*theta)+a**3*(a*m*(-1+r)+(a-r)*(a+r)*omega)*np.cos(4*theta))*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),((1j/8)*(a**2+(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(3*a**4+8*r**4+8*a**2*r*(1+r))-6*a**3*r*omega+4*a**2*(a**2*m+2*m*(-1+r)*r+2*a*r*omega)*np.cos(2*theta)+a**3*(a*m-2*r*omega)*np.cos(4*theta))*_cot(theta)*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),((-4*1j)*a*m*(-1+r)-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+a**2*(a**2+(-2+r)*r)*omega**2*np.cos(2*theta)+(16*(-192*r**5*(a**2+r**2)-16*r**3*(a**2*(-5+r)+(-11+r)*r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))-2*(a**4-4*(-5+a**2)*r**3+3*r**4-4*r**5)*(a**2+2*r**2+a**2*np.cos(2*theta))**2-(r**2*(-5+3*r)+a**2*(-1+3*r))*(a**2+2*r**2+a**2*np.cos(2*theta))**3))/(a**2+2*r**2+a**2*np.cos(2*theta))**4-2*(1+m**2)*(a**2+(-2+r)*r)*_csc(theta)**2)/(2*(a**2+r**2)**2),(-8*a*r**3*(r**2+a**2*np.cos(theta)**2)*(8*(3-4*r)*r**4+4*a**2*r**2*(-10+3*r)+a**4*(-3+39*r)+4*a**2*(r**2*(-10+9*r)+a**2*(-1+10*r))*np.cos(2*theta)+a**4*(-1+r)*np.cos(4*theta))*np.sin(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5),(-16*a*r**2*(a**2+(-2+r)*r)*np.cos(theta)*(-3*a**4+26*a**2*r**2+20*r**4-2*a**2*(a**2+5*r**2)*np.cos(2*theta)+a**4*np.cos(4*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**4),((4*1j)*r**2*(-(a*m)+(a**2+r**2)*omega)*(a**2-2*r**2+a**2*np.cos(2*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(-8*a*r**2*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)*(-3*a**4+4*r**3*(-2+5*r)+a**2*r*(10+17*r)+4*(2*a**4+a**2*r**2-(-2+r)*r**3)*np.cos(2*theta)+a**2*(11*a**2+r*(-10+11*r))*np.cos(4*theta))*_csc(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5),((-8*1j)*a*r**2*(a**2+(-2+r)*r)*(2*m-a*omega+a*omega*np.cos(2*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(4*(a*r**2*(a**2+(-2+r)*r)*_csc(theta)+(a*(-192*r**6*(a**2+r**2)**2+16*r**4*(5+2*r)*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))-2*r*(a**2+r**2)*(a**4+2*a**4*r+2*a**2*(3+1j*a*m)*r**2+4*a**2*r**3+(1+(2*1j)*a*m)*r**4+2*r**5)*(a**2+2*r**2+a**2*np.cos(2*theta))**2+(a**6+1j*a**5*m*r+(2*1j)*a**3*m*r**3+3*a**2*(-2+r)*r**3-r**5+1j*a*m*r**5+a**4*r*(-1+4*r))*(a**2+2*r**2+a**2*np.cos(2*theta))**3)*np.sin(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))**4))/(a**2+r**2)**4],[(8*(3-2*r)*r**8+3*a**8*(-1+9*r)+2*a**6*r**2*(-31+29*r)-4*a**2*r**5*(12+7*(-2+r)*r)+a**4*r**3*(12+r*(-39+19*r))+4*a**2*(a**6*(-1+6*r)+a**4*r**2*(-12+19*r)+a**2*r**4*(-29+20*r)+r**5*(12+r*(-22+7*r)))*np.cos(2*theta)-a**4*(2*a**2*r**2*(-7+3*r)+a**4*(1+3*r)+r**3*(12+r*(-11+3*r)))*np.cos(4*theta))/(r**4*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),((4*1j)*(-(a*m*(a**4-3*a**2*r**2+2*(4-3*r)*r**3))+(a**6+a**2*(4-3*r)*r**3-2*r**6)*omega+a**2*(a*m*(-a**2+r**2)+(a**4+2*a**2*r**2+(-4+r)*r**3)*omega)*np.cos(2*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,(4*a*(-48*r**2*(a**2+(-2+r)*r)*(a**2+(-1+r)*r)+2*(a**2+(-2+r)*r)*(9*a**2+r*(-6+7*r))*(a**2+2*r**2+a**2*np.cos(2*theta))+((a-r)*(-1+r)*(a+r)*(a**2+2*r**2+a**2*np.cos(2*theta))**2)/r)*np.sin(theta))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(28+(4*(3*a**4+(3*1j)*a**3*m*r+1j*a*m*(-4+r)*r**2-3*r**3+a**2*r*(-9+2*r)))/r**2-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+(a**2+(-2+r)*r)*(a**2*omega**2*np.cos(2*theta)+(8*(-4*r**3-(5*a**2+(2*1j)*a*m*r+2*(-5+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))))/(a**2+2*r**2+a**2*np.cos(2*theta))**2-2*m**2*_csc(theta)**2))/(2*(a**2+r**2)**2),(-8*r**2*(a**2+(-2+r)*r)**2*_cot(theta)-4*a**2*(2*a**2+1j*a*m*r+(-3+r)*r)*(a**2+(-2+r)*r)*np.sin(2*theta))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),((-8*1j)*m*r*(a**2+(-2+r)*r)**2*_csc(theta)+(2*1j)*a*((4*a*m*r**2*(-3+2*r)+a**3*m*(-1+5*r)+a**4*omega-5*a**2*r**2*omega+4*(4-3*r)*r**3*omega)*np.sin(theta)+a**2*(a*m*(-1+r)+(a-r)*(a+r)*omega)*np.sin(3*theta)))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),((a**2+(-2+r)*r)**2*(a**4+8*(-3+r)*r**3+4*a**2*r*(3+2*r)+12*a**2*r*np.cos(2*theta)-a**4*np.cos(4*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),0*r,((a**2+2*r**2+a**2*np.cos(2*theta))*((-96*a**2*r**2*(a**2+(-2+r)*r)*(a**2+(-1+r)*r)*(a**2+r**2)**2)/(a**2+2*r**2+a**2*np.cos(2*theta))**3+(4*a**2*(a**2+(-2+r)*r)*(a**2+r**2)*(9*a**4+r**3*(6+r)+2*a**2*r*(-3+5*r)))/(a**2+2*r**2+a**2*np.cos(2*theta))**2+(2*a**2*(a-r)*(a+r)*(2*a**2*r**2*(-8+5*r)+a**4*(-1+5*r)+r**3*(12+5*(-3+r)*r)))/(r*(a**2+2*r**2+a**2*np.cos(2*theta)))+(a**2+(-2+r)*r)**2*((-3+r)*r**2+a**2*(3+r))*_csc(theta)**2)*np.sin(theta)**2)/(r*(a**2+r**2)**5)],[(-2*a**2*(3*a**6+a**2*(32-33*r)*r**3+2*(16-9*r)*r**5-4*a**4*r*(1+3*r)+a**2*(3*a**4+r**3*(-8+3*r)+2*a**2*r*(-2+3*r))*np.cos(2*theta))*np.sin(2*theta))/(r**3*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),((4*1j)*a**2*(-(a*m)+(a**2+r**2)*omega)*np.sin(2*theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),((2*1j)*(-(a*m*(a**4-3*a**2*r**2+2*(4-3*r)*r**3))+(a**6+a**2*(4-3*r)*r**3-2*r**6)*omega+a**2*(a*m*(-a**2+r**2)+(a**4+2*a**2*r**2+(-4+r)*r**3)*omega)*np.cos(2*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(4*a*np.sin(theta)*(-4*(a**2+(-2+r)*r)*(a**2-3*r**2)*(a**2+r**2)*_cot(theta)+2*a**2*r*(a**2*(-1+8*r)+r**2*(-16+9*r))*np.sin(2*theta)+a**4*(-a**2+r)*np.sin(4*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),(2*a**2*(3*a**2+1j*a*m*r+r*(-5+2*r))*np.sin(2*theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(16+(4*(a**4+(2*1j)*a**3*m*r+1j*a*m*(-3+r)*r**2+2*(-3+r)*r**3+a**2*r*(-4+3*r)))/r**2-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+(a**2+(-2+r)*r)*(a**2*omega**2*np.cos(2*theta)+(16*(4*r**3-(2*a**2+1j*a*m*r+2*(-1+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))))/(a**2+2*r**2+a**2*np.cos(2*theta))**2-2*(1+m**2)*_csc(theta)**2))/(2*(a**2+r**2)**2),((-4*1j)*_csc(theta)*(m*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(2*theta))*_cot(theta)+a**3*(a*m-2*r*omega)*np.cos(theta)*np.sin(theta)**3))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(-2*(a**2+(-2+r)*r)*(2*r**2*(a**2+(-2+r)*r)*_cot(theta)+a**2*(a**2+1j*a*m*r+(-2+r)*r)*np.sin(2*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),((-4*1j)*m*r*(a**2+(-2+r)*r)**2*_csc(theta)+1j*a*((4*a*m*r**2*(-3+2*r)+a**3*m*(-1+5*r)+a**4*omega-5*a**2*r**2*omega+4*(4-3*r)*r**3*omega)*np.sin(theta)+a**2*(a*m*(-1+r)+(a-r)*(a+r)*omega)*np.sin(3*theta)))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(np.sin(theta)**2*(-8*(a**2+(-2+r)*r)*_cot(theta)*(3*a**2*(-4+r)*r**2+a**4*(2+3*r)-r*(a**2+(-2+r)*r)*(a**2+r**2)*_csc(theta)**2)+2*a**4*(a**2*(3+5*r)+2*r*(-4+3*(-1+r)*r))*np.sin(2*theta)+a**6*(-1+r)*np.sin(4*theta)))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2)],[0*r,(-2*a*(3*a**8+(3*1j)*a**7*m*r-(3*1j)*a**5*m*r**3+31*a**4*(-2+r)*r**3-(24*1j)*a**3*m*r**5-24*a**2*r**6-(24*1j)*a*m*r**7-24*(-2+r)*r**7+2*a**6*r*(-3+17*r)+4*a**2*(a**6+1j*a**5*m*r-1j*a**3*m*r**3-(4*1j)*a*m*r**5+6*(-2+r)*r**5+2*a**4*r*(-1+4*r)+a**2*r**3*(-14+13*r))*np.cos(2*theta)+a**4*(a**4+1j*a**3*m*r-1j*a*m*r**3-3*(-2+r)*r**3-2*a**2*r*(1+r))*np.cos(4*theta))*np.sin(theta))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),(4*a*(a**2+(-2+r)*r)*np.cos(theta)*(5*a**6-21*a**4*r**2-56*a**2*r**4-24*r**6+4*a**2*(a**4+3*a**2*r**2+4*r**4)*np.cos(2*theta)+a**4*(-a**2+r**2)*np.cos(4*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),((4*1j)*(r**2+a**2*np.cos(theta)**2)*(-(a*m*(a**4-3*a**2*r**2+2*(4-3*r)*r**3))+(a**6+a**2*(4-3*r)*r**3-2*r**6)*omega+a**2*(a*m*(-a**2+r**2)+(a**4+2*a**2*r**2+(-4+r)*r**3)*omega)*np.cos(2*theta)))/(r**2*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),((1j/8)*r*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(8*(-2+r)*r**4+4*a**2*r**2*(-3+2*r)+a**4*(1+3*r))+a*(-a**4+5*a**2*r**2+12*r**4)*omega+4*a*r*(a*m*(a**2+r*(-1+2*r))-r*(a**2+3*r**2)*omega)*np.cos(2*theta)+a**3*(a*m*(-1+r)+(a-r)*(a+r)*omega)*np.cos(4*theta))*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),((1j/8)*(a**2+(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(3*a**4+8*r**4+8*a**2*r*(1+r))-6*a**3*r*omega+4*a**2*(a**2*m+2*m*(-1+r)*r+2*a*r*omega)*np.cos(2*theta)+a**3*(a*m-2*r*omega)*np.cos(4*theta))*_cot(theta)*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),(((8*1j)*a**3*m*r-8*a*m*r**2*(1j+r*omega)+a**4*(4+r**2*omega**2)+2*r**3*(4-2*r+r**3*omega**2)+a**2*r*(-8+r**2*(2+3*r)*omega**2))/r**2+a**2*(a**2+(-2+r)*r)*omega**2*np.cos(2*theta)+(8*(48*r**3*(a**2+(-2+r)*r)*(a**2+r**2)-4*r*(3*a**4-1j*a**3*m*r-1j*a*m*r**3+6*(-2+r)*r**3+3*a**2*r*(-2+3*r))*(a**2+2*r**2+a**2*np.cos(2*theta))-(((-6+r)*(-2+r)*r**3+1j*a*m*(-1+r)*r**3+2*a**4*(1+r)+1j*a**3*m*r*(1+r)+a**2*r*(-4+r*(-8+3*r)))*(a**2+2*r**2+a**2*np.cos(2*theta))**2)/r))/(a**2+2*r**2+a**2*np.cos(2*theta))**3-2*(1+m**2)*(a**2+(-2+r)*r)*_csc(theta)**2)/(2*(a**2+r**2)**2),0*r,((a**2+(-2+r)*r)*(-4*(a**2+(-2+r)*r)*_cot(theta)+(a**2*(-a**2-1j*a*m*r+r**2-(48*r**3*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))**2+(8*r*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta)))*np.sin(2*theta))/r**2))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)),((-4*1j)*(r**2+a**2*np.cos(theta)**2)*(a*(2*a*m*(3-2*r)*r**2+a**3*(m-3*m*r)-a**4*omega+3*a**2*r**2*omega+2*r**3*(-4+3*r)*omega)+a**3*(a*(m-m*r)-a**2*omega+r**2*omega)*np.cos(2*theta)+2*m*r*(a**2+(-2+r)*r)**2*_csc(theta)**2)*np.sin(theta))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2)],[(4*(r**2+a**2*np.cos(theta)**2)*(-9*a**6+a**4*(4-7*r)*r+4*r**6+2*a**2*r**3*(-4+3*r)-2*a**2*(3*a**4+8*a**2*r**2+r**3*(-4+5*r))*np.cos(2*theta)+a**4*(3*a**2+r*(-4+3*r))*np.cos(4*theta)))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),0*r,((8*1j)*a**2*(-(a*m)+(a**2+r**2)*omega)*np.sin(2*theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),(8*a*(r**2+a**2*np.cos(theta)**2)*(-15*a**4-2*a**2*(-8+r)*r+4*r**3*(-4+3*r)-2*a**2*(7*a**2+r*(-8+7*r))*np.cos(2*theta)+a**4*np.cos(4*theta))*np.sin(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),(r**2*(2*(-3+r)*r**3+a**2*r*(3+2*r)+3*a**2*r*np.cos(2*theta)+a**4*np.cos(theta)*np.sin(theta)*np.sin(2*theta)))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**2),(4*a**2*(2*a**2+1j*a*m*r+2*(-2+r)*r)*np.sin(2*theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,((4*(-1+1j*a*m+2*r)*(a**2+(-2+r)*r))/r-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+(a**2+(-2+r)*r)*(a**2*omega**2*np.cos(2*theta)+(8*(-4*r**3-(a**2+(2*1j)*a*m*r+2*(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))))/(a**2+2*r**2+a**2*np.cos(2*theta))**2-2*(2+m**2)*_csc(theta)**2))/(2*(a**2+r**2)**2),((-16*1j)*(r**2+a**2*np.cos(theta)**2)*_csc(theta)*(m*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(2*theta))*_cot(theta)+a**3*(a*m-2*r*omega)*np.cos(theta)*np.sin(theta)**3))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2),-((3*a**2+2*(-9+r)*r+a**2*np.cos(2*theta)+(4*r*(9*a**4+2*r**3*(4+5*r)+a**2*r*(-8+11*r)+a**2*(9*a**2+r*(-8+17*r))*np.cos(2*theta)))/(a**2+2*r**2+a**2*np.cos(2*theta))**2-2*(a**2+(-2+r)*r)*_csc(theta)**2)/(a**2+r**2)**2)],[0*r,(16*a**3*(a**2+(-2+r)*r)*np.cos(theta)*(a**2-10*r**2+a**2*np.cos(2*theta))*np.sin(theta)**2)/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),(2*a*((-1j)*a**7*m+16*a**6*r+1j*a**5*m*r**2+(16*1j)*a**3*m*r**4+4*a**2*(10-17*r)*r**4+(24*1j)*a*m*r**6-48*(-2+r)*r**6-4*a**4*r**2*(8+r))*np.sin(theta)+a**3*((-3*1j)*a**5*m+32*a**4*r+(3*1j)*a**3*m*r**2+(16*1j)*a*m*r**4+24*(-2+r)*r**4+8*a**2*r**2*(-8+7*r))*np.sin(3*theta)-1j*a**6*m*(a-r)*(a+r)*np.sin(5*theta))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**3),((4*1j)*a**2*(-(a*m)+(a**2+r**2)*omega)*np.sin(2*theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))),0*r,((1j/8)*r*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(8*(-2+r)*r**4+4*a**2*r**2*(-3+2*r)+a**4*(1+3*r))+a*(-a**4+5*a**2*r**2+12*r**4)*omega+4*a*r*(a*m*(a**2+r*(-1+2*r))-r*(a**2+3*r**2)*omega)*np.cos(2*theta)+a**3*(a*m*(-1+r)+(a-r)*(a+r)*omega)*np.cos(4*theta))*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),(2*(-(r**2*(a**2+(-2+r)*r)*_cot(theta))+((-48*a**2*r**3*(a**2+(-2+r)*r)*(a**2+r**2)+4*a**2*r*(a**2+(-2+r)*r)*(a**2+r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))+(2*a**4+1j*a**3*m*r+1j*a*m*r**3+(-2+r)*r**3+a**2*r*(-4+3*r))*(a**3+2*a*r**2+a**3*np.cos(2*theta))**2)*np.sin(2*theta))/(a**2+2*r**2+a**2*np.cos(2*theta))**3))/(a**2+r**2)**3,((1j/8)*(a**2+(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(3*a**4+8*r**4+8*a**2*r*(1+r))-6*a**3*r*omega+4*a**2*(a**2*m+2*m*(-1+r)*r+2*a*r*omega)*np.cos(2*theta)+a**3*(a*m-2*r*omega)*np.cos(4*theta))*_cot(theta)*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),(8*a**2-(4*1j)*a*m+((4*1j)*a**3*m)/r+8*(-2+r)*r-8*a*m*r*omega+(a**4+2*r**4+a**2*r*(2+3*r))*omega**2+a**2*(a**2+(-2+r)*r)*omega**2*np.cos(2*theta)+(8*(-48*r**3*(a**2+(-2+r)*r)*(a**2+r**2)+4*r*(5*a**4+1j*a**3*m*r+1j*a*m*r**3+9*(-2+r)*r**3+2*a**2*r*(-5+7*r))*(a**2+2*r**2+a**2*np.cos(2*theta))-((1j*a*m*(-1+r)*r**3+a**4*(1+r)+1j*a**3*m*r*(1+r)+(-2+r)*r**3*(7+r)+2*a**2*r*(-1+r*(3+r)))*(a**2+2*r**2+a**2*np.cos(2*theta))**2)/r))/(a**2+2*r**2+a**2*np.cos(2*theta))**3-2*(4+m**2)*(a**2+(-2+r)*r)*_csc(theta)**2)/(2*(a**2+r**2)**2),((-8*1j)*(r**2+a**2*np.cos(theta)**2)*_csc(theta)*(m*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(2*theta))*_cot(theta)+a**3*(a*m-2*r*omega)*np.cos(theta)*np.sin(theta)**3))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**2)],[(2*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)*((-4*(3*a**2-2*r**2+3*a**2*np.cos(2*theta))*(a**4+2*r**4+a**2*r*(2+3*r)+a**2*(a**2+(-2+r)*r)*np.cos(2*theta))**2)/((a**2+(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))**5)+(a**2*(r**2*(a**2+3*r**2)+a**2*(-a+r)*(a+r)*np.cos(theta)**2)**2*np.sin(theta)**2)/(r*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)**5)+(4*a**6*r*np.cos(theta)**2*np.sin(theta)**4)/(r**2+a**2*np.cos(theta)**2)**5))/(r*(a**2+r**2)**2),0*r,0*r,(8*a*(192*r**5*(a**2+r**2)+16*r**3*(r**2*(-11+4*r)+a**2*(-5+4*r))*(a**2+2*r**2+a**2*np.cos(2*theta))+2*(a**4+(2*1j)*a**3*m*r**2+(2*1j)*a*m*r**4-2*a**2*r**2*(4+r)+r**3*(20-r*(9+2*r)))*(a**2+2*r**2+a**2*np.cos(2*theta))**2+(a-r)*(a+r)*(-1-1j*a*m+r)*(a**2+2*r**2+a**2*np.cos(2*theta))**3)*np.sin(theta))/(r*(a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**4),(2*r**2*(r**2+a**2*np.cos(theta)**2)*(32*(-3+r)*r**7+48*a**2*r**4*(1+(-3+r)*r)+2*a**6*(-1+r*(38+5*r))+4*a**4*r**2*(-10+r*(8+9*r))+a**2*(48*a**2*r**3*(2+r)+48*r**4*(-1+r+r**2)+a**4*(-1+5*r*(10+3*r)))*np.cos(2*theta)+2*a**4*(a**2*(1+r*(-14+3*r))+2*r**2*(10+r*(-8+3*r)))*np.cos(4*theta)+a**6*(-1+r)**2*np.cos(6*theta)))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5),(4*r*(a**2+(-2+r)*r)*(r**2+a**2*np.cos(theta)**2)*(2*(5*a**6*(-1+r)+16*r**7+8*a**2*r**4*(10+3*r)+2*a**4*r**2*(35+9*r))+a**2*(48*a**2*(-3+r)*r**2+16*r**4*(-10+3*r)+a**4*(1+15*r))*np.cos(2*theta)+2*a**4*(2*r**2*(1+3*r)+a**2*(5+3*r))*np.cos(4*theta)+a**6*(-1+r)*np.cos(6*theta))*_cot(theta))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**5),((1j/4)*r*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(8*(-2+r)*r**4+4*a**2*r**2*(-3+2*r)+a**4*(1+3*r))+a*(-a**4+5*a**2*r**2+12*r**4)*omega+4*a*r*(a*m*(a**2+r*(-1+2*r))-r*(a**2+3*r**2)*omega)*np.cos(2*theta)+a**3*(a*m*(-1+r)+(a-r)*(a+r)*omega)*np.cos(4*theta))*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),((a**2+(-2+r)*r)*(2*(-35*a**8-16*(-2+r)*r**7-5*a**6*r*(4+25*r)+2*a**4*r**2*(10+9*(1-9*r)*r)-8*a**2*r**4*(2+r*(-9+11*r)))-a**2*(47*a**6+16*a**2*r**3*(1+9*r)+a**4*r*(2+143*r)+16*r**4*(-2+r+3*r**2))*np.cos(2*theta)-a**4*(a**2+(-2+r)*r)*(2*(5*a**2+2*r*(-5+3*r))*np.cos(4*theta)+a**2*np.cos(6*theta))+32*(a**2+r**2)**4*_csc(theta)**2))/((a**2+r**2)**2*(a**2+2*r**2+a**2*np.cos(2*theta))**4),((1j/4)*(a**2+(-2+r)*r)*(a**2+2*r**2+a**2*np.cos(2*theta))*(m*(3*a**4+8*r**4+8*a**2*r*(1+r))-6*a**3*r*omega+4*a**2*(a**2*m+2*m*(-1+r)*r+2*a*r*omega)*np.cos(2*theta)+a**3*(a*m-2*r*omega)*np.cos(4*theta))*_cot(theta)*_csc(theta))/((a**2+r**2)**2*(r**2+a**2*np.cos(theta)**2)**3),(-4+((4*1j)*a**3*m)/r+4*r+a**4*omega**2+2*r**4*omega**2-4*a*m*r*(1j+2*omega)+(a**2*(4+r*(-4+r*(2+3*r)*omega**2)))/r+a**2*(a**2+(-2+r)*r)*omega**2*np.cos(2*theta)+(16*(96*r**5*(a**2+r**2)+8*r**3*(a**2*(-5+r)+(-11+r)*r**2)*(a**2+2*r**2+a**2*np.cos(2*theta))+(a**4+4*a**4*r+2*a**2*(-3+(2*1j)*a*m)*r**2+4*(5+2*a**2)*r**3+(-15+(4*1j)*a*m)*r**4+4*r**5)*(a**2+2*r**2+a**2*np.cos(2*theta))**2-((a**4+1j*a**3*m*r+1j*a*m*r**3+r**3*(-5+2*r)+a**2*r*(-1+3*r))*(a**2+2*r**2+a**2*np.cos(2*theta))**3)/r))/(a**2+2*r**2+a**2*np.cos(2*theta))**4-2*(2+m**2)*(a**2+(-2+r)*r)*_csc(theta)**2)/(2*(a**2+r**2)**2)]]
    return _stack(_rows, r.shape)


# --------------------------------------------------------------------------
# The radial boundary conditions
# --------------------------------------------------------------------------


def near_horizon_boundary_condition(m, a, omega, r, theta, drstar, dtheta):
    """The near-horizon condition at the inner edge (radial offsets 0..6: the seven radii of the strip
    x_A, App. A). See the module docstring."""
    theta = np.asarray(theta, dtype=float)
    _one = np.ones_like(theta, dtype=complex)
    gamma = np.sqrt(1-a**2)/(1+np.sqrt(1-a**2))
    b0 = ((-1260 + drstar*(-7*gamma*(3000 + gamma*drstar*(11025 + 16*gamma*drstar*(725 + 189*gamma*drstar))) + 
      (60*1j)*(175 + 2*gamma*drstar*(735 + gamma*drstar*(1421 + 3*gamma*drstar*(245 + 24*gamma*drstar))))*omega + 
      30*drstar*(735 + gamma*drstar*(3248 + 3*gamma*drstar*(1029 + 200*gamma*drstar)))*omega**2 - (280*1j)*drstar**2*(58 + 9*gamma*drstar*(14 + 5*gamma*drstar))*
       omega**3 - 90*drstar**3*(49 + 40*gamma*drstar)*omega**4 + (360*1j)*drstar**4*omega**5)))/(360*drstar**5)
    b1 = ((20 + drstar*(gamma*(310 + gamma*drstar*(1015 + 6*gamma*drstar*(145 + 24*gamma*drstar))) - 
      1j*(155 + gamma*drstar*(1160 + 3*gamma*drstar*(609 + 200*gamma*drstar)))*omega - 2*drstar*(145 + 9*gamma*drstar*(58 + 35*gamma*drstar))*omega**2 + 
      (6*1j)*drstar**2*(29 + 40*gamma*drstar)*omega**3 + 30*drstar**3*omega**4)))/drstar**5
    b2 = (5*(-76 + drstar*(-(gamma*(1096 + gamma*drstar*(3227 + 36*gamma*drstar*(65 + 8*gamma*drstar)))) + 
      (2*1j)*(274 + gamma*drstar*(1844 + 3*gamma*drstar*(819 + 200*gamma*drstar)))*omega + 2*drstar*(461 + 18*gamma*drstar*(78 + 35*gamma*drstar))*omega**2 - 
      (12*1j)*drstar**2*(39 + 40*gamma*drstar)*omega**3 - 60*drstar**3*omega**4)))/(8*drstar**5)
    b3 = (10*(54 + drstar*(gamma*(726 + gamma*drstar*(1953 + 2*gamma*drstar*(635 + 72*gamma*drstar))) - 
      (3*1j)*(121 + gamma*drstar*(744 + gamma*drstar*(889 + 200*gamma*drstar)))*omega - 6*drstar*(93 + gamma*drstar*(254 + 105*gamma*drstar))*omega**2 + 
      (2*1j)*drstar**2*(127 + 120*gamma*drstar)*omega**3 + 30*drstar**3*omega**4)))/(9*drstar**5)
    b4 = (5*(-68 + drstar*(-(gamma*(856 + gamma*drstar*(2149 + 24*gamma*drstar*(55 + 6*gamma*drstar)))) + 
      (4*1j)*(107 + gamma*drstar*(614 + 3*gamma*drstar*(231 + 50*gamma*drstar)))*omega + 2*drstar*(307 + 9*gamma*drstar*(88 + 35*gamma*drstar))*omega**2 - 
      (24*1j)*drstar**2*(11 + 10*gamma*drstar)*omega**3 - 30*drstar**3*omega**4)))/(8*drstar**5)
    b5 = ((80 + drstar*(gamma*(950 + gamma*drstar*(2275 + 18*gamma*drstar*(75 + 8*gamma*drstar))) - 
      (5*1j)*(95 + gamma*drstar*(520 + 3*gamma*drstar*(189 + 40*gamma*drstar)))*omega - 10*drstar*(65 + 9*gamma*drstar*(18 + 7*gamma*drstar))*omega**2 + 
      (30*1j)*drstar**2*(9 + 8*gamma*drstar)*omega**3 + 30*drstar**3*omega**4)))/(5*drstar**5)
    b6 = ((-180 + drstar*(-(gamma*(2040 + gamma*drstar*(4725 + 4*gamma*drstar*(685 + 72*gamma*drstar)))) + 
      (6*1j)*(170 + gamma*drstar*(900 + gamma*drstar*(959 + 200*gamma*drstar)))*omega + 6*drstar*(225 + 2*gamma*drstar*(274 + 105*gamma*drstar))*omega**2 - 
      (4*1j)*drstar**2*(137 + 120*gamma*drstar)*omega**3 - 60*drstar**3*omega**4)))/(72*drstar**5)
    entries = []
    entries.append((None, 0, None, 0, 0, [(((1j/8)*(r*(-2*a*m+r**3*omega+a**2*(2+r)*omega)+a**2*(a**2+(-2+r)*r)*omega*np.cos(theta)**2)*(a**2+2*r**2+a**2*np.cos(2*theta))**3)/(r*(r**2+a**2*np.cos(theta)**2)**4)) * _one, ((1j*r*(r*(-2*a*m+r**3*omega+a**2*(2+r)*omega)+a**2*(a**2+(-2+r)*r)*omega*np.cos(theta)**2))/(r**2+a**2*np.cos(theta)**2)) * _one, ((1j*(-2*a*m*r+r**4*omega+a**2*r*(2+r)*omega+a**2*(a**2+(-2+r)*r)*omega*np.cos(theta)**2))/(r**2+a**2*np.cos(theta)**2)) * _one, ((1j*(-4*a*m*r+a**4*omega+2*r**4*omega+a**2*r*(2+3*r)*omega+a**2*(a**2+(-2+r)*r)*omega*np.cos(2*theta))*np.sin(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one, (b0) * _one, (b0) * _one, (b0) * _one, (b0) * _one, (b0) * _one, (b0) * _one]))
    entries.append((None, 0, None, 1, 0, [(0) * _one, (0) * _one, (0) * _one, (0) * _one, (b1) * _one, (b1) * _one, (b1) * _one, (b1) * _one, (b1) * _one, (b1) * _one]))
    entries.append((None, 0, None, 2, 0, [(0) * _one, (0) * _one, (0) * _one, (0) * _one, (b2) * _one, (b2) * _one, (b2) * _one, (b2) * _one, (b2) * _one, (b2) * _one]))
    entries.append((None, 0, None, 3, 0, [(0) * _one, (0) * _one, (0) * _one, (0) * _one, (b3) * _one, (b3) * _one, (b3) * _one, (b3) * _one, (b3) * _one, (b3) * _one]))
    entries.append((None, 0, None, 4, 0, [(0) * _one, (0) * _one, (0) * _one, (0) * _one, (b4) * _one, (b4) * _one, (b4) * _one, (b4) * _one, (b4) * _one, (b4) * _one]))
    entries.append((None, 0, None, 5, 0, [(0) * _one, (0) * _one, (0) * _one, (0) * _one, (b5) * _one, (b5) * _one, (b5) * _one, (b5) * _one, (b5) * _one, (b5) * _one]))
    entries.append((None, 0, None, 6, 0, [(0) * _one, (0) * _one, (0) * _one, (0) * _one, (b6) * _one, (b6) * _one, (b6) * _one, (b6) * _one, (b6) * _one, (b6) * _one]))
    entries.append((0, 0, 1, 0, 0, ((2*(a**2+1j*a*m*r+(-2+r)*r))/(a**2+2*r**2+a**2*np.cos(2*theta)) + _d1_fwd0(drstar)*(2*r*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((0, 0, 2, 0, 0, ((2*(a**2+(-2+r)*r)*_cot(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((0, 0, 3, 0, 0, ((1j*(a**2*m+2*m*(-2+r)*r+2*a*r*omega+a*(a*m-2*r*omega)*np.cos(2*theta))*_csc(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((0, 0, 1, 1, 0, (_d1_fwd1(drstar)*(2*r*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((0, 0, 1, 2, 0, (_d1_fwd2(drstar)*(2*r*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((0, 0, 2, 0, -1, (_d1_down(dtheta)*((a**2+(-2+r)*r))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((0, 0, 2, 0, 1, (_d1_up(dtheta)*((a**2+(-2+r)*r))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((1, 0, 0, 0, 0, (-1/4*((a**2+2*r**2+a**2*np.cos(2*theta))*(a**6+a**2*(4-3*r)*r**3-2*r**6+a**2*(a**4+2*a**2*r**2+(-4+r)*r**3)*np.cos(2*theta)))/(r*(r**2+a**2*np.cos(theta)**2)**3)) * _one))
    entries.append((1, 0, 3, 0, 0, (-(4*a*(a**4-3*a**2*r**2+2*(4-3*r)*r**3+a**2*(a**2-r**2)*np.cos(2*theta))*np.sin(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))**2) * _one))
    entries.append((1, 0, 4, 0, 0, ((2*r**2*(3*a**4+1j*a**3*m*r+(2*1j)*a*m*r**3+2*(-3+r)*r**3+a**2*r*(-5+6*r)+a**2*(3*a**2+1j*a*m*r+r*(-5+2*r))*np.cos(2*theta)))/(a**2+2*r**2+a**2*np.cos(2*theta))**2 + _d1_fwd0(drstar)*(2*r**3*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((1, 0, 5, 0, 0, ((2*r**2*(a**2+(-2+r)*r)*_cot(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((1, 0, 6, 0, 0, ((1j*r**2*(2*a*r*omega+a**2*m*_cot(theta)**2+m*(-2+r)*r*_csc(theta)**2)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((1, 0, 7, 0, 0, (-(r**2*(a**2+(-2+r)*r)**2)/(r**2+a**2*np.cos(theta)**2)**2) * _one))
    entries.append((1, 0, 9, 0, 0, (-(r*(a**4+3*a**4*r-20*a**2*r**2+32*r**3+8*a**2*r**3-32*r**4+8*r**5+4*a**2*r*(a**2+r*(-3+2*r))*np.cos(2*theta)+a**4*(-1+r)*np.cos(4*theta)))/(2*(a**2+2*r**2+a**2*np.cos(2*theta))**2)) * _one))
    entries.append((1, 0, 4, 1, 0, (_d1_fwd1(drstar)*(2*r**3*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((1, 0, 4, 2, 0, (_d1_fwd2(drstar)*(2*r**3*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((1, 0, 5, 0, -1, (_d1_down(dtheta)*(r**2*(a**2+(-2+r)*r))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((1, 0, 5, 0, 1, (_d1_up(dtheta)*(r**2*(a**2+(-2+r)*r))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((2, 0, 0, 0, 0, ((-4*a**2*(a**2+r**2)*np.sin(2*theta))/(a**2+2*r**2+a**2*np.cos(2*theta))**2) * _one))
    entries.append((2, 0, 3, 0, 0, (-(8*a**3*r*np.sin(theta)*np.sin(2*theta))/(a**2+2*r**2+a**2*np.cos(2*theta))**2) * _one))
    entries.append((2, 0, 4, 0, 0, ((a**2*r**3*np.cos(theta)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)**2) * _one))
    entries.append((2, 0, 5, 0, 0, ((2*r*(2*a**2+1j*a*m*r+2*(-2+r)*r))/(a**2+2*r**2+a**2*np.cos(2*theta)) + _d1_fwd0(drstar)*(2*r**2*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((2, 0, 7, 0, 0, ((4*r*(a**2+(-2+r)*r)*(a**2+r**2)*_cot(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))**2) * _one))
    entries.append((2, 0, 8, 0, 0, ((1j*r*(2*a*r*omega+a**2*m*_cot(theta)**2+m*(-2+r)*r*_csc(theta)**2)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((2, 0, 9, 0, 0, (-(r*(3*a**4+8*a**2*r**2+8*(-2+r)*r**3+4*a**2*(a**2+2*(-2+r)*r)*np.cos(2*theta)+a**4*np.cos(4*theta))*_cot(theta))/(2*(a**2+2*r**2+a**2*np.cos(2*theta))**2)) * _one))
    entries.append((2, 0, 5, 1, 0, (_d1_fwd1(drstar)*(2*r**2*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((2, 0, 5, 2, 0, (_d1_fwd2(drstar)*(2*r**2*(a**2+r**2))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((2, 0, 7, 0, -1, (_d1_down(dtheta)*(r*(a**2+(-2+r)*r))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((2, 0, 7, 0, 1, (_d1_up(dtheta)*(r*(a**2+(-2+r)*r))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((3, 0, 6, 0, 0, ((2*r*(2*a**2+1j*a*m*r+2*(-2+r)*r)*np.sin(theta))/(a**2+2*r**2+a**2*np.cos(2*theta)) + _d1_fwd0(drstar)*(r**2*(a**2+r**2)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((3, 0, 8, 0, 0, ((4*r*(a**2+(-2+r)*r)*np.cos(theta))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((3, 0, 9, 0, 0, ((1j*r*(a**2*m+2*m*(-2+r)*r+2*a*r*omega+a*(a*m-2*r*omega)*np.cos(2*theta)))/(a**2+2*r**2+a**2*np.cos(2*theta))) * _one))
    entries.append((3, 0, 6, 1, 0, (_d1_fwd1(drstar)*(r**2*(a**2+r**2)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((3, 0, 6, 2, 0, (_d1_fwd2(drstar)*(r**2*(a**2+r**2)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((3, 0, 8, 0, -1, (_d1_down(dtheta)*(r*(a**2+(-2+r)*r)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)) * _one))
    entries.append((3, 0, 8, 0, 1, (_d1_up(dtheta)*(r*(a**2+(-2+r)*r)*np.sin(theta))/(r**2+a**2*np.cos(theta)**2)) * _one))
    return entries


def outer_boundary_condition(m, a, omega, r, theta, drstar, dtheta):
    """Outgoing radiation condition at the outer edge, for m != 0 (radial offsets 0..-4: the five radii of
    the strip x_C, App. A)."""
    theta = np.asarray(theta, dtype=float)
    _one = np.ones_like(theta, dtype=complex)
    entries = []
    entries.append((None, 0, None, 0, 0, [(((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one, (((10 + 1j*drstar*omega*(-35 + drstar*omega*(25*1j + 4*drstar*omega))))/(4*drstar**3)) * _one]))
    entries.append((None, 0, None, -1, 0, [(((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one, (((-9 + 2*drstar*omega*(13*1j + 6*drstar*omega)))/drstar**3) * _one]))
    entries.append((None, 0, None, -2, 0, [(- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one, (- (3*(1j + 2*drstar*omega)*(8*1j + 3*drstar*omega))/(2*drstar**3)) * _one]))
    entries.append((None, 0, None, -3, 0, [(((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one, (((-7 + 2*drstar*omega*(7*1j + 2*drstar*omega)))/drstar**3) * _one]))
    entries.append((None, 0, None, -4, 0, [(-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one, (-1/4*((3*1j + drstar*omega)*(2*1j + 3*drstar*omega))/drstar**3) * _one]))
    return entries


def outer_boundary_condition_m0(m, a, omega, r, theta, drstar, dtheta):
    """Static outgoing condition at the outer edge, for m = 0."""
    theta = np.asarray(theta, dtype=float)
    _one = np.ones_like(theta, dtype=complex)
    entries = []
    entries.append((None, 0, None, 0, 0, [(- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one, (- 3 / (2*drstar)) * _one]))
    entries.append((None, 0, None, -1, 0, [(2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one, (2 / drstar) * _one]))
    entries.append((None, 0, None, -2, 0, [(-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one, (-1 / (2*drstar)) * _one]))
    return entries


# ======================================================================
# Assembly
# ======================================================================

@dataclass
class AssembledSystem:
    """The assembled sparse matrix M together with what is needed to interpret it."""

    M: sp.csr_matrix
    m: int
    orbit: Orbit
    grid: Grid
    n: int = N_COMPONENTS
    timings: dict = _dc_field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.M.shape[0]

    @property
    def omega(self) -> float:
        return self.orbit.mode_frequency(self.m)

    def index(self, l: int, i: int, j: int) -> int:
        """Global row/column index of field l at grid node (i, j)."""
        return unknown_index(l, i, j, self.grid.n_rstar)


def polar_condition_classes(m: int) -> np.ndarray:
    """
    Which fields get a Dirichlet condition at the poles, per mode number.

    A 1 means the component is set to zero at theta = 0 and pi; a 0 means its
    one-sided theta derivative is set to zero instead. Which fields vanish on
    the axis depends on their spin weight and on m, so the pattern changes for
    |m| = 0, 1, 2 and settles down for |m| >= 3.
    """
    if m == 0:
        dirichlets = [0, 0, 1, 1, 0, 1, 1, 1, 1, 1]
    elif abs(m) == 1:
        dirichlets = [1, 1, 0, 0, 1, 0, 0, 1, 1, 1]
    elif abs(m) == 2:
        dirichlets = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    else:
        dirichlets = [1] * N_COMPONENTS
    return np.asarray(dirichlets, dtype=float)


def _coefficient_arrays(
    m: int, a: float, omega: float, r: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Q_r*, Q_theta and Q_0 over the grid, as arrays of shape (N_theta, 10, 10, N_r*).

    Boundary nodes are left at zero, which is what keeps interior stencils out
    of boundary rows. The layout is theta-major so that a slice at fixed
    (l, p) is contiguous in the raveled interior-node order used below.
    """
    nr, nth = len(r), len(theta)
    shape = (nth, N_COMPONENTS, N_COMPONENTS, nr)
    Q_rstar = np.zeros(shape, dtype=complex)
    Q_theta = np.zeros(shape, dtype=complex)
    Q_0 = np.zeros(shape, dtype=complex)

    r_int = r[1:-1]
    for j in range(1, nth - 1):
        th = theta[j]
        Q_rstar[j, :, :, 1:-1] = first_rstar_derivative_coefficients(m, a, omega, r_int, th)
        Q_theta[j, :, :, 1:-1] = first_theta_derivative_coefficients(m, a, omega, r_int, th)
        Q_0[j, :, :, 1:-1] = non_derivative_coefficients(m, a, omega, r_int, th)
    return Q_rstar, Q_theta, Q_0


def assemble(
    m: int, orbit: Orbit, grid: Grid, *, drop_tol: float = 0.0
) -> AssembledSystem:
    """
    Assemble the sparse matrix M for mode `m` on `grid` (Secs. II C and III).

    `drop_tol` discards entries with |value| <= drop_tol. The default of zero
    keeps every entry the stencils produce, including any that happen to
    vanish numerically, so the sparsity pattern is a property of the
    discretization and not of the particular r0 and a.
    """
    if grid.a != orbit.a:
        raise ValueError(
            f"grid was built for spin a = {grid.a} but the orbit has a = {orbit.a}"
        )

    t0 = time.perf_counter()

    n = N_COMPONENTS
    nr, nth = grid.n_rstar, grid.n_theta
    if nr < 7:
        raise ValueError(
            f"the near-horizon condition reaches six nodes into the grid, so it "
            f"needs more than {nr} radial nodes"
        )
    if nth < 4:
        raise ValueError(f"the polar conditions need at least 4 polar nodes, got {nth}")

    N = n * nr * nth
    drs, dth = grid.delta_rstar, grid.delta_theta
    a = orbit.a
    omega = orbit.mode_frequency(m)

    Q_rstar, Q_theta, Q_0 = _coefficient_arrays(m, a, omega, grid.r, grid.theta)
    t_coef = time.perf_counter()

    # Interior node indices, theta-major so that raveling matches the
    # coefficient arrays' (Nth, Nr) layout.
    ii = np.arange(1, nr - 1)
    jj = np.arange(1, nth - 1)
    JJ, II = np.meshgrid(jj, ii, indexing="ij")
    base = (n * (II + nr * JJ)).ravel()  # index(0, i, j) at every interior node

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    vals_parts: list[np.ndarray] = []

    def append_triplet_batch(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> None:
        """Append one vectorized batch of triplets, dropping structural zeros."""
        keep = np.abs(vals) > drop_tol if drop_tol > 0 else vals != 0
        if not keep.any():
            return
        rows_parts.append(rows[keep])
        cols_parts.append(cols[keep])
        vals_parts.append(vals[keep])

    inv2dr = 1.0 / (2.0 * drs)
    inv2dth = 1.0 / (2.0 * dth)
    idrs2 = 1.0 / drs**2
    idth2 = 1.0 / dth**2
    step_r = n            # index(., i+1, j) - index(., i, j)
    step_th = n * nr      # index(., i, j+1) - index(., i, j)

    # ---- component couplings: Q_0 on-node, Q_r* d/dr*, Q_theta d/dtheta ----
    for l in range(n):
        rows_l = base + l
        for p in range(n):
            cols_p = base + p

            c = Q_0[1:-1, l, p, 1:-1].ravel()
            if np.any(c):
                append_triplet_batch(rows_l, cols_p, c)

            av = Q_rstar[1:-1, l, p, 1:-1].ravel()
            if np.any(av):
                append_triplet_batch(rows_l, cols_p + step_r, av * inv2dr)
                append_triplet_batch(rows_l, cols_p - step_r, -av * inv2dr)

            bv = Q_theta[1:-1, l, p, 1:-1].ravel()
            if np.any(bv):
                append_triplet_batch(rows_l, cols_p + step_th, bv * inv2dth)
                append_triplet_batch(rows_l, cols_p - step_th, -bv * inv2dth)

    # ---- second derivatives, diagonal in the field index -------------------
    # The polar term carries Delta / (r^2 + a^2)^2; the radial one is bare.
    polar_coeff = np.zeros((nth, nr))
    polar_coeff[1:-1, 1:-1] = (
        kerr_delta(a, grid.x) / (grid.r**2 + a * a) ** 2
    )[None, 1:-1]
    c_th = polar_coeff[1:-1, 1:-1].ravel().astype(complex)

    interior = np.zeros((nth, nr))
    interior[1:-1, 1:-1] = 1.0
    c_rs = interior[1:-1, 1:-1].ravel().astype(complex)

    for l in range(n):
        rows_l = cols_l = base + l
        append_triplet_batch(rows_l, cols_l, -2.0 * c_th * idth2)
        append_triplet_batch(rows_l, cols_l + step_th, c_th * idth2)
        append_triplet_batch(rows_l, cols_l - step_th, c_th * idth2)
        append_triplet_batch(rows_l, cols_l, -2.0 * c_rs * idrs2)
        append_triplet_batch(rows_l, cols_l + step_r, c_rs * idrs2)
        append_triplet_batch(rows_l, cols_l - step_r, c_rs * idrs2)

    # ---- polar conditions at both poles, every i ---------------------------
    dirichlet = polar_condition_classes(m)
    neumann = 1.0 - dirichlet
    all_i = np.arange(nr)
    def constant_over_radius(value: complex) -> np.ndarray:
        return np.full(nr, value, dtype=complex)

    for l in range(n):
        # theta = 0: forward one-sided derivative, weights (-3, 4, -1) / 2 dtheta,
        # switched off by `neumann[l]` when this field takes Dirichlet instead.
        at_pole = l + n * all_i
        append_triplet_batch(
            at_pole,
            at_pole,
            constant_over_radius(-3.0 * inv2dth * neumann[l] + dirichlet[l]),
        )
        append_triplet_batch(
            at_pole, at_pole + step_th, constant_over_radius(4.0 * inv2dth * neumann[l])
        )
        append_triplet_batch(
            at_pole, at_pole + 2 * step_th, constant_over_radius(-1.0 * inv2dth * neumann[l])
        )

        # theta = pi: the mirror image, reaching the other way.
        at_other_pole = l + n * all_i + step_th * (nth - 1)
        append_triplet_batch(
            at_other_pole,
            at_other_pole,
            constant_over_radius(3.0 * inv2dth * neumann[l] + dirichlet[l]),
        )
        append_triplet_batch(
            at_other_pole,
            at_other_pole - step_th,
            constant_over_radius(-4.0 * inv2dth * neumann[l]),
        )
        append_triplet_batch(
            at_other_pole,
            at_other_pole - 2 * step_th,
            constant_over_radius(1.0 * inv2dth * neumann[l]),
        )

    # ---- radial conditions, interior theta only ----------------------------
    jb = np.arange(1, nth - 1)
    thb = grid.theta[jb]

    def append_boundary_condition_stencil(entries, edge: int) -> None:
        """Place one boundary-condition stencil, anchored at radial index `edge`."""
        for row_field, row_off, col_field, col_off, col_dj, value in entries:
            i_row, i_col = edge + row_off, edge + col_off
            if not (0 <= i_row < nr and 0 <= i_col < nr):
                raise IndexError(
                    f"a boundary condition reaches radial index {i_col}, outside a "
                    f"grid of {nr} nodes"
                )
            jcol = jb + col_dj
            if jcol.min() < 0 or jcol.max() >= nth:
                raise IndexError("a boundary condition reaches outside the polar grid")
            if row_field is None:
                for l in range(n):
                    append_triplet_batch(
                        l + n * (i_row + nr * jb),
                        l + n * (i_col + nr * jcol),
                        np.asarray(value[l], dtype=complex),
                    )
            else:
                append_triplet_batch(
                    row_field + n * (i_row + nr * jb),
                    col_field + n * (i_col + nr * jcol),
                    np.asarray(value, dtype=complex),
                )

    append_boundary_condition_stencil(
        near_horizon_boundary_condition(m, a, omega, grid.r[0], thb, drs, dth), edge=0
    )
    outer = outer_boundary_condition_m0 if m == 0 else outer_boundary_condition
    append_boundary_condition_stencil(
        outer(m, a, omega, grid.r[-1], thb, drs, dth), edge=nr - 1
    )

    t_triplets = time.perf_counter()

    # Concatenate into preallocated buffers, releasing each part as it is
    # copied. np.concatenate would hold the parts and the result at once,
    # roughly doubling peak memory at exactly the resolutions where memory is
    # the binding constraint. Narrowing the index dtype saves another third.
    idx_dtype = np.int32 if N < np.iinfo(np.int32).max else np.int64
    total = sum(part.size for part in vals_parts)
    rows = np.empty(total, dtype=idx_dtype)
    cols = np.empty(total, dtype=idx_dtype)
    vals = np.empty(total, dtype=complex)
    at = 0
    while rows_parts:
        r_part, c_part, v_part = rows_parts.pop(), cols_parts.pop(), vals_parts.pop()
        k = v_part.size
        rows[at : at + k] = r_part
        cols[at : at + k] = c_part
        vals[at : at + k] = v_part
        at += k
        del r_part, c_part, v_part

    M = sp.coo_matrix((vals, (rows, cols)), shape=(N, N))
    del rows, cols, vals
    M = M.tocsr()
    M.sum_duplicates()
    t_build = time.perf_counter()

    return AssembledSystem(
        M=M,
        m=m,
        orbit=orbit,
        grid=grid,
        timings={
            "coefficients": t_coef - t0,
            "triplets": t_triplets - t_coef,
            "sparse_build": t_build - t_triplets,
            "total": t_build - t0,
        },
    )


# ======================================================================
# The reference assembly (for tests only)
# ======================================================================

def assemble_reference(m: int, orbit: Orbit, grid: Grid) -> sp.csr_matrix:
    """Assemble M entry by entry, transcribed directly from the finite-difference equations. Far too slow
    for real grids; on a small grid it is an independent implementation to check `assemble` against
    (tests/test_assemble.py). Keep it dumb: no vectorization, no helpers shared with `assemble`."""
    n = N_COMPONENTS
    nr, nth = grid.n_rstar, grid.n_theta
    size = n * nr * nth
    drs, dth = grid.delta_rstar, grid.delta_theta
    a = orbit.a
    omega = orbit.mode_frequency(m)

    def idx(l, i, j):
        return l + n * (i + nr * j)

    # Coefficients node by node, left at zero on every boundary.
    Q_rstar = np.zeros((nth, n, n, nr), dtype=complex)
    Q_theta = np.zeros((nth, n, n, nr), dtype=complex)
    Q_0 = np.zeros((nth, n, n, nr), dtype=complex)
    for j in range(1, nth - 1):
        for i in range(1, nr - 1):
            rv = np.array([grid.r[i]])
            Q_rstar[j, :, :, i] = first_rstar_derivative_coefficients(m, a, omega, rv, grid.theta[j])[:, :, 0]
            Q_theta[j, :, :, i] = first_theta_derivative_coefficients(m, a, omega, rv, grid.theta[j])[:, :, 0]
            Q_0[j, :, :, i] = non_derivative_coefficients(m, a, omega, rv, grid.theta[j])[:, :, 0]

    polar = kerr_delta(a, grid.x) / (grid.r**2 + a * a) ** 2

    mat = sp.lil_matrix((size, size), dtype=complex)

    # Interior equations.
    for j in range(1, nth - 1):
        for i in range(1, nr - 1):
            for l in range(n):
                row = idx(l, i, j)
                # d2/dr*^2
                mat[row, idx(l, i, j)] += -2.0 / drs**2
                mat[row, idx(l, i + 1, j)] += 1.0 / drs**2
                mat[row, idx(l, i - 1, j)] += 1.0 / drs**2
                # (Delta / (r^2 + a^2)^2) d2/dtheta^2
                c = polar[i]
                mat[row, idx(l, i, j)] += -2.0 * c / dth**2
                mat[row, idx(l, i, j + 1)] += c / dth**2
                mat[row, idx(l, i, j - 1)] += c / dth**2
                for p in range(n):
                    # Q_r* d/dr*
                    mat[row, idx(p, i + 1, j)] += Q_rstar[j, l, p, i] / (2.0 * drs)
                    mat[row, idx(p, i - 1, j)] += -Q_rstar[j, l, p, i] / (2.0 * drs)
                    # Q_theta d/dtheta
                    mat[row, idx(p, i, j + 1)] += Q_theta[j, l, p, i] / (2.0 * dth)
                    mat[row, idx(p, i, j - 1)] += -Q_theta[j, l, p, i] / (2.0 * dth)
                    # Q_0
                    mat[row, idx(p, i, j)] += Q_0[j, l, p, i]

    # Polar conditions, every i including the corners.
    dirichlet = polar_condition_classes(m)
    neumann = 1.0 - dirichlet
    for i in range(nr):
        for l in range(n):
            row = idx(l, i, 0)
            mat[row, idx(l, i, 0)] += -3.0 / (2.0 * dth) * neumann[l] + dirichlet[l]
            mat[row, idx(l, i, 1)] += 4.0 / (2.0 * dth) * neumann[l]
            mat[row, idx(l, i, 2)] += -1.0 / (2.0 * dth) * neumann[l]

            row = idx(l, i, nth - 1)
            mat[row, idx(l, i, nth - 1)] += 3.0 / (2.0 * dth) * neumann[l] + dirichlet[l]
            mat[row, idx(l, i, nth - 2)] += -4.0 / (2.0 * dth) * neumann[l]
            mat[row, idx(l, i, nth - 3)] += 1.0 / (2.0 * dth) * neumann[l]

    # Radial conditions, interior theta only, placed one entry at a time.
    jb = np.arange(1, nth - 1)
    thb = grid.theta[jb]
    blocks = [
        (near_horizon_boundary_condition(m, a, omega, grid.r[0], thb, drs, dth), 0),
        (
            (outer_boundary_condition_m0 if m == 0 else outer_boundary_condition)(
                m, a, omega, grid.r[-1], thb, drs, dth
            ),
            nr - 1,
        ),
    ]
    for entries, edge in blocks:
        for row_field, row_off, col_field, col_off, col_dj, value in entries:
            i_row, i_col = edge + row_off, edge + col_off
            for k, j in enumerate(jb):
                if row_field is None:
                    for l in range(n):
                        mat[idx(l, i_row, j), idx(l, i_col, j + col_dj)] += value[l][k]
                else:
                    mat[
                        idx(row_field, i_row, j), idx(col_field, i_col, j + col_dj)
                    ] += value[k]

    return mat.tocsr()
