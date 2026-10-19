"""
A deterministic stand-in for the compiled effective-source library.

The point evaluations are the only part of the package that needs a C library,
so a fake pair of them lets the whole assembly path -- the puncture field, the
effective source, the jump terms, the solve -- be tested on a fresh checkout
with nothing built. The values are smooth, bounded and nonzero everywhere;
they are not physics, and nothing here checks physics.
"""

from __future__ import annotations

import numpy as np

from ellipticgsf_neural.problem import Orbit
from ellipticgsf_neural.source import N_COMPONENTS, PunctureField


def stub_h_s(m: int, r: float, theta: float, n: int = N_COMPONENTS) -> np.ndarray:
    """Stand-in for the raw puncture field: 2n floats, real parts then imaginary."""
    k = np.arange(1, 2 * n + 1, dtype=float)
    return np.sin((k + 2 * m) / 8 + r / 16 + theta / 4) / (1 + k / 8)


def stub_s_eff(m: int, r: float, theta: float, n: int = N_COMPONENTS) -> np.ndarray:
    """Stand-in for the raw effective source. Deliberately unlike `stub_h_s`."""
    k = np.arange(1, 2 * n + 1, dtype=float)
    return np.cos((k + 3 * m) / 8 + r / 32 + theta / 2) / (2 + k / 4)


def stub_puncture_field(orbit: Orbit) -> PunctureField:
    """A `PunctureField` wired to the two stubs."""
    return PunctureField(orbit=orbit, h_s=stub_h_s, s_eff=stub_s_eff)
