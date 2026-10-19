"""
The neural preconditioner's plumbing (neural.py): the untrained network gives ILU(0), the Crout and
Doolittle forms agree, the GMRES-cycle loss gradient matches finite differences, and solve() takes any
network. On the small problem of common.py; skipped without PyTorch.
"""

from __future__ import annotations

import numpy as np

from ellipticgsf_neural.iterative import SchurComplementPreconditioner
from ellipticgsf_neural.solve import solve

from .common import GMRES, small_problem


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def test_the_untrained_network_reproduces_textbook_ilu0():
    """The last layer of Psi^(K-1) starts at zero: E^(K) = 0 and P_neural is textbook ILU(0) (Sec. IV B)."""
    if not _torch():
        return
    import torch
    from ellipticgsf_neural import neural
    p = small_problem()
    P = SchurComplementPreconditioner(p, "ilu0")
    inputs = neural.gnn_inputs(P, p)
    with torch.no_grad():
        correction = neural.MessagePassingGNN()(*neural.MessagePassingGNN.tensors(inputs)).numpy()
    assert np.all(correction == 0)
    values = neural.neural_factor_values(P, inputs, correction)
    assert np.allclose(values, P.ilu.values, rtol=1e-12, atol=0)


def test_the_network_has_the_papers_size():
    """K = 4 rounds, 7 networks (4 Psi, 3 Phi), 86,978 parameters (Sec. IV B)."""
    if not _torch():
        return
    from ellipticgsf_neural import neural
    net = neural.MessagePassingGNN()
    assert net.K == 4 and len(net.psi) == 4 and len(net.phi) == 3
    assert sum(q.numel() for q in net.parameters()) == 86_978


def test_crout_and_doolittle_forms_are_the_same_factorization():
    """Crout: L0 carries the pivots, U0 is unit; Doolittle: the reverse. Both give the same product."""
    if not _torch():
        return
    from ellipticgsf_neural import neural
    p = small_problem()
    F = SchurComplementPreconditioner(p, "ilu0").ilu
    crout = neural.doolittle_to_crout(F.indptr, F.indices, F.diagonal, F.values)
    assert np.allclose(neural.crout_to_doolittle(F.indptr, F.indices, F.diagonal, crout), F.values, rtol=1e-12)
    import scipy.sparse as sp
    rows = np.repeat(np.arange(F.n), np.diff(F.indptr))
    low, diag, up = F.indices < rows, F.indices == rows, F.indices > rows
    mat = lambda vals, keep: sp.csr_matrix((vals[keep], (rows[keep], F.indices[keep])), shape=(F.n, F.n))
    doolittle = (mat(F.values, low) + sp.identity(F.n)) @ (mat(F.values, diag | up))
    crout_product = mat(crout, low | diag) @ (mat(crout, up) + sp.identity(F.n))
    assert abs(doolittle - crout_product).max() <= 1e-10 * abs(doolittle).max()


def test_the_cycle_loss_gradient_matches_a_finite_difference():
    if not _torch():
        return
    from ellipticgsf_neural import neural
    p = small_problem()
    P = SchurComplementPreconditioner(p, "ilu0")
    inputs = neural.gnn_inputs(P, p)
    loss = neural.GMRESCycleLoss(p, P, inputs)
    rng = np.random.default_rng(4)
    correction = (1e-3 * rng.standard_normal((inputs["row_i"].size, 2))).astype(np.float32)
    loss.set_correction(correction)
    gr = np.zeros(correction.shape[0], np.float32)
    gi = np.zeros_like(gr)
    f0 = loss.loss_and_gradient(p.b, 8, None, gr, gi, 1.0)
    direction = rng.standard_normal(correction.shape).astype(np.float32)
    h = 1e-4
    fs = []
    for sign in (1, -1):
        loss.set_correction(correction + sign * h * direction)
        fs.append(loss.loss_and_gradient(p.b, 8, None, np.zeros_like(gr), np.zeros_like(gi), 1.0))
    fd = (fs[0] - fs[1]) / (2 * h)
    an = float(np.sum(gr * direction[:, 0]) + np.sum(gi * direction[:, 1]))
    assert np.isfinite(f0) and abs(fd - an) <= 1e-3 * max(abs(fd), 1e-8) + 1e-6


def test_solve_takes_another_network_for_the_neural_preconditioner():
    """A network passed to solve() is used in place of the shipped one: here the untrained network, which
    gives textbook ILU(0), on a mode with no shipped network (as when m = 8 uses the m = 7 network)."""
    if not _torch():
        return
    from ellipticgsf_neural import neural
    p = small_problem(m=9)
    s = solve(p, "neural", GMRES, network=neural.GNNInference(neural.MessagePassingGNN(), compile=False))
    assert s.converged and s.iterations == solve(p, "ilu0", GMRES).iterations


def test_the_shipped_networks_load():
    if not _torch():
        return
    from ellipticgsf_neural import neural
    for m in range(1, 8):
        net = neural.load_network(m)
        assert net.K == 4 and sum(q.numel() for q in net.parameters()) == 86_978
