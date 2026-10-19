"""
The neural preconditioner (Sec. IV): a graph neural network (GNN) that reads the Schur complement M_S
(App. A; "M" in Sec. IV) as the weighted adjacency matrix of a graph and corrects its textbook ILU(0)
factorization, P_neural = (L0 + dL)(U0 + dU); and everything needed to train it.

    network = load_network(m)                              # the trained GNN shipped in models/gnn_m<m>.pt
    P = neural_preconditioner(problem, network)            # SchurComplementPreconditioner with P_neural
    train(m, init="models/gnn_m7.pt", out_dir="runs/m6", grid_parameters=..., gmres=..., training=...)

THE GRAPH (Sec. IV A)
---------------------
One node per bulk unknown, one edge per nonzero (M_S)_ij, self-loops on the diagonal. Inputs
(`gnn_inputs`):

    E^(0)_ij = (M_S)_ij / |(M_S)_ii|                                  as (Re, Im): 2 numbers per edge
    v^(0)_i  = (m/10, (M Omega - 0.05)/0.02, r_i/50 - 1, 2 theta_i/pi - 1, c^tensor_i, f^bound_r_i,
                f^bound_theta_i)                                       16 numbers per node

c^tensor_i is the one-hot tensor component (10 numbers) and f^bound are three-state flags: -1 / 0 / +1
for the inner / neither / outer bulk radius (next to the strips x_A and x_C), and for theta = 0 /
neither / pi. The orbit's frequency enters as M Omega = M omega / m (the network also sees m), and every
input is normalized to order one; the order of the entries is fixed by the trained networks.

MESSAGE PASSING (Sec. IV A)
---------------------------
K = 4 rounds n = 0 .. K-1 of

    E^(n+1)_ij = Psi^(n)(v^(n)_i, v^(n)_j, E^(n)_ij, E^(n)_ji, sign(i - j), f^tr_ij)        every edge
    v^(n+1)_i  = Phi^(n)(v^(n)_i, (1/N_row) sum_j E^(n+1)_ij, (1/N_col) sum_j E^(n+1)_ji)    every node

(E^(n)_ji = 0 and f^tr_ij = 0 when (M_S)_ji is not in the graph; v^(K) is not computed). Every Psi^(n)
and Phi^(n) is a fully connected network with two hidden layers and SiLU activations (Sec. IV B): 32
numbers per edge and per node between rounds, hidden width 64, 86,978 parameters in all. Within a round
all edges share Psi^(n) and all nodes share Phi^(n).

THE PRECONDITIONER (Sec. IV A)
------------------------------
The output E^(K)_ij (2 numbers: Re, Im, times 1e-3) corrects ILU(0) in Crout form (U0 unit upper
triangular, the pivots on the diagonal of L0):

    P_neural = (L0 + dL)(U0 + dU),    (dL)_ii = E^(K)_ii |(M_S)_ii|,    (dL)_ij = E^(K)_ij |(L0)_jj| (i > j),
                                      (dU)_ij = E^(K)_ij (i < j)

The last layer of Psi^(K-1) starts at zero, so the untrained network gives textbook ILU(0). In the code
all three cases share one scale, s_ij = |(L0)_jj| for i >= j and 1 for i < j (`correction_scale`): the
network's output on the diagonal is multiplied by |(M_S)_ii| / |(L0)_ii| before it leaves the network
(`diagonal_rescale`), so

    Crout entries of P_neural = Crout entries of L0 U0 + s_ij * delta_ij,
    delta_ij = E^(K)_ij (i != j),   delta_ii = E^(K)_ii |(M_S)_ii| / |(L0)_ii|                (`correction`)

The triangular solves use the Doolittle storage of iterative.IncompleteLU; `crout_to_doolittle` converts.

INFERENCE (Sec. IV B)
---------------------
16-bit floats (bfloat16) where the CPU computes them natively (AVX512-BF16 or AMX; float32 elsewhere,
where bfloat16 is emulated and slower), through torch.compile with freezing, edges processed 100k at a
time (~0.6 GB instead of ~2 GB at m = 2). At m = 2 on an 8-thread AVX512-BF16 CPU: 1.2 s compiled with
freezing, 2.4 s compiled without, ~2.5 s uncompiled; the first compiled call per process adds ~7 s.
bfloat16 changes the corrections by ~1e-2 relative and the iteration counts by a few; the GMRES
arithmetic stays in 64-bit. `benchmark_inference` times every combination.

TRAINING (Sec. IV B)
--------------------
The loss is the GMRES restart-cycle loss

    L = < log( ||cycle c end residual|| / ||cycle c start residual|| ) >_{c in {1, 2, 4, 7, 11}}

over the cycles that a GMRES solve with the current network reaches. The cycles' start residuals r_c
and deflation spaces C_c come from that solve and are held fixed. Cycle c is N_c Arnoldi steps on
(I - C_c C_c^H) M P_neural^-1 from r_c, as in the solve: N_c = 50 for c = 1 (no deflation space yet) and
50 - 10 = 40 afterwards. dL/dE^(K) comes from a hand-written backward pass through the cycle (GMRESCycleLoss:
least squares -> Hessenberg -> normalization and Gram-Schmidt -> M^dagger -> adjoint preconditioner
solves), and PyTorch carries it back through the network. Adam with linear warm-up and gradient-norm
clipping; after each epoch the total GMRES iterations on held-out orbits decide whether to keep the
epoch (`train`).
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from . import TORCH_IMPORT_ERROR

if TORCH_IMPORT_ERROR is not None:
    raise ImportError(
        f"the neural preconditioner needs PyTorch, which could not be loaded: {TORCH_IMPORT_ERROR}\n"
        "Check it on its own in the same environment:  python -c \"import torch; print(torch.__version__)\"\n"
        "If that fails too, the PyTorch install is broken: reinstall it (pip install --force-reinstall torch, "
        "or the command from pytorch.org for your platform). On Windows also update the Microsoft Visual C++ "
        "Redistributable (2015-2022, x64).") from TORCH_IMPORT_ERROR

import numba as nb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .iterative import DeflatedRestartGMRES, GMRESParameters, SchurComplementPreconditioner, csr_matvec
from .problem import N_COMPONENTS, GridParameters, build_problem

__all__ = [
    "MODEL_DIR",
    "orbital_radius",
    "gnn_inputs",
    "neural_factor_values",
    "MessagePassingGNN",
    "load_network",
    "save_network",
    "GNNInference",
    "shipped_network_inference",
    "neural_preconditioner",
    "benchmark_inference",
    "GMRESCycleLoss",
    "TrainingParameters",
    "evaluate",
    "train",
]

MODEL_DIR = Path(__file__).resolve().parent / "models"


def orbital_radius(omega_over_m: float) -> float:
    """r0 / M of the circular orbit with M omega / m = M Omega (Schwarzschild: Omega = r0^(-3/2))."""
    return float(omega_over_m) ** (-2.0 / 3.0)


# ======================================================================
# The graph: inputs, and turning the output into P_neural (Sec. IV A)
# ======================================================================

def gnn_inputs(P: SchurComplementPreconditioner, problem) -> dict:
    """The GNN's inputs for the Schur complement M_S of P (built with "ilu0"), plus what is needed to turn
    its output into P_neural:

        v0, E0                  v^(0)_i (n, 16) and E^(0)_ij (edges, 2: Re, Im)
        row_i, col_j            each edge's (i, j); edges in the CSR order of M_S
        transpose_position      the position of edge (j, i) (0 if absent)
        f_tr                    1 if (j, i) is an edge, else 0
        sign_i_minus_j          sign(i - j)
        diagonal_rescale        |(M_S)_ii| / |(L0)_ii| on the diagonal, 1 elsewhere
        crout0                  the Crout entries of L0 U0 per edge: (L0)_ij for i >= j, (U0)_ij for i < j
        correction_scale        s_ij = |(L0)_jj| for i >= j, 1 for i < j
    """
    M_S, F, grid = P.M_S, P.ilu, P.grid
    if F is None or F.level != 0:
        raise ValueError("the neural preconditioner needs a SchurComplementPreconditioner built with 'ilu0'")
    if not (np.array_equal(M_S.indptr, F.indptr) and np.array_equal(M_S.indices, F.indices)):
        raise ValueError("the ILU(0) pattern differs from that of M_S")
    n = F.n
    # One edge per stored entry of M_S, in CSR order: edge q joins node i = rows[q] to node j = cols[q]. The
    # ILU(0) factors have exactly this pattern (checked above), so the same q also indexes the factor entries
    # that the network's output corrects.
    rows = np.repeat(np.arange(n), np.diff(F.indptr))
    cols = F.indices.astype(np.int64)
    abs_M_ii = np.abs(M_S.data[F.diagonal])
    abs_L0_ii = np.abs(F.values[F.diagonal])                       # |pivots| (Doolittle U = Crout L0)
    # E^(0): every row divided by the size of its diagonal entry, so the inputs are of order one whatever
    # the scale of that row of M_S (the diagonal edge becomes a unit complex number)
    E0 = M_S.data / abs_M_ii[rows]
    # transpose partners: CSR is sorted by (row, col), so a key search finds (j, i)
    key, transpose_key = rows * n + cols, cols * n + rows
    # searchsorted gives where (j, i) would sit; clip that to a valid position, then the equality says
    # whether the entry there really is (j, i). Absent partners keep position 0, masked by f_tr = 0.
    pos = np.minimum(np.searchsorted(key, transpose_key), key.size - 1)
    f_tr = key[pos] == transpose_key
    # node features v^(0): the grid position and tensor component of each bulk unknown (problem.py)
    g = P.bulk
    # invert problem.unknown_index: the global index g = l + 10 (i + n_rstar j) of each bulk unknown gives
    # its tensor component l, radial node i and polar node j
    component, ir, jt = g % N_COMPONENTS, (g // N_COMPONENTS) % grid.n_rstar, g // (N_COMPONENTS * grid.n_rstar)
    c_tensor = np.zeros((n, N_COMPONENTS))
    c_tensor[np.arange(n), component] = 1.0
    # the 16 node features, each mapped to roughly [-1, 1]; the boundary flags mark the first and last bulk
    # radius (next to the strips x_A and x_C) and the polar axis theta = 0, pi
    v0 = np.column_stack([
        np.full(n, problem.m / 10.0), np.full(n, (problem.orbit.omega - 0.05) / 0.02),
        grid.r[ir] / 50.0 - 1.0, 2.0 * grid.theta[jt] / np.pi - 1.0, c_tensor,
        np.where(ir == ir.min(), -1.0, np.where(ir == ir.max(), 1.0, 0.0)),            # f^bound_r
        np.where(jt == 0, -1.0, np.where(jt == grid.n_theta - 1, 1.0, 0.0))])          # f^bound_theta
    # The paper scales the correction by |(M_S)_ii| on the diagonal, by |(L0)_jj| below it and by 1 above it
    # (Sec. IV A). Folding the ratio |(M_S)_ii| / |(L0)_ii| into the network's output lets one per-edge scale
    # s_ij (correction_scale) serve every entry: s_ij = |(L0)_jj| on and below the diagonal, 1 above it
    # (U0 is unit, so its entries are already of order one).
    diagonal_rescale = np.ones(rows.size)
    diagonal_rescale[F.diagonal] = abs_M_ii / abs_L0_ii
    return dict(
        v0=v0.astype(np.float32),
        E0=np.stack([E0.real, E0.imag], axis=1).astype(np.float32),
        row_i=rows, col_j=cols, transpose_position=np.where(f_tr, pos, 0), f_tr=f_tr.astype(np.float32),
        sign_i_minus_j=np.sign(rows - cols).astype(np.float32), diagonal_rescale=diagonal_rescale.astype(np.float32),
        crout0=doolittle_to_crout(F.indptr, F.indices, F.diagonal, F.values),
        correction_scale=np.where(cols <= rows, abs_L0_ii[cols], 1.0))


def neural_factor_values(P: SchurComplementPreconditioner, inputs: dict, correction) -> np.ndarray:
    """P_neural's values in Doolittle storage (for P.set_bulk_factor_values), from the network's
    `correction` (edges, 2: Re, Im; see the module docstring)."""
    d = np.asarray(correction, np.float64)
    # the correction is added in Crout form, (L0 + dL)(U0 + dU), where every entry is a factor entry of the
    # paper; the triangular solves then need it back in Doolittle storage
    crout = inputs["crout0"] + inputs["correction_scale"] * (d[:, 0] + 1j * d[:, 1])
    return crout_to_doolittle(P.ilu.indptr, P.ilu.indices, P.ilu.diagonal, crout)


@nb.njit(cache=True)
def doolittle_to_crout(ptr, idx, diag, values):
    """Doolittle storage (unit L strict part, U with the pivots) -> Crout entries ((L0)_ij = l_ij d_j below
    the diagonal, the pivots d_i on it, (U0)_ij = u_ij / d_i above)."""
    # With D = diag(d), L U = (L D)(D^-1 U): Crout's L0 = L D takes the pivots as its diagonal, and
    # U0 = D^-1 U is unit. Entry (i, k) of row i:
    #   k < i:  (L0)_ik = l_ik d_k    (column k of L scaled by the pivot of that column)
    #   k = i:  d_i                   (the pivot, unchanged)
    #   k > i:  (U0)_ik = u_ik / d_i  (row i of U divided by its own pivot)
    c = np.empty(values.size, np.complex128)
    for i in range(ptr.size - 1):
        for q in range(ptr[i], ptr[i + 1]):
            k = idx[q]
            c[q] = values[q] * values[diag[k]] if k < i else (values[q] if k == i else values[q] / values[diag[i]])
    return c


@nb.njit(cache=True)
def crout_to_doolittle(ptr, idx, diag, c):
    """The inverse of doolittle_to_crout."""
    # the same three cases undone: the pivots d are the diagonal Crout entries c[diag[.]], which both forms share
    values = np.empty(c.size, np.complex128)
    for i in range(ptr.size - 1):
        for q in range(ptr[i], ptr[i + 1]):
            k = idx[q]
            values[q] = c[q] / c[diag[k]] if k < i else (c[q] if k == i else c[q] * c[diag[i]])
    return values


# ======================================================================
# The network (Secs. IV A and IV B)
# ======================================================================

class MLP(nn.Module):
    """W3 sigma(W2 sigma(W1 z + b1) + b2) + b3 with sigma = SiLU (for n_hidden = 2), the form of every Psi^(n)
    and Phi^(n) (Sec. IV B)."""

    def __init__(self, d_in, d_hidden, d_out, n_hidden=2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, d_hidden), nn.SiLU()]
            d = d_hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def edge_update(Psi, v, E, E_chunk, row_i, col_j, transpose_position, f_tr, sign_i_minus_j):
    """E^(n+1)_ij = Psi^(n)(v_i, v_j, E_ij, E_ji, sign(i - j), f^tr_ij) on one chunk of edges (the per-edge
    tensors already sliced to the chunk; E: all edges, for E_ji)."""
    # Gather the features of each edge's two end nodes, and the transpose edge's features: E[transpose_position]
    # reads E_ji (from anywhere in the graph, which is why the whole E is passed), and the factor f_tr zeroes
    # it where (j, i) is not an edge (position 0 is only a placeholder there). sign(i - j) tells Psi whether
    # the edge feeds the lower factor, the diagonal, or the upper factor.
    return Psi(torch.cat([v[row_i], v[col_j], E_chunk, E[transpose_position] * f_tr, sign_i_minus_j, f_tr], dim=1))


def node_update(Phi, v, E, row_i, col_j, inverse_N_row, inverse_N_col):
    """v^(n+1)_i = Phi^(n)(v_i, (1/N_row) sum_j E_ij, (1/N_col) sum_j E_ji) on every node (mean aggregation)."""
    n = v.shape[0]
    # Scatter-add each edge's features to a node: by row_i, node i receives sum_j E_ij (the edges leaving it,
    # row i of M_S); by col_j, node i receives sum_j E_ji (the edges arriving, column i). Dividing by the
    # number of entries turns the sums into means, so nodes with fewer neighbours (at the boundaries of the
    # grid) see inputs of the same size as the rest.
    mean_row = torch.zeros(n, E.shape[1], dtype=E.dtype).index_add_(0, row_i, E) * inverse_N_row
    mean_col = torch.zeros(n, E.shape[1], dtype=E.dtype).index_add_(0, col_j, E) * inverse_N_col
    return Phi(torch.cat([v, mean_row, mean_col], dim=1))


def _padded(args, size):
    """Edge-update arguments (n, v, E, then per-edge tensors) with the per-edge tensors padded to `size` rows
    by repeating row 0, so every chunk has the same shape."""
    n, v, E, *per_edge = args
    # Row 0 is a valid edge, so the padding rows compute harmless (discarded) values; v and E, which are read
    # by index, are left as they are.
    pad = size - per_edge[0].shape[0]
    return (n, v, E, *(torch.cat([t, t[:1].expand(pad, *t.shape[1:])]) for t in per_edge))


class MessagePassingGNN(nn.Module):
    """
    The GNN of Sec. IV A: K rounds of edge updates Psi^(n) (self.psi[n]) and node updates Phi^(n)
    (self.phi[n], none after the last round). forward() returns the per-edge `correction` (edges, 2)
    that neural_factor_values turns into P_neural.

    K            message-passing rounds (4)
    node_in      size of v^(0) (16);  edge_in: size of E^(0) (2: Re, Im)
    d_node       size of v^(n) for 0 < n < K (32);  d_edge: size of E^(n) for 0 < n < K (32)
    d_hidden     width of the two hidden layers of every Psi and Phi (64)
    out_scale    E^(K) = out_scale x the output of Psi^(K-1) (1e-3), keeping the trained outputs of order one
    chunk        edges per chunk in training (each chunk checkpointed, so the activations are never all
                 held at once)
    """

    INPUTS = ("v0", "E0", "row_i", "col_j", "transpose_position", "f_tr", "sign_i_minus_j", "diagonal_rescale")
    INFERENCE_CHUNK = 100_000           # edges per chunk in inference
    pad_chunks = False                  # pad the last chunk to the full size (compiled inference: one shape)

    def __init__(self, K=4, d_node=32, d_edge=32, d_hidden=64, n_hidden=2, node_in=16, edge_in=2,
                 out_scale=1e-3, chunk=300_000):
        super().__init__()
        self.K, self.out_scale, self.chunk = K, out_scale, chunk
        self.psi, self.phi = nn.ModuleList(), nn.ModuleList()
        for n in range(K):
            # round 0 reads the raw inputs (16 per node, 2 per edge); later rounds read the 32 + 32 hidden
            # features; the last round's Psi outputs the 2 numbers (Re, Im) of the correction
            vin = node_in if n == 0 else d_node
            ein = edge_in if n == 0 else d_edge
            eout = 2 if n == K - 1 else d_edge
            # Psi^(n) reads v_i, v_j (2 vin), E_ij, E_ji (2 ein), sign(i - j) and f^tr_ij (2)
            self.psi.append(MLP(2 * vin + 2 * ein + 2, d_hidden, eout, n_hidden))
            # Phi^(n) reads v_i and the row and column means of the new edge features; there is no Phi after
            # the last round, because only E^(K) is used
            if n < K - 1:
                self.phi.append(MLP(vin + 2 * eout, d_hidden, d_node, n_hidden))
        nn.init.zeros_(self.psi[-1].net[-1].weight)          # untrained: E^(K) = 0, P_neural = ILU(0)
        nn.init.zeros_(self.psi[-1].net[-1].bias)
        self.config = dict(K=K, d_node=d_node, d_edge=d_edge, d_hidden=d_hidden, n_hidden=n_hidden,
                           node_in=node_in, edge_in=edge_in, out_scale=out_scale, chunk=chunk)

    def _edges(self, n, v, E, E_chunk, row_i, col_j, transpose_position, f_tr, sign_i_minus_j):
        return edge_update(self.psi[n], v, E, E_chunk, row_i, col_j, transpose_position, f_tr, sign_i_minus_j)

    def _nodes(self, n, v, E, row_i, col_j, inverse_N_row, inverse_N_col):
        return node_update(self.phi[n], v, E, row_i, col_j, inverse_N_row, inverse_N_col)

    def forward(self, v0, E0, row_i, col_j, transpose_position, f_tr, sign_i_minus_j, diagonal_rescale):
        v, E = v0, E0
        n_nodes, n_edges = v.shape[0], E.shape[0]
        # 1/N_row and 1/N_col: the number of stored entries in each row and column of M_S (a self-loop always
        # exists, so clamp_min(1) only guards against degenerate inputs); fixed for all K rounds
        inverse_N_row = (1.0 / torch.bincount(row_i, minlength=n_nodes).clamp_min(1).to(v.dtype))[:, None]
        inverse_N_col = (1.0 / torch.bincount(col_j, minlength=n_nodes).clamp_min(1).to(v.dtype))[:, None]
        # "training" means the gradient is being recorded (train), not nn.Module.training: evaluate() runs a
        # network in train mode under no_grad, and it should then take the inference path
        training = torch.is_grad_enabled()
        chunk = self.chunk if training else self.INFERENCE_CHUNK
        for n in range(self.K):
            # edges in chunks: in training each chunk is checkpointed (recomputed in the backward pass); in
            # inference, chunks that fit in cache are faster and need a fraction of the memory
            parts = []
            for lo in range(0, n_edges, chunk):
                c = slice(lo, min(lo + chunk, n_edges))
                args = (n, v, E, E[c], row_i[c], col_j[c], transpose_position[c], f_tr[c], sign_i_minus_j[c])
                if training:
                    parts.append(checkpoint(self._edges, *args, use_reentrant=False))
                elif self.pad_chunks and c.stop - c.start < chunk:
                    parts.append(self._edges(*_padded(args, chunk))[:c.stop - c.start])
                else:
                    parts.append(self._edges(*args))
            # every chunk reads E^(n) (for E_ji), so E^(n+1) replaces it only once all the chunks are done
            E = torch.cat(parts, dim=0)                                      # E^(n+1)
            # the node update needs every edge of the round, so it runs on the whole graph at once
            if n < self.K - 1:
                args = (n, v, E, row_i, col_j, inverse_N_row, inverse_N_col)
                v = checkpoint(self._nodes, *args, use_reentrant=False) if training else self._nodes(*args)
        # E^(K) = out_scale x the last Psi's output; diagonal_rescale converts its diagonal to the shared scale
        # s_ij (see gnn_inputs)
        return self.out_scale * E * diagonal_rescale                         # the correction, (edges, 2)

    @staticmethod
    def tensors(inputs, dtype=torch.float32):
        """The forward() arguments as tensors, from gnn_inputs' dict."""
        out = []
        # Per-edge scalars (f_tr, sign, diagonal_rescale) become columns (edges, 1), so that they concatenate
        # with the edge features and broadcast against the (edges, 2) output; the index arrays (row_i, col_j,
        # transpose_position) stay 1-D integers. Floating inputs take the inference precision.
        for name in MessagePassingGNN.INPUTS:
            t = torch.from_numpy(np.ascontiguousarray(inputs[name]))
            out.append(t.to(dtype)[:, None] if t.ndim == 1 and t.is_floating_point() else
                       t.to(dtype) if t.is_floating_point() else t)
        return out


def save_network(network, path, **extra):
    torch.save(dict(state=network.state_dict(), config=network.config, **extra), path)


def load_network(m_or_path) -> MessagePassingGNN:
    """The shipped network for mode m (an int), or a checkpoint path."""
    path = MODEL_DIR / f"gnn_m{m_or_path}.pt" if isinstance(m_or_path, int) else Path(m_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"no network {path} (shipped: {sorted(p.name for p in MODEL_DIR.glob('*.pt'))})")
    # A checkpoint holds the weights, the constructor arguments (config) and extras such as the held-out
    # scores; weights_only=False because the extras are plain Python objects, not tensors. The network is
    # rebuilt from its config, so a checkpoint carries its own sizes.
    ck = torch.load(path, weights_only=False, map_location="cpu")
    network = MessagePassingGNN(**ck["config"])
    network.load_state_dict(ck["state"])
    return network.eval()


# ======================================================================
# Inference (Sec. IV B)
# ======================================================================

def compiler_problem():
    """Why torch.compile cannot work here (a string), or None if a usable C++ compiler is available.

    Windows: torch.compile builds with MSVC's cl.exe, which must be the 64-bit (x64) compiler -- open an
    "x64 Native Tools Command Prompt for VS" (the plain "Developer PowerShell" is 32-bit). With conda-forge's
    PyTorch, torch.lib sits in <env>/Library/lib, which the linker is pointed to here."""
    if sys.platform != "win32":
        names = [os.environ.get("CXX", "g++"), "c++", "clang++"]
        return None if any(n and shutil.which(n) for n in names) else "no C++ compiler (g++) on the PATH"
    if not shutil.which("cl"):
        return ("MSVC's cl.exe is not on the PATH (install Visual Studio Build Tools with 'Desktop development "
                "with C++' and run from an 'x64 Native Tools Command Prompt for VS')")
    # cl.exe run without arguments prints a banner naming its target ("... for x64" or "... for x86")
    out = subprocess.run(["cl"], capture_output=True, text=True)
    banner = out.stderr + out.stdout
    if "x64" not in banner:
        return ("the cl.exe on the PATH is the 32-bit (x86) compiler; run from an "
                "'x64 Native Tools Command Prompt for VS' instead")
    # the generated C++ links against torch.lib; MSVC's linker finds it only through the LIB variable
    for lib in (Path(sys.prefix) / "Library" / "lib", Path(torch.__file__).parent / "lib"):
        if (lib / "torch.lib").is_file() and str(lib) not in os.environ.get("LIB", ""):
            os.environ["LIB"] = str(lib) + os.pathsep + os.environ.get("LIB", "")
    return None


def default_precision():
    """bfloat16 where the CPU computes it natively (AVX512-BF16 or AMX), float32 elsewhere: without
    hardware support bfloat16 is emulated and slower than float32."""
    try:
        native = torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported()
    except Exception:                                     # private API: assume support if it moved
        native = True
    return torch.bfloat16 if native else torch.float32


class GNNInference:
    """A network ready for inference: call it on gnn_inputs(...) to get the correction. precision: a torch
    dtype, or None for default_precision(). compile: the per-chunk edge and node updates through
    torch.compile with freezing (falls back to uncompiled if compilation is impossible or fails)."""

    _warned = False

    def __init__(self, network, precision=None, compile=True):
        self.dtype = precision = precision or default_precision()
        # a private copy: converting to bfloat16 and compiling must not touch the caller's (float32) network,
        # which training keeps updating
        self.module = copy.deepcopy(network).to(precision).eval()
        self.compiled = False
        self.compiling = False          # whether the last call compiled (first call for its graph size)
        self._sizes = set()
        problem = compiler_problem() if compile else None
        if problem:
            if not GNNInference._warned:
                print(f"note: not compiling the network with torch.compile: {problem}. It runs uncompiled "
                      "(about 10% slower; results unchanged).", flush=True)
                GNNInference._warned = True
            compile = False
        if compile:
            try:
                # Static shapes (dynamic=False): every edge chunk is padded to the same size, so each round n
                # compiles once per problem size (4 edge + 3 node graphs). Dynamic shapes would serve every
                # size with one graph, but recompiling them (one per n) hits a PyTorch guard bug in some
                # versions ("sources must not be empty for symbol ..."); static shapes never create symbols.
                for key in ("recompile_limit", "cache_size_limit"):
                    if hasattr(torch._dynamo.config, key):
                        setattr(torch._dynamo.config, key, max(getattr(torch._dynamo.config, key), 64))
                # freezing: the weights become constants that are pre-packed for oneDNN, so the matrix
                # products use the CPU's native bfloat16 instructions (without it, compiled bfloat16 was
                # no faster than float32: 2.4 s vs 1.2 s with freezing on an AVX512-BF16 CPU)
                options = {"freezing": True}
                # Only the per-chunk updates are compiled: the chunk loop in forward() stays Python, so every
                # round reuses one compiled graph. Assigning to the instance shadows the class's methods
                # (_eager undoes it). torch.compile is lazy: the compilation itself happens at the first call.
                self.module._edges = torch.compile(self.module._edges, dynamic=False, options=options)
                self.module._nodes = torch.compile(self.module._nodes, dynamic=False, options=options)
                self.module.pad_chunks = True
                self.compiled = True
            except Exception as exc:                      # torch.compile unavailable here
                warnings.warn(f"torch.compile unavailable ({exc}); running eagerly")

    def _eager(self):
        # drop the compiled instance attributes, so the plain methods of MessagePassingGNN show through again
        for name in ("_edges", "_nodes"):
            self.module.__dict__.pop(name, None)
        self.module.pad_chunks = False
        self.compiled = False

    def __call__(self, inputs) -> np.ndarray:
        x = MessagePassingGNN.tensors(inputs, self.dtype)
        size = (inputs["v0"].shape[0], inputs["row_i"].size)
        self.compiling = self.compiled and size not in self._sizes    # first call for this graph size
        self._sizes.add(size)
        with torch.no_grad():
            try:
                out = self.module(*x)
            except Exception as exc:                      # compilation failed at the first call
                if not self.compiled:
                    raise
                warnings.warn(f"torch.compile failed with PyTorch {torch.__version__} ({type(exc).__name__}: {exc}); "
                              "running uncompiled. For details set TORCHDYNAMO_VERBOSE=1")
                self._eager()
                out = self.module(*x)
        # back to float32 for numpy (which has no bfloat16); neural_factor_values continues in float64
        return out.float().numpy()


_INFERENCE: dict = {}


def shipped_network_inference(m: int, precision=None, compile=True) -> GNNInference:
    """The (cached) GNNInference of the shipped network for mode m; compilation happens at its first call."""
    key = (m, precision or default_precision(), compile)
    if key not in _INFERENCE:
        _INFERENCE[key] = GNNInference(load_network(m), precision, compile)
    return _INFERENCE[key]


def neural_preconditioner(problem, network=None, precision=None, compile=True) -> SchurComplementPreconditioner:
    """The preconditioner of App. A with P_S = P_neural. network: a MessagePassingGNN, a GNNInference, or
    None for the shipped network of problem.m. The timings gain 'network' (inputs + forward pass) and
    'network_compiled' (whether that time includes compilation)."""
    # Build the App. A preconditioner with textbook ILU(0) on M_S: this gives M_S (the graph), L0 U0 (what
    # the network corrects) and the sparsity pattern; the network's factor values then replace L0 U0's.
    P = SchurComplementPreconditioner(problem, "ilu0")
    t = time.perf_counter()
    if network is None:
        network = shipped_network_inference(problem.m, precision, compile)
    elif isinstance(network, MessagePassingGNN):
        network = GNNInference(network, precision, compile)
    inputs = gnn_inputs(P, problem)
    P.set_bulk_factor_values(neural_factor_values(P, inputs, network(inputs)))
    P.timings["network"] = time.perf_counter() - t
    P.timings["network_compiled"] = getattr(network, "compiling", False)
    return P


def benchmark_inference(problem, network=None, repeats: int = 3):
    """Time the network alone on one problem, for bfloat16 / float32, compiled / uncompiled: the first call
    (which compiles) and the best of `repeats` further calls. Prints a table and returns the rows."""
    network = network or load_network(problem.m)
    P = SchurComplementPreconditioner(problem, "ilu0")
    inputs = gnn_inputs(P, problem)

    def check(f):
        try:
            return f()
        except Exception as exc:                          # private APIs: may move between versions
            return f"unknown ({type(exc).__name__})"
    from torch._inductor import cpu_vec_isa
    print(f"PyTorch {torch.__version__}, CPU capability {torch.backends.cpu.get_cpu_capability()}, "
          f"{torch.get_num_threads()} threads; m = {problem.m}, r0 = {problem.orbit.r0:g}: "
          f"{inputs['row_i'].size:,} edges\n"
          f"  native bfloat16: AVX512-BF16 {check(torch.cpu._is_avx512_bf16_supported)}, "
          f"AMX {check(torch.cpu._is_amx_tile_supported)}\n"
          f"  torch.compile: C++ compiler {compiler_problem() or 'ok'}; vector ISA for generated code: "
          f"{check(lambda: str(cpu_vec_isa.pick_vec_isa()) or 'NONE (scalar code)')}; default precision "
          f"{str(default_precision()).split('.')[-1]}")
    rows = []
    for precision in (torch.bfloat16, torch.float32):
        for compile in (False, True):
            inference = GNNInference(network, precision, compile)
            # the first call includes compilation (when compiling); the best of the next calls is the cost
            # of every later preconditioner build in the same process
            t = time.perf_counter()
            inference(inputs)
            first = time.perf_counter() - t
            times = []
            for _ in range(repeats):
                t = time.perf_counter()
                inference(inputs)
                times.append(time.perf_counter() - t)
            rows.append(dict(precision=str(precision).split(".")[-1], compiled=inference.compiled,
                             first_s=first, steady_s=min(times)))
            print(f"  {rows[-1]['precision']:9s} {'compiled' if inference.compiled else 'uncompiled':10s} "
                  f"first call {first:6.1f} s, then {min(times):5.2f} s", flush=True)
    return rows


# ======================================================================
# The GMRES restart-cycle loss and its gradient (Sec. IV B)
#
# One cycle from the residual r with the deflation rows C held fixed: N Arnoldi steps on
# (I - C C^H) M P_neural^-1, then the least-squares residual rho = min_y || beta e1 - H y ||, and the loss
# log(rho / beta). The backward pass runs through the least squares, the normalizations and Gram-Schmidt
# steps of Arnoldi (in reverse), M^dagger and the adjoint preconditioner solves; at each Arnoldi step the
# sensitivity to the factor entries is an outer product of a forward and an adjoint vector on the
# sparsity pattern of M_S. Double precision throughout.
# ======================================================================

@nb.njit(cache=True)
def _doolittle_values_from_correction(ptr, idx, diag, crout0, abs_L0_ii, dr, di, out):
    """Doolittle values of P_neural, crout0 + s (dr + i di) with s = |(L0)_kk| (k <= i) or 1, into out."""
    # neural_factor_values and crout_to_doolittle fused into one pass, in double precision: each corrected
    # Crout entry cq is converted to Doolittle as soon as it is formed, which needs the corrected pivots
    # d_i (for the U entries of row i) and d_k (for the L entries of column k), recomputed where needed
    for i in range(ptr.size - 1):
        qi = diag[i]
        dii = crout0[qi] + abs_L0_ii[i] * (np.float64(dr[qi]) + 1j * np.float64(di[qi]))     # pivot d_i
        for q in range(ptr[i], ptr[i + 1]):
            k = idx[q]
            sq = abs_L0_ii[k] if k <= i else 1.0
            cq = crout0[q] + sq * (np.float64(dr[q]) + 1j * np.float64(di[q]))
            if k < i:                                  # l_ik = (L0 + dL)_ik / d_k
                qk = diag[k]
                out[q] = cq / (crout0[qk] + abs_L0_ii[k] * (np.float64(dr[qk]) + 1j * np.float64(di[qk])))
            elif k == i:                               # the pivot d_i
                out[q] = dii
            else:                                      # u_ik = (U0 + dU)_ik d_i
                out[q] = cq * dii


@nb.njit(cache=True)
def _accumulate_gradient(ptr, idx, diag, crout0, abs_L0_ii, dr, di, x_B, z, gr, gi, w2):
    """Add w2 * d(loss)/d(correction) for one Arnoldi step: -z_i conj((U x)_k) for L entries (k <= i) and
    -(L^H z)_i conj(x_k) for U entries, times the entry's scale s (entries = crout0 + s correction)."""
    # The bulk solve gives x_B = (L U)^-1 g with L = L0 + dL, U = U0 + dU (Crout), and z is the adjoint
    # solution (L U)^-H of x_B's sensitivity. Since d(x_B) = -(L U)^-1 (dL U + L dU) x_B, the loss
    # changes by -2 Re[ z^H dL (U x_B) + (L^H z)^H dU x_B ]: an outer product z (U x_B)^H restricted to the
    # pattern of L, and (L^H z) x_B^H on the pattern of U. The first loop forms both vectors, U x_B (U unit)
    # and L^H z, in one pass over the rows.
    n = ptr.size - 1
    Ux = x_B.copy()
    LHz = np.zeros(n, np.complex128)
    for i in range(n):
        acc = Ux[i]
        zi = z[i]
        for q in range(ptr[i], ptr[i + 1]):
            k = idx[q]
            sq = abs_L0_ii[k] if k <= i else 1.0
            cq = crout0[q] + sq * (np.float64(dr[q]) + 1j * np.float64(di[q]))
            if k > i:
                acc += cq * x_B[k]
            else:
                LHz[k] += np.conj(cq) * zi
        Ux[i] = acc
    # Second pass: each entry's derivative t (with respect to conj(entry)), times the entry's scale s (the
    # entry is crout0 + s correction), split into the real and imaginary parts of the correction
    for i in range(n):
        zi = z[i]
        lz = LHz[i]
        for q in range(ptr[i], ptr[i + 1]):
            k = idx[q]
            if k <= i:
                t = -zi * np.conj(Ux[k])
                sq = abs_L0_ii[k]
            else:
                t = -lz * np.conj(x_B[k])
                sq = 1.0
            gr[q] += np.float32(w2 * sq * t.real)
            gi[q] += np.float32(w2 * sq * t.imag)


@nb.njit(cache=True, fastmath=True)
def _ilu_solve(ptr, idx, val, diag, x):
    """(L U) x = b in place, Doolittle storage."""
    n = ptr.size - 1
    for i in range(n):                                 # L y = b: unit lower triangle, entries left of the diagonal
        a = x[i]
        for q in range(ptr[i], diag[i]):
            a -= val[q] * x[idx[q]]
        x[i] = a
    for i in range(n - 1, -1, -1):                     # U x = y: entries right of the diagonal, then the pivot
        a = x[i]
        for q in range(diag[i] + 1, ptr[i + 1]):
            a -= val[q] * x[idx[q]]
        x[i] = a / val[diag[i]]


@nb.njit(cache=True)
def _ilu_solve_adjoint(ptr, idx, val, diag, b):
    """(L U)^H z = b, Doolittle storage."""
    # (L U)^H = U^H L^H: solve U^H t = b, then L^H z = t. Both transposes are triangular in the other sense
    # (U^H lower, L^H upper), and row i of U (or L) in CSR is column i of its transpose, so each solve is a
    # column sweep: once unknown i is final, its contribution is subtracted from the unknowns it couples to.
    t = b.copy()
    n = ptr.size - 1
    for i in range(n):                                 # U^H: forward column sweep over the rows of U
        t[i] = t[i] / np.conj(val[diag[i]])
        ti = t[i]
        for q in range(diag[i] + 1, ptr[i + 1]):
            t[idx[q]] -= np.conj(val[q]) * ti
    for i in range(n - 1, -1, -1):                     # L^H (unit): backward column sweep over the rows of L
        zi = t[i]
        for q in range(ptr[i], diag[i]):
            t[idx[q]] -= np.conj(val[q]) * zi
    return t


@nb.njit(cache=True)
def _csr_adjoint_matvec(ptr, idx, val, w, out):
    """out = M^dagger w (the conjugate transpose) from M's CSR arrays."""
    # (M^dagger w)_k = sum_i conj(M_ik) w_i: a scatter over the rows of M, so its transpose is never formed
    out[:] = 0j
    for i in range(ptr.size - 1):
        wi = w[i]
        for q in range(ptr[i], ptr[i + 1]):
            out[idx[q]] += np.conj(val[q]) * wi


@nb.njit(cache=True)
def _gram_schmidt_forward(V, j, w, H):
    """Modified Gram-Schmidt of w against V[0..j], in place; fills H[0..j, j]."""
    # one pass of modified Gram-Schmidt (the solve in iterative.py runs two); _gram_schmidt_backward undoes
    # exactly this pass
    for i in range(j + 1):
        h = 0j
        for k in range(w.size):
            h += np.conj(V[i, k]) * w[k]
        H[i, j] = h
        for k in range(w.size):
            w[k] -= h * V[i, k]


@nb.njit(cache=True)
def _gram_schmidt_backward(V, Vbar, j, w, wbar, H, G):
    """Reverse sweep of _gram_schmidt_forward for column j (w: the final vector in, the first one out)."""
    # The forward steps, i = 0 .. j, were h = V_i^H w, w <- w - h V_i. Undo them in reverse order, carrying
    # adjoints (derivatives of the loss with respect to the conjugates: wbar for w, Vbar for V, G for H):
    for i in range(j, -1, -1):
        h = H[i, j]
        s = 0j
        for k in range(w.size):
            s += np.conj(V[i, k]) * wbar[k]
        # h enters the loss directly through H (G) and through the subtraction -h V_i (-V_i^H wbar)
        hbar = G[i, j] - s
        ch, chb = np.conj(h), np.conj(hbar)
        for k in range(w.size):
            v = V[i, k]
            wp = w[k] + h * v                          # w before this step, recovered instead of stored
            # V_i enters through -h V_i and through h = V_i^H w; w enters through w itself and through h
            Vbar[i, k] += -ch * wbar[k] + chb * wp
            wbar[k] += hbar * v
            w[k] = wp


class GMRESCycleLoss:
    """The loss of one GMRES restart cycle as a function of the network's correction, and its gradient
    (Sec. IV B; see the comment above). Double precision throughout."""

    def __init__(self, problem, P: SchurComplementPreconditioner, inputs: dict):
        # Everything that does not depend on the correction, prepared once per orbit: M (for M P^-1 and its
        # adjoint), the pattern and the ILU(0) Crout entries the correction is added to, and the transposes of
        # the strip couplings (for the adjoint of the App. A preconditioner)
        M = problem.M
        self.M_csr = (M.indptr.astype(np.int32), M.indices.astype(np.int32), M.data.astype(np.complex128))
        self.n = M.shape[0]
        self.P = P
        F = P.ilu
        self.ptr, self.idx, self.diag = (F.indptr.astype(np.int32), F.indices.astype(np.int32),
                                         F.diagonal.astype(np.int32))
        self.crout0 = inputs["crout0"]
        self.abs_L0_ii = np.abs(F.values[F.diagonal])
        self.M_0B_T, self.M_B0_T = P.M_0B.T, P.M_B0.T
        self.values = np.empty(F.nnz, np.complex128)

    def set_correction(self, correction):
        """Set P_neural from the correction; returns its Doolittle values and its smallest |pivot|."""
        # kept: the gradient kernel re-forms the corrected Crout entries from them
        self.dr = np.ascontiguousarray(correction[:, 0], np.float32)
        self.di = np.ascontiguousarray(correction[:, 1], np.float32)
        _doolittle_values_from_correction(self.ptr, self.idx, self.diag, self.crout0, self.abs_L0_ii,
                                          self.dr, self.di, self.values)
        return self.values, float(np.abs(self.values[self.diag]).min())

    def _apply(self, v):
        """P^-1 v and its bulk part x_B (App. A)."""
        # the block substitution of SchurComplementPreconditioner.apply_into, in double precision:
        #   y_0 = M_0^-1 v_0,  x_B = P_neural^-1 (v_B - M_B0 y_0),  x_0 = y_0 - M_0^-1 M_0B x_B
        # x_B is also returned: it is the vector the factor entries act on, needed for the gradient
        P = self.P
        y_0 = P.M_0_lu.solve(v[P.strip])
        x_B = v[P.bulk] - P.M_B0 @ y_0
        _ilu_solve(self.ptr, self.idx, self.values, self.diag, x_B)
        x = np.empty(v.size, np.complex128)
        x[P.bulk] = x_B
        x[P.strip] = y_0 - P.M_0_lu.solve(P.M_0B @ x_B)
        return x, x_B

    def _adjoint(self, g):
        """P^-dagger g and its bulk part."""
        # _apply's three steps transposed and run backwards. A^H u is computed as conj(A^T conj(u)), so the
        # sparse blocks are never conjugated. First the sensitivity of x_B through x_0 = ... - M_0^-1 M_0B x_B:
        #   g_B <- g_B - M_0B^H M_0^-H g_0,  then  z = P_neural^-H g_B,  then  out_0 = M_0^-H (g_0 - M_B0^H z)
        # z, the bulk part, is what the gradient kernel needs.
        P = self.P
        g_B = g[P.bulk] - np.conj(self.M_0B_T @ np.conj(P.M_0_lu.solve(g[P.strip], trans="H")))
        z = _ilu_solve_adjoint(self.ptr, self.idx, self.values, self.diag, np.ascontiguousarray(g_B))
        out = np.empty(g.size, np.complex128)
        out[P.bulk] = z
        out[P.strip] = P.M_0_lu.solve(g[P.strip] - np.conj(self.M_B0_T @ np.conj(z)), trans="H")
        return out, z

    def loss_and_gradient(self, r, N, C, gr, gi, weight):
        """log(rho / beta) of the cycle of N Arnoldi steps from r with deflation rows C (None: a plain GMRES
        cycle); adds weight * gradient into gr, gi (float32, per edge)."""
        # C holds the deflation vectors as rows (as in DeflatedRestartGMRES); the projection removes their span
        Cc = None if C is None else np.conj(C)

        def project(w):                               # w <- (I - C C^H) w, Hermitian
            if C is not None:
                w -= (C.T @ (Cc @ w.astype(C.dtype))).astype(np.complex128)

        n = r.size
        beta = float(np.linalg.norm(r))
        V = np.empty((N + 1, n), np.complex128)
        V[0] = r / beta
        H = np.zeros((N + 1, N), np.complex128)
        w = np.empty(n, np.complex128)
        # ---- forward: the cycle as GMRES runs it, in double precision, keeping V and H. Each step:
        # w = (I - C C^H) M P^-1 v_j, orthogonalized against v_0 .. v_j (column j of H), then normalized
        for j in range(N):                            # forward: projected Arnoldi
            x, _ = self._apply(V[j])
            csr_matvec(*self.M_csr, x, w)
            project(w)
            _gram_schmidt_forward(V, j, w, H)
            H[j + 1, j] = eta = np.linalg.norm(w)
            V[j + 1] = w / eta
        # ---- the cycle's end residual: rho = min_y || beta e1 - H y ||, the GMRES residual after N steps
        e1 = np.zeros(N + 1, np.complex128)
        e1[0] = beta
        y = np.linalg.lstsq(H, e1, rcond=None)[0]
        res = e1 - H @ y
        rho = float(np.linalg.norm(res))
        # Adjoints throughout are derivatives with respect to the conjugate (Wirtinger, d loss / d conj(H));
        # the gradient of a real parameter is twice the real part, hence the 2 passed to _accumulate_gradient.
        # y minimizes the residual, so only H's explicit appearance counts: d(rho^2) = -2 Re[res^H dH y],
        # and loss = log(rho) - log(beta) = (1/2) log(rho^2) + const.
        G = -np.outer(res, np.conj(y)) / (2 * rho ** 2)   # d loss / d H
        Vbar = np.zeros((N + 1, n), np.complex128)
        MHw = np.empty(n, np.complex128)
        # ---- reverse: undo the Arnoldi steps last to first. Vbar[j + 1] is complete when step j is reached,
        # because v_{j+1} is used only by step j (which made it) and by the later steps (already undone).
        # r, and hence beta and v_0, are held fixed, so the sweep stops at v_0.
        for j in range(N - 1, -1, -1):                # reverse sweep
            # normalization v_{j+1} = w / eta with eta = ||w|| = H[j+1, j]: eta reaches the loss through H
            # (2 Re G) and through v_{j+1} (-2 Re <Vbar, w> / eta^2), and d eta / d conj(w) = w / (2 eta)
            eta = H[j + 1, j].real
            wfin = eta * V[j + 1]
            c_eta = 2 * G[j + 1, j].real - 2 * np.vdot(Vbar[j + 1], wfin).real / eta ** 2
            wbar = Vbar[j + 1] / eta + (c_eta / (2 * eta)) * wfin
            # Gram-Schmidt: adds the sensitivities of v_0 .. v_j, and turns wbar into that of the vector
            # before orthogonalization
            _gram_schmidt_backward(V, Vbar, j, wfin, wbar, H, G)
            # the projection is Hermitian, so it is its own adjoint; then w = M x gives xbar = M^dagger wbar,
            # and x = P^-1 v_j gives vbar = P^-dagger xbar (z: its bulk part)
            project(wbar)
            _csr_adjoint_matvec(*self.M_csr, wbar, MHw)
            vbar, z = self._adjoint(MHw)
            Vbar[j] += vbar
            # P^-1 also depends on the factor entries: their share of the gradient pairs x_B (forward) with z
            # (adjoint) on the pattern of M_S
            _, x_B = self._apply(V[j])                # recomputed rather than stored
            _accumulate_gradient(self.ptr, self.idx, self.diag, self.crout0, self.abs_L0_ii, self.dr, self.di,
                                 x_B, z, gr, gi, 2.0 * weight)
        return float(np.log(rho / beta))


class _MeanCycleLoss(torch.autograd.Function):
    """The loss L: the mean cycle loss over the recorded cycles, as a differentiable function of the
    correction (edges, 2)."""

    @staticmethod
    def forward(ctx, correction, loss: GMRESCycleLoss, cycles):
        # The gradient is hand-written, so it is computed here together with the loss (each cycle's forward
        # and reverse sweeps share V and H) and saved for backward, which only scales it.
        _, min_pivot = loss.set_correction(correction.detach().numpy())
        gr = np.zeros(correction.shape[0], np.float32)
        gi = np.zeros(correction.shape[0], np.float32)
        f = 0.0
        with np.errstate(all="ignore"):
            for r, C, N in cycles:
                f += loss.loss_and_gradient(r, N, C, gr, gi, 1.0 / len(cycles)) / len(cycles)
        # a zero pivot (P_neural singular) or an overflow: report NaN so that train() skips this update
        if not (min_pivot > 0 and np.isfinite(f) and np.isfinite(gr).all() and np.isfinite(gi).all()):
            f, gr[:], gi[:] = float("nan"), 0.0, 0.0
        ctx.save_for_backward(torch.from_numpy(np.stack([gr, gi], axis=1)))
        return torch.tensor(f, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_out):
        (g,) = ctx.saved_tensors
        # chain rule into the network: d loss / d correction; the other two arguments are not tensors
        return grad_out * g, None, None


# ======================================================================
# Training (Sec. IV B)
# ======================================================================

@dataclass(frozen=True)
class TrainingParameters:
    """The training of Sec. IV B (the values used in the paper are in scripts/settings.py)."""

    training_frequencies: tuple        #: M omega / m of the training orbits (one weight update each per epoch)
    validation_frequencies: tuple      #: M omega / m of the held-out orbits that score each epoch
    loss_cycles: tuple                 #: the restart cycles c the loss averages over
    learning_rate: float               #: Adam's learning rate
    warmup_updates: int                #: linear learning-rate warm-up over the first updates
    gradient_clip: float               #: gradient-norm clipping
    min_improvement: float             #: an epoch must cut the held-out iterations by this fraction ...
    learning_rate_decay: float         #: ... else it is reverted and the learning rate multiplied by this
    patience: int                      #: stop after this many epochs in a row without enough improvement
    max_epochs: int
    seed: int = 0                      #: for the order of the training orbits in each epoch


def _gmres_solve(problem, P, gmres: GMRESParameters, cycle_start_callback=None):
    solver = DeflatedRestartGMRES(problem.M, P, gmres.restart_length, gmres.deflation_vectors, np.complex64)
    _, info = solver.solve(problem.b, gmres.tolerance, gmres.max_iterations, cycle_start_callback=cycle_start_callback)
    solver.release()                                  # free the Krylov basis before the next problem is built
    return info


def evaluate(network, m: int, frequencies, grid_parameters: GridParameters, gmres: GMRESParameters,
             log=print, label="") -> list[dict]:
    """GMRES iterations on each orbit (M omega / m in `frequencies`) with textbook ILU(0) and with the
    network (float32, uncompiled)."""
    rows = []
    for w in frequencies:
        problem = build_problem(m, orbital_radius(w), grid_parameters)
        # the same P twice: first with its ILU(0) values, then with the network's values set into it
        P = SchurComplementPreconditioner(problem, "ilu0")
        textbook = _gmres_solve(problem, P, gmres)
        inputs = gnn_inputs(P, problem)
        with torch.no_grad():
            correction = network(*MessagePassingGNN.tensors(inputs)).numpy()
        P.set_bulk_factor_values(neural_factor_values(P, inputs, correction))
        net = _gmres_solve(problem, P, gmres)
        row = dict(omega=w, r0=problem.orbit.r0, textbook=textbook["iterations"],
                   textbook_converged=textbook["converged"], gnn=net["iterations"], gnn_converged=net["converged"])
        log(f"  [{label}] r0 = {row['r0']:.3f}: textbook ILU(0) {row['textbook']} its"
            f"{'' if row['textbook_converged'] else ' (not converged)'}, network {row['gnn']} its"
            f"{'' if row['gnn_converged'] else ' (not converged)'}")
        rows.append(row)
    return rows


def train(m: int, init, out_dir, *, grid_parameters: GridParameters, gmres: GMRESParameters,
          training: TrainingParameters, log=None):
    """Train the network `init` (a checkpoint path, or a MessagePassingGNN) for mode m with the GMRES cycle
    loss (Sec. IV B).

    Writes <out_dir>/gnn_m<m>.pt (the best network), <out_dir>/gnn_m<m>_last.pt (the last epoch, with the
    optimizer state) and <out_dir>/training.jsonl. The score of an epoch is the total GMRES iterations on
    the held-out orbits that textbook ILU(0) or the starting network solves (an orbit neither solves would
    only contribute a constant gmres.max_iterations)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / f"gnn_m{m}.pt", out_dir / f"gnn_m{m}_last.pt"
    journal = open(out_dir / "training.jsonl", "a")
    t_ = training

    def log_(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if log:
            log(line)

    def record(**kw):
        journal.write(json.dumps(dict(m=m, **kw)) + "\n")
        journal.flush()

    # ---- setup: the starting network and a fresh Adam state
    network = load_network(init) if not isinstance(init, MessagePassingGNN) else init
    network.train()
    opt = torch.optim.Adam(network.parameters(), lr=t_.learning_rate)
    log_(f"training {init} for m = {m}: {sum(p.numel() for p in network.parameters()):,} parameters, "
         f"GMRES({gmres.restart_length}, deflation {gmres.deflation_vectors}), cycles {list(t_.loss_cycles)}, "
         f"learning rate {t_.learning_rate:g}")

    # the held-out score: total iterations over the counted orbits, an unsolved orbit counting as the cap
    def score(rows):
        return int(sum(r["gnn"] if r["gnn_converged"] else gmres.max_iterations
                       for r in rows if r["r0"] in counted))

    # ---- epoch 0: score the starting network. Orbits that neither ILU(0) nor the starting network solves
    # are left out of every later score; best_state is the snapshot an unsuccessful epoch reverts to.
    rows = evaluate(network, m, t_.validation_frequencies, grid_parameters, gmres, log_, "start")
    counted = {r["r0"] for r in rows if r["textbook_converged"] or r["gnn_converged"]}
    best = score(rows)
    record(kind="eval", epoch=0, rows=rows, total=best)
    log_(f"held-out score {best} (orbits r0 = {', '.join(f'{r:.3f}' for r in sorted(counted))})")
    best_state = copy.deepcopy(dict(model=network.state_dict(), opt=opt.state_dict()))
    save_network(network, best_path, m=m, epoch=0, heldout=rows)
    lr_base, bad, step = t_.learning_rate, 0, 0
    order = list(t_.training_frequencies)
    rng = random.Random(t_.seed)
    t0 = time.perf_counter()
    k = gmres.deflation_vectors
    for epoch in range(1, t_.max_epochs + 1):
        rng.shuffle(order)
        iters = []
        # ---- one weight update per training orbit, in a shuffled order
        for w in order:
            step += 1
            # the forward pass with the gradient recorded (float32, checkpointed chunks), on this orbit's M_S
            problem = build_problem(m, orbital_radius(w), grid_parameters)
            P = SchurComplementPreconditioner(problem, "ilu0")
            inputs = gnn_inputs(P, problem)
            correction = network(*MessagePassingGNN.tensors(inputs))
            # the current network's P_neural, in the solver's preconditioner
            loss_ops = GMRESCycleLoss(problem, P, inputs)
            values, _ = loss_ops.set_correction(correction.detach().numpy())
            P.set_bulk_factor_values(values)
            got = {}

            def keep(c, r, C):                        # the cycles the loss looks at, with their N_c
                if c in t_.loss_cycles:
                    got[c] = (r.copy(), None if C is None else C.copy(),
                              gmres.restart_length if C is None else gmres.restart_length - k)
            # A full GMRES solve with the current network records, at the start of each loss cycle, its residual
            # and deflation rows; a cycle the solve never reaches (it converged first) drops out of the mean.
            # N_c: the first cycle builds all restart_length vectors; later ones restart_length - k new ones.
            info = _gmres_solve(problem, P, gmres, keep)
            iters.append(info["iterations"])
            opt.zero_grad()
            # the loss and its gradient with respect to the correction; backward() carries it into the weights
            loss = _MeanCycleLoss.apply(correction, loss_ops, [got[c] for c in sorted(got)])
            if not torch.isfinite(loss):
                log_(f"  step {step}: non-finite loss at r0 = {problem.orbit.r0:.3f}; skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), t_.gradient_clip)
            # Adam's moment estimates are poor for the first few updates, so the learning rate ramps up to
            # lr_base over warmup_updates steps (skipped steps count) and then stays there
            if step <= t_.warmup_updates:             # linear warm-up of the fresh Adam state
                for g in opt.param_groups:
                    g["lr"] = lr_base * step / t_.warmup_updates
            opt.step()
            log_(f"  epoch {epoch} step {step}: r0 = {problem.orbit.r0:.3f}, {info['iterations']} its before the "
                 f"step, loss {loss.item():.4f} ({(time.perf_counter() - t0) / 60:.1f} min)")
            record(kind="step", epoch=epoch, step=step, omega=w, iters=info["iterations"], loss=loss.item())
        # ---- end of epoch: score on the held-out orbits. "improved" (any gain) decides whether this network
        # becomes the best one; "significant" (a gain of at least min_improvement) whether the epoch counts as
        # progress. An improved but insignificant epoch is kept, but still decays the learning rate and counts
        # towards the patience.
        rows = evaluate(network, m, t_.validation_frequencies, grid_parameters, gmres, log_, f"epoch {epoch}")
        total = score(rows)
        improved, significant = total < best, total < best * (1.0 - t_.min_improvement)
        if improved:
            best = total
            best_state = copy.deepcopy(dict(model=network.state_dict(), opt=opt.state_dict()))
            save_network(network, best_path, m=m, epoch=epoch, heldout=rows)
        bad = 0 if significant else bad + 1
        if not significant:                           # revert to the best weights, decay the learning rate
            # the new rate is taken before the revert, because loading the old optimizer state restores its rate
            new_lr = opt.param_groups[0]["lr"] * t_.learning_rate_decay
            if not improved:
                network.load_state_dict(best_state["model"])
                opt.load_state_dict(best_state["opt"])
            for g in opt.param_groups:
                g["lr"] = new_lr
        # the last state, with the optimizer, so that a run can be continued
        save_network(network, last_path, m=m, epoch=epoch, opt=opt.state_dict(), best=best, bad_epochs=bad)
        record(kind="eval", epoch=epoch, rows=rows, total=total, best=best, mean_train_iters=float(np.mean(iters)),
               lr=opt.param_groups[0]["lr"])
        log_(f"epoch {epoch}: held-out score {total} (best {best}); mean training iterations {np.mean(iters):.1f}; "
             f"{bad}/{t_.patience} epochs without improvement; learning rate {opt.param_groups[0]['lr']:g}")
        if bad >= t_.patience:
            log_(f"stopping; best network in {best_path}")
            break
    journal.close()
    return best_path
