"""
Iterative solution of M x = b: GMRES with deflated restarts (Sec. III B), the boundary-strip Schur
complement (App. A), and incomplete LU factorizations of it (Sec. III C).

    GMRESParameters                  restart length, deflation, tolerance, iteration cap
                                     (the values used in the paper are in scripts/settings.py)
    DeflatedRestartGMRES             GMRES with deflated restarts, right-preconditioned
    SchurComplementPreconditioner    exact solves on the boundary strips + an approximate inverse of the
                                     Schur complement M_S on the bulk (App. A)
    IncompleteLU                     ILU(0) / ILU(1) of M_S (Sec. III C)
    csr_matvec                       the compiled product M x

HOW THE PIECES FIT TOGETHER (App. A)
------------------------------------
The unknowns split into the boundary strips x_0 = (x_A, x_C) -- the radii coupled to the near-horizon
and outer boundary conditions, assemble.HORIZON_STRIP_RADII and OUTER_STRIP_RADII -- and the bulk x_B.
Ordering x_0 first,

    M = [ M_0   M_0B ]  =  [ I              0 ] [ M_0   M_0B ]        M_S = M_B - M_B0 M_0^-1 M_0B
        [ M_B0  M_B  ]     [ M_B0 M_0^-1    I ] [ 0     M_S  ]        (the Schur complement)

The preconditioner P is this block factorization with M_S replaced by a factorization P_S that is
cheap to invert -- textbook ILU(0) L0 U0 or ILU(1) L1 U1 (Sec. III C), the neural (L0 + dL)(U0 + dU)
(Sec. IV, neural.py), or for comparison only the diagonal of M_S (point Jacobi). Applying P^-1 to v:

    y_0 = M_0^-1 v_0                       exact strip solve (SuperLU's factors of M_0)
    x_B = P_S^-1 (v_B - M_B0 y_0)          the bulk solve
    x_0 = y_0 - (M_0^-1 M_0B) x_B          M_0^-1 M_0B is dense on the few interface columns

GMRES runs on M with P as a right preconditioner. Since the strip part of P is exact,

    M P^-1 = L diag(I, M_S P_S^-1) L^-1,       L = [ I 0 ; M_B0 M_0^-1 I ],

so the iteration is governed by M_S P_S^-1: it effectively solves M_S x_B = b_S, with x_0 following
exactly, as in App. A.

PRECISION
---------
The Krylov basis and the factor values are stored in complex64 (half the memory traffic); every sum,
the residuals and the solution are complex128, and each restart cycle begins from the true residual
b - M x, so single precision costs efficiency, never accuracy (Sec. IV B).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numba as nb
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg.blas import get_blas_funcs

from .assemble import HORIZON_STRIP_RADII, N_COMPONENTS, OUTER_STRIP_RADII

__all__ = [
    "GMRESParameters",
    "DeflatedRestartGMRES",
    "IdentityPreconditioner",
    "SchurComplementPreconditioner",
    "IncompleteLU",
    "BULK_FACTORIZATIONS",
    "boundary_strip_mask",
    "csr_matvec",
]


# ======================================================================
# GMRES with deflated restarts (Sec. III B)
# ======================================================================

@dataclass(frozen=True)
class GMRESParameters:
    """The settings of a GMRES solve (Sec. III B)."""

    restart_length: int        #: Krylov vectors per restart cycle (including the deflation vectors)
    deflation_vectors: int     #: vectors carried from one cycle to the next ("the slowest directions")
    tolerance: float           #: stop when ||b - M x|| / ||b|| <= tolerance
    max_iterations: int        #: cap on the total number of iterations


@nb.njit(parallel=True, cache=True)
def csr_matvec(ptr, idx, val, x, out):
    """out = M x for M in CSR arrays (parallel over rows)."""
    for i in nb.prange(ptr.size - 1):
        acc = 0j
        for q in range(ptr[i], ptr[i + 1]):
            acc += val[q] * x[idx[q]]
        out[i] = acc


class IdentityPreconditioner:
    """'No preconditioner' (P = I), with the apply_into interface DeflatedRestartGMRES expects."""

    def apply_into(self, v, out):
        out[:] = v
        return out


def _givens(R, cs, sn, g, h, j):
    """Rotate column j of the Hessenberg matrix (h, length j + 2) into R[:, j]; return |g[j + 1]|, the
    residual norm after step j."""
    # GMRES minimizes || beta e1 - H y || over y, with H the (j + 2) x (j + 1) Hessenberg matrix built so far.
    # Rather than solving that least-squares problem from scratch at every step, H is reduced to upper
    # triangular form R one column at a time by Givens rotations (one new rotation per step), and the same
    # rotations are applied to g = beta e1. After step j, |g[j + 1]| is then exactly the residual norm
    # || b - M x_j || (in exact arithmetic), at no extra cost and without forming x_j.
    col = h.copy()
    for i in range(j):                        # bring the new column up to date with the earlier rotations
        c, s = cs[i], sn[i]
        t = c * col[i] + s * col[i + 1]
        col[i + 1] = -np.conj(s) * col[i] + np.conj(c) * col[i + 1]
        col[i] = t
    # the new rotation, chosen to zero the subdiagonal entry H[j + 1, j] (complex; real cosine c)
    a, bb = col[j], col[j + 1]
    den = np.hypot(abs(a), abs(bb))
    if den == 0.0:                            # nothing to rotate (breakdown): the identity
        c, s = 1.0, 0.0
    elif a == 0:
        c, s = 0.0, np.conj(bb) / abs(bb)
    else:
        c, s = abs(a) / den, (a / abs(a)) * np.conj(bb) / den
    cs[j], sn[j] = c, s
    col[j] = c * a + s * bb
    col[j + 1] = 0.0
    R[:j + 2, j] = col
    g[j + 1] = -np.conj(s) * g[j]             # rotate the right-hand side as well: g[j + 1] is the residual
    g[j] = c * g[j]
    return abs(g[j + 1])


class DeflatedRestartGMRES:
    """
    GMRES with deflated restarts (Sec. III B; Morgan, SIAM J. Sci. Comput. 24 (2002) 20; in the
    GCRO-DR form of Parks, de Sturler, Mackey, Johnson & Maiti, SIAM J. Sci. Comput. 28 (2006) 1651),
    right-preconditioned.

        gmres = DeflatedRestartGMRES(M, P, restart_length=50, deflation_vectors=10)
        x, info = gmres.solve(b, tolerance=1e-10, max_iterations=6000)
        info: iterations, converged, relative_residual (true ||b - M x|| / ||b||), cycles,
              history (residual estimate per iteration, relative), time

    P is any object with apply_into(v, out), out = P^-1 v. GMRES restarts every `restart_length`
    iterations, but keeps the `deflation_vectors` slowest directions of the previous cycle (the
    harmonic Ritz vectors of smallest magnitude), U, and their images C = M P^-1 U (orthonormal), and
    every new cycle works orthogonally to them. One cycle, with B = M P^-1:

        r <- r - C C^H r                          x picks up P^-1 U C^H r
        Arnoldi on (I - C C^H) B from r            B V = C B_k + V' H     (H: the Hessenberg matrix)
        y = argmin || beta e1 - H y ||             (Givens rotations: a residual estimate every step)
        x += P^-1 (V y - U B_k y)
        harmonic Ritz vectors of [U V] -> the next U, C

    The first cycle (no U yet) is plain GMRES followed by the same update. Every cycle starts from the
    true residual b - M x, so the single-precision basis costs efficiency, never accuracy.
    """

    def __init__(self, M, P, restart_length: int, deflation_vectors: int, basis_dtype=np.complex64):
        if not 0 < deflation_vectors < restart_length:
            raise ValueError("need 0 < deflation_vectors < restart_length")
        # M as raw CSR arrays for the compiled product M x (int32 indices: less memory traffic)
        M = sp.csr_matrix(M)
        M.sort_indices()
        self.M = (M.indptr.astype(np.int32), M.indices.astype(np.int32), M.data.astype(np.complex128))
        self.n = M.shape[0]
        self.P = P
        self.restart_length, self.deflation_vectors = restart_length, deflation_vectors
        self.dtype = np.dtype(basis_dtype)
        # Vectors are stored as ROWS of 2-D arrays (contiguous in memory), so U, C and the basis are (k, n),
        # and a product with "the matrix whose columns are the vectors" is a BLAS gemv on the transpose.
        self.U = self.C = None                # deflation space, (k, n) rows in the basis dtype
        self.W = None                         # [C; V] storage, allocated by solve
        self.gemv = get_blas_funcs("gemv", dtype=self.dtype)
        self.nrm2 = get_blas_funcs("nrm2", dtype=self.dtype)

    def matvec(self, x):
        """M x, in complex128."""
        out = np.empty(self.n, np.complex128)
        csr_matvec(*self.M, np.ascontiguousarray(x, np.complex128), out)
        return out

    def release(self):
        """Free the Krylov basis (the deflation space U, C is kept)."""
        self.W = None

    def solve(self, b, tolerance: float, max_iterations: int, cycle_start_callback=None):
        """Solve M x = b. cycle_start_callback(c, r, C): at the start of every restart cycle c = 1, 2, ...,
        the (projected) residual r the cycle starts from and the C rows in use (None in the first cycle).
        Training uses it to record the cycles of the loss (Sec. IV B)."""
        b = np.ascontiguousarray(b, np.complex128)
        n, restart, k_max, dt = self.n, self.restart_length, self.deflation_vectors, self.dtype
        gemv, nrm2 = self.gemv, self.nrm2
        bnorm = float(np.linalg.norm(b))
        tol = tolerance * bnorm                       # stop when ||b - M x|| <= tolerance ||b||
        x = np.zeros(n, np.complex128)                # the initial guess x = 0, so the first residual is b
        xt = np.empty(n, np.complex128)               # scratch for P^-1 applied to a vector
        hist = np.zeros(max_iterations + restart)
        its = cycles = 0
        t0 = time.perf_counter()
        r = b.copy()
        beta = bnorm
        # One block of rows for the whole solve: W = [C; V], the k deflation vectors followed by the
        # restart - k + 1 Krylov vectors of the current cycle. Together they are the orthonormal basis the
        # new Krylov vectors are orthogonalized against.
        self.W = W = np.empty((k_max + restart + 1, n), dt)
        while beta > tol and its < max_iterations:   # one pass of this loop = one restart cycle
            # ---- start of a cycle: work orthogonally to the deflation space (none in the first cycle)
            k = k_max if self.U is not None else 0
            z0 = None
            if k:
                # r = C z0 + r_perp with z0 = C^H r. The part C z0 is solved at once: C = M P^-1 U, so
                # x += P^-1 U z0 removes it (done below, together with the cycle's own update).
                z0 = gemv(1.0, self.C.T, r.astype(dt), trans=2)     # C^H r
                r = r - (self.C.T @ z0).astype(np.complex128)
                beta = float(np.linalg.norm(r))
                z0 = z0.astype(np.complex128)
                W[:k] = self.C
            if cycle_start_callback is not None:
                cycle_start_callback(cycles + 1, r, self.C if k else None)
            # ---- Arnoldi on (I - C C^H) M P^-1 from r: mk = restart - k new basis vectors
            mk = restart - k
            V = W[k:]                                               # the Krylov vectors, rows of W after C
            V[0] = (r / beta).astype(dt)
            Hb = np.zeros((mk + 1, mk), np.complex128)              # the Hessenberg matrix H
            Bk = np.zeros((k, mk), np.complex128)                   # the components along C, C^H M P^-1 V
            R = np.zeros((mk + 1, mk), np.complex128)               # H reduced to triangular form (_givens)
            cs = np.zeros(mk, np.complex128)
            sn = np.zeros(mk, np.complex128)
            g = np.zeros(mk + 1, np.complex128)                     # beta e1, rotated along with H
            g[0] = beta
            steps = 0
            for j in range(mk):                                     # one pass = one GMRES iteration
                if its >= max_iterations:
                    break
                # w = M P^-1 v_j: right preconditioning, so GMRES minimizes the TRUE residual b - M x
                w = V[j + 1]                                        # the new vector, built in place
                self.P.apply_into(V[j], xt)
                csr_matvec(*self.M, xt, w)
                # Orthogonalize w against [C; v_0 .. v_j]. Classical Gram-Schmidt as two BLAS gemv calls,
                # repeated once ("twice is enough") to recover the orthogonality one pass loses in single
                # precision. The coefficients along C go to Bk, those along V to the column j of H.
                Q = W[:k + j + 1].T                                 # n x (k + j + 1), a view
                for _ in range(2):
                    h = gemv(1.0, Q, w, trans=2)                    # h = Q^H w
                    gemv(-1.0, Q, h, beta=1.0, y=w, overwrite_y=1)  # w -= Q h
                    h = h.astype(np.complex128)
                    Bk[:, j] += h[:k]
                    Hb[:j + 1, j] += h[k:]
                eta = float(nrm2(w))                                # normalize: v_{j+1} = w / ||w||
                Hb[j + 1, j] = eta
                w *= dt.type(1.0 / eta).real
                res = _givens(R, cs, sn, g, Hb[:j + 2, j], j)       # the residual norm after this iteration
                hist[its] = res
                its += 1
                steps = j + 1
                if res <= tol:                                      # converged (by the estimate): stop early
                    break
            if steps == 0:
                break
            # ---- end of the cycle: the GMRES update. y minimizes ||beta e1 - H y||; the triangular system
            # R y = g is that least-squares solution after the Givens rotations.
            y = sla.solve_triangular(R[:steps, :steps], g[:steps])
            update = gemv(1.0, V[:steps].T, y.astype(dt))           # V y
            if k:
                # GCRO-DR: M P^-1 V y also has components along C (Bk y), which the U part cancels:
                # the correction is P^-1 (V y + U (z0 - Bk y)), z0 being the part removed at the start
                gemv(1.0, self.U.T, (z0 - Bk[:, :steps] @ y).astype(dt), beta=1.0, y=update, overwrite_y=1)
            self.P.apply_into(update, xt)                           # back from the preconditioned variable
            x += xt
            cycles += 1
            # The next cycle starts from the TRUE residual, computed in double precision, not from the
            # estimate: rounding in the single-precision basis cannot accumulate across cycles.
            r = b - self.matvec(x)
            beta = float(np.linalg.norm(r))
            if beta <= tol or its >= max_iterations:
                break
            if steps < mk:              # the estimate converged but the true residual did not: restart
                continue
            # a full cycle: keep its slowest directions for the next one
            self._update_deflation_space(Hb, Bk, k, mk)
        residual = float(np.linalg.norm(b - self.matvec(x)) / bnorm)
        return x, dict(iterations=its, converged=residual <= tolerance, relative_residual=residual,
                       cycles=cycles, history=hist[:its] / bnorm, time=time.perf_counter() - t0)

    def _update_deflation_space(self, Hb, Bk, k, mk):
        """Keep the harmonic Ritz vectors of smallest |theta| as the next U (and C = M P^-1 U)."""
        k_max, dt, W = self.deflation_vectors, self.dtype, self.W
        V = W[k:]
        mm = k + mk
        # The cycle produced the relation  B [Ut V_mk] = W G,  B = M P^-1, W = [C; V_mk+1] (orthonormal), with
        #     G = [ D  Bk ]      D = diag(1 / |u_i|) (Ut: the rows of U scaled to unit length, B Ut = C D)
        #         [ 0  H  ]      H = the (mk + 1) x mk Hessenberg matrix
        # (in the first cycle there is no U: G = H). Harmonic Ritz vectors of B on the space [Ut V_mk] are the
        # solutions of  G^H G z = theta G^H W^H [Ut V_mk] z; those with the smallest |theta| approximate the
        # eigenvectors of B with the smallest eigenvalues, the directions that slow GMRES down.
        G = np.zeros((mm + 1, mm), np.complex128)
        WV = np.zeros((mm + 1, mm), np.complex128)                  # W^H [Ut V_mk]
        if k:
            D = 1.0 / np.linalg.norm(self.U, axis=1).astype(np.float64)
            G[:k, :k] = np.diag(D)
            G[:k, k:] = Bk
            Ut = self.U * D.astype(np.float32)[:, None]             # unit rows
            WV[:, :k] = np.conj(np.conj(Ut) @ W[:k + mk + 1].T).T
        G[k:, k:] = Hb
        WV[k:k + mk, k:] = np.eye(mk)                               # W^H V_mk: V_mk is part of W
        theta, Z = sla.eig(G.conj().T @ G, G.conj().T @ WV)
        theta = np.where(np.isfinite(theta), theta, np.inf)
        Pk = Z[:, np.argsort(np.abs(theta))[:k_max]]               # the k_max smallest |theta|
        # Y = [Ut V_mk] Pk spans the new deflation space, and B Y = W G Pk. With the QR factorization
        # G Pk = Q Rr: B (Y Rr^-1) = W Q, so U = Y Rr^-1 and C = W Q, which has orthonormal rows and C = B U.
        Q, Rr = np.linalg.qr(G @ Pk)
        Y = Pk[k:].T.astype(dt) @ V[:mk]
        if k:
            Y += Pk[:k].T.astype(dt) @ Ut
        self.C = np.ascontiguousarray(Q.T.astype(dt) @ W[:k + mk + 1])
        self.U = np.ascontiguousarray(np.linalg.inv(Rr.T).astype(dt) @ Y)


# ======================================================================
# Incomplete LU of the Schur complement (Sec. III C)
#
# Storage: one CSR (indptr, indices, values) holding the strict part of a UNIT lower-triangular factor
# and all of the upper-triangular factor, pivots included, with `diagonal[i]` the position of the pivot
# in row i (Doolittle form, which the triangular solves below use). The paper writes the same product
# in Crout form, M_S ~ L0 U0 with U0 unit upper triangular and the pivots on the diagonal of L0; the two
# differ only in where the pivots are stored, and neural.py converts between them.
# ======================================================================

def _with_diagonal(M) -> sp.csr_matrix:
    """0/1 structure of M (explicit zeros dropped) plus the diagonal, sorted CSR."""
    # Every row needs a diagonal position, for its pivot, even where M has a structural zero there.
    P = sp.csr_matrix(M, copy=True)
    P.eliminate_zeros()
    P.data = np.ones(P.nnz)
    P = P + sp.identity(M.shape[0], format="csr")
    P.data[:] = 1.0                       # (entries where M already had a diagonal became 2)
    P.sort_indices()
    return P


@nb.njit(cache=True)
def _symbolic_ilu_pattern(n, indptr, indices, p):
    """Symbolic ILU(p) by level of fill (Saad, Iterative Methods, alg. 10.6). Returns the pattern's CSR
    indptr / indices (rows sorted); the input pattern must contain the diagonal."""
    # Levels of fill: an entry of the original pattern has level 0. Eliminating row i with an earlier row k
    # (k < i, entry (i, k) present) creates fill at (i, j) for every entry (k, j > k) of row k, with level
    #     level(i, j) = level(i, k) + level(k, j) + 1,
    # and ILU(p) keeps an entry if its lowest level is <= p. Row i is held as a sorted linked list of its
    # columns, so fill can be inserted in order while the list is being walked.
    out_ptr = np.zeros(n + 1, dtype=np.int64)
    cap = max(indices.size * 4, 16)
    out_idx = np.empty(cap, dtype=np.int64)
    out_lev = np.empty(cap, dtype=np.int64)
    level = np.full(n, -1, dtype=np.int64)          # -1: not in the current row
    nxt = np.full(n + 1, -1, dtype=np.int64)        # sorted linked list of the row, head at n
    diagpos = np.empty(n, dtype=np.int64)
    pos = 0
    for i in range(n):
        head = n
        nxt[head] = -1
        last = head
        for q in range(indptr[i], indptr[i + 1]):   # row i of the pattern, level 0
            c = indices[q]
            level[c] = 0
            nxt[last] = c
            nxt[c] = -1
            last = c
        k = nxt[head]
        while k != -1 and k < i:                    # eliminate with each earlier row k, in order
            lik = level[k]
            prev = k
            for q in range(diagpos[k] + 1, out_ptr[k + 1]):   # the upper part of the finished row k
                j = out_idx[q]
                newlev = lik + out_lev[q] + 1
                if newlev > p:                      # too high a level: this fill is dropped
                    continue
                if level[j] == -1:                  # new fill: insert in sorted position
                    while nxt[prev] != -1 and nxt[prev] < j:
                        prev = nxt[prev]
                    nxt[j] = nxt[prev]
                    nxt[prev] = j
                    level[j] = newlev
                    prev = j
                else:
                    if newlev < level[j]:
                        level[j] = newlev
                    while nxt[prev] != -1 and nxt[prev] <= j:
                        prev = nxt[prev]
            k = nxt[k]
        c = nxt[head]                               # emit row i
        while c != -1:
            if pos >= cap:
                cap *= 2
                t1 = np.empty(cap, dtype=np.int64)
                t1[:pos] = out_idx[:pos]
                out_idx = t1
                t2 = np.empty(cap, dtype=np.int64)
                t2[:pos] = out_lev[:pos]
                out_lev = t2
            if c == i:
                diagpos[i] = pos
            out_idx[pos] = c
            out_lev[pos] = level[c]
            level[c] = -1
            pos += 1
            c = nxt[c]
        out_ptr[i + 1] = pos
    return out_ptr, out_idx[:pos].copy()


@nb.njit(cache=True)
def _numeric_ilu(n, a_ptr, a_idx, a_val, p_ptr, p_idx):
    """IKJ incomplete LU of a matrix (a_*) restricted to the pattern (p_*). Returns (values, diagonal)."""
    # Row-by-row ("IKJ") Gaussian elimination in which every update that would land outside the pattern is
    # simply dropped. Row i of the factor is built in place: its pattern positions are filled with row i of
    # the matrix, then eliminated with the finished rows k < i in increasing order of k.
    val = np.zeros(p_idx.size, dtype=np.complex128)
    diag = np.empty(n, dtype=np.int64)
    where = np.full(n, -1, dtype=np.int64)          # column -> its position in row i (-1: not in the pattern)
    for i in range(n):
        s, e = p_ptr[i], p_ptr[i + 1]
        diag[i] = -1
        for q in range(s, e):
            where[p_idx[q]] = q
            if p_idx[q] == i:
                diag[i] = q
        for q in range(a_ptr[i], a_ptr[i + 1]):     # scatter row i of the matrix into the pattern
            val[where[a_idx[q]]] += a_val[q]
        for q in range(s, e):                       # the strictly lower entries (i, k), k ascending
            k = p_idx[q]
            if k >= i:
                break
            lik = val[q] / val[diag[k]]             # the multiplier l_ik = a_ik / u_kk ...
            val[q] = lik                            # ... stored in place of a_ik (unit L: no diagonal)
            for r in range(diag[k] + 1, p_ptr[k + 1]):   # row i -= l_ik * (upper part of row k) ...
                t = where[p_idx[r]]
                if t >= 0:                          # ... on the pattern only: the dropped fill
                    val[t] -= lik * val[r]
        for q in range(s, e):                       # reset the map for the next row
            where[p_idx[q]] = -1
    return val, diag


class IncompleteLU:
    """ILU(level) of the Schur complement M_S (Sec. III C): level 0 keeps the sparsity pattern of M_S
    (L0 U0), level 1 adds first-level fill (L1 U1). Stored in Doolittle form (see above)."""

    def __init__(self, M_S: sp.csr_matrix, level: int):
        # the pattern: that of M_S (ILU(0)), or with the fill of levels up to `level` added (ILU(level))
        pattern = _with_diagonal(M_S)
        ptr, idx = pattern.indptr.astype(np.int64), pattern.indices.astype(np.int64)
        if level > 0:
            ptr, idx = _symbolic_ilu_pattern(M_S.shape[0], ptr, idx, level)
        # the values: incomplete elimination on that pattern
        self.level = level
        self.indptr, self.indices = ptr, idx
        self.values, self.diagonal = _numeric_ilu(M_S.shape[0], M_S.indptr.astype(np.int64),
                                                  M_S.indices.astype(np.int64), M_S.data.astype(np.complex128),
                                                  ptr, idx)
        if np.any(self.diagonal < 0):
            raise ValueError("the pattern is missing a diagonal entry")

    @property
    def n(self) -> int:
        return self.indptr.size - 1

    @property
    def nnz(self) -> int:
        return self.indices.size


# ======================================================================
# Compiled kernels for applying P^-1
# ======================================================================

@nb.njit(cache=True)
def _superlu_lower_solve(ptr, idx, val, x):
    """L x = b in place; L lower triangular CSR with the diagonal last in each row (SuperLU's L)."""
    for i in range(ptr.size - 1):
        acc = x[i]
        e = ptr[i + 1] - 1
        for q in range(ptr[i], e):
            acc -= val[q] * x[idx[q]]
        x[i] = acc / val[e]


@nb.njit(cache=True)
def _superlu_upper_solve(ptr, idx, val, x):
    """U x = b in place; U upper triangular CSR with the diagonal first in each row (SuperLU's U)."""
    for i in range(ptr.size - 2, -1, -1):
        acc = x[i]
        s = ptr[i]
        for q in range(s + 1, ptr[i + 1]):
            acc -= val[q] * x[idx[q]]
        x[i] = acc / val[s]


@nb.njit(cache=True)
def _strip_solve(Lp, Li, Lv, Up, Ui, Uv, perm_r, perm_c, b, x):
    """x = M_0^-1 b with SuperLU's factors: Pr M_0 Pc = L U."""
    # M_0^-1 = Pc U^-1 L^-1 Pr: permute the rows, two triangular solves, permute back the columns
    y = np.empty(b.size, np.complex128)
    for k in range(b.size):
        y[perm_r[k]] = b[k]
    _superlu_lower_solve(Lp, Li, Lv, y)
    _superlu_upper_solve(Up, Ui, Uv, y)
    for k in range(b.size):
        x[k] = y[perm_c[k]]


@nb.njit(cache=True, fastmath=True)
def _bulk_ilu_solve(Lp, Li, Lv, Up, Ui, Uv, inverse_pivots, x):
    """(L U) x = b in place, with the strict L and U parts stored separately and the inverse pivots
    precomputed. Values may be complex64; the accumulation is complex128."""
    n = Lp.size - 1
    for i in range(n):                              # forward substitution with the unit lower factor
        a = x[i]
        for q in range(Lp[i], Lp[i + 1]):
            a -= np.complex128(Lv[q]) * x[Li[q]]
        x[i] = a
    for i in range(n - 1, -1, -1):                  # back substitution with the upper factor (pivots on its
        a = x[i]                                    # diagonal: multiply by their precomputed inverses)
        for q in range(Up[i], Up[i + 1]):
            a -= np.complex128(Uv[q]) * x[Ui[q]]
        x[i] = a * inverse_pivots[i]


@nb.njit(cache=True)
def _bulk_jacobi_solve(inverse_diagonal, x):
    """x = D^-1 x in place, D = diag(M_S)."""
    for i in range(x.size):
        x[i] = x[i] * inverse_diagonal[i]


@nb.njit(cache=True)
def _split_strip_bulk(v, strip, bulk, v_0, v_B):
    """v_0 = v[strip], v_B = v[bulk]."""
    for k in range(strip.size):
        v_0[k] = v[strip[k]]
    for k in range(bulk.size):
        v_B[k] = v[bulk[k]]


@nb.njit(cache=True)
def _subtract_M_B0_y_0(x_B, rows, Rp, Ri, Rv, y_0):
    """x_B[rows[r]] -= (M_B0 y_0)[rows[r]] (M_B0 is nonzero only on the interface rows)."""
    for r in range(rows.size):
        acc = 0j
        for q in range(Rp[r], Rp[r + 1]):
            acc += Rv[q] * y_0[Ri[q]]
        x_B[rows[r]] -= acc


@nb.njit(cache=True)
def _merge_strip_bulk(x_B, y_0, z_0, strip, bulk, out):
    """out[bulk] = x_B, out[strip] = y_0 - z_0."""
    for k in range(bulk.size):
        out[bulk[k]] = x_B[k]
    for k in range(strip.size):
        out[strip[k]] = y_0[k] - z_0[k]


# ======================================================================
# The preconditioner (App. A)
# ======================================================================

#: The factorizations P_S of the Schur complement SchurComplementPreconditioner offers. The neural
#: one is "ilu0" with new values (neural.py).
BULK_FACTORIZATIONS = ("ilu0", "ilu1", "jacobi")


def boundary_strip_mask(grid) -> np.ndarray:
    """True for the unknowns in the boundary strips x_0 = (x_A, x_C) of App. A: every component and every
    theta at the HORIZON_STRIP_RADII innermost and OUTER_STRIP_RADII outermost radii."""
    i = np.arange(grid.n_rstar)
    radial = (i < HORIZON_STRIP_RADII) | (i >= grid.n_rstar - OUTER_STRIP_RADII)
    return np.broadcast_to(radial[None, :, None], (grid.n_theta, grid.n_rstar, N_COMPONENTS)).ravel().copy()


class SchurComplementPreconditioner:
    """
    The preconditioner P of App. A: exact solves on the boundary strips, and a factorization P_S of the
    Schur complement M_S on the bulk (see the module docstring).

        P = SchurComplementPreconditioner(problem, "ilu0")      # or "ilu1", "jacobi"
        P.apply_into(v, out)                 # out = P^-1 v, the interface DeflatedRestartGMRES uses
        P.set_bulk_factor_values(values)     # new values for P_S on the same pattern (the neural one)

    Attributes used by the neural preconditioner and its training:
        M_S              the Schur complement (CSR, sorted)
        ilu              the textbook IncompleteLU of M_S (None for "jacobi")
        strip, bulk      the indices of x_0 and x_B in x
        M_0_lu           SuperLU's factorization of M_0
        M_0B, M_B0       the coupling blocks (CSR)
        grid
        timings          strip_lu, schur_complement, bulk_factorization (seconds)
    """

    def __init__(self, problem, bulk_factorization: str = "ilu0", factor_precision=np.complex64):
        if bulk_factorization not in BULK_FACTORIZATIONS:
            raise ValueError(f"bulk_factorization must be one of {BULK_FACTORIZATIONS}, not {bulk_factorization!r}")
        self.bulk_factorization = bulk_factorization
        t0 = time.perf_counter()
        M, self.grid = problem.M, problem.grid
        self.n = M.shape[0]
        self.dtype = factor_precision
        # ---- split the unknowns into the boundary strips x_0 = (x_A, x_C) and the bulk x_B, and M into the
        # four blocks M_0, M_0B, M_B0, M_B (the reordering of App. A: indices, no data is moved in x)
        mask = boundary_strip_mask(self.grid)
        self.strip, self.bulk = np.flatnonzero(mask), np.flatnonzero(~mask)
        M_strip_rows = M[self.strip]
        M_bulk_rows = M[self.bulk]
        self.M_0B = M_strip_rows[:, self.bulk].tocsr()
        self.M_B0 = M_bulk_rows[:, self.strip].tocsr()
        M_B = M_bulk_rows[:, self.bulk].tocsr()
        # ---- the exact sparse LU of M_0 (SuperLU). The strips are thin (12 radii of the ~400), so this is
        # cheap, and it is the step that removes the high-order radial boundary conditions from the iteration
        self.M_0_lu = spla.splu(M_strip_rows[:, self.strip].tocsc())
        t1 = time.perf_counter()

        # ---- the Schur complement M_S = M_B - M_B0 M_0^-1 M_0B (App. A).
        # The finite-difference stencils couple the strips to the bulk only across the two interfaces, so
        # M_0B is nonzero only in a few columns (bulk unknowns on the radii next to the strips) and M_B0
        # only in a few rows. M_0^-1 M_0B is therefore formed exactly, as a dense block on those interface
        # columns alone, and the correction M_B0 (M_0^-1 M_0B) only touches interface rows x interface
        # columns: M_S is M_B plus a small dense block, and stays sparse.
        M_0B_csc = self.M_0B.tocsc()
        self.interface_columns = np.flatnonzero(np.diff(M_0B_csc.indptr))     # columns with any entry
        M_0_inverse_M_0B = self.M_0_lu.solve(M_0B_csc[:, self.interface_columns].toarray().astype(np.complex128))
        interface_rows = np.flatnonzero(np.diff(self.M_B0.indptr))           # rows with any entry
        correction = sp.coo_matrix(self.M_B0[interface_rows] @ M_0_inverse_M_0B)
        correction = sp.csr_matrix((correction.data, (interface_rows[correction.row],
                                                      self.interface_columns[correction.col])), shape=M_B.shape)
        M_S = (M_B - correction).tocsr()
        M_S.sum_duplicates()
        M_S.sort_indices()
        self.M_S = M_S
        t2 = time.perf_counter()

        # ---- the factorization P_S of M_S

        if bulk_factorization == "jacobi":
            self.ilu = None
            self._inverse_diagonal = (1.0 / M_S.diagonal()).astype(np.complex128)
        else:
            self.ilu = IncompleteLU(M_S, level=0 if bulk_factorization == "ilu0" else 1)
        t3 = time.perf_counter()
        self.timings = dict(strip_lu=t1 - t0, schur_complement=t2 - t1, bulk_factorization=t3 - t2)

        # ---- data for the compiled apply_into, prepared once: the interface rows of M_B0, the dense
        # M_0^-1 M_0B, SuperLU's factors as plain CSR arrays (int32 indices, values in factor_precision), and
        # work buffers, so that applying P^-1 allocates nothing but the small z_0
        R = self.M_B0[interface_rows].tocsr()
        self._interface_rows = interface_rows.astype(np.int32)
        self._M_B0_interface = (R.indptr.astype(np.int32), R.indices.astype(np.int32), R.data.astype(np.complex128))
        self._interface_columns = self.interface_columns.astype(np.int32)
        self._M_0_inverse_M_0B = np.ascontiguousarray(M_0_inverse_M_0B, factor_precision)
        L, U = sp.csr_matrix(self.M_0_lu.L), sp.csr_matrix(self.M_0_lu.U)
        L.sort_indices()
        U.sort_indices()
        self._M_0_factors = (L.indptr.astype(np.int32), L.indices.astype(np.int32), L.data.astype(factor_precision),
                             U.indptr.astype(np.int32), U.indices.astype(np.int32), U.data.astype(factor_precision),
                             self.M_0_lu.perm_r.astype(np.int32), self.M_0_lu.perm_c.astype(np.int32))
        self._strip32, self._bulk32 = self.strip.astype(np.int32), self.bulk.astype(np.int32)
        self._v_0 = np.empty(self.strip.size, np.complex128)
        self._y_0 = np.empty(self.strip.size, np.complex128)
        self._x_B = np.empty(self.bulk.size, np.complex128)
        if self.ilu is None:
            return
        # The triangular solves read the strict lower part, the strict upper part and the pivots from three
        # separate arrays (no branch on the column inside the loops). Record once where each part sits in
        # the combined storage; set_bulk_factor_values then gathers any values on this pattern into them.
        F = self.ilu
        row = np.repeat(np.arange(F.n), np.diff(F.indptr))
        self._lower_positions = np.flatnonzero(F.indices < row)       # strict-L entries
        self._upper_positions = np.flatnonzero(F.indices > row)       # strict-U entries
        count = lambda q: np.concatenate([[0], np.cumsum(np.bincount(row[q], minlength=F.n))]).astype(np.int32)
        self._Lp, self._Li = count(self._lower_positions), F.indices[self._lower_positions].astype(np.int32)
        self._Up, self._Ui = count(self._upper_positions), F.indices[self._upper_positions].astype(np.int32)
        self.set_bulk_factor_values(F.values)

    def set_bulk_factor_values(self, values):
        """Use `values` for P_S (Doolittle storage on the pattern of self.ilu): the textbook ILU's own
        values, or the neural ones (neural.py)."""
        values = np.asarray(values)
        self._Lv = values[self._lower_positions].astype(self.dtype)
        self._Uv = values[self._upper_positions].astype(self.dtype)
        # the inverse pivots, from the pivots as stored (rounded to factor_precision), so that the solve is
        # exactly the triangular factorization held in memory
        self._inverse_pivots = 1.0 / values[self.ilu.diagonal].astype(self.dtype).astype(np.complex128)

    def apply_into(self, v, out):
        """out = P^-1 v."""
        # The block LU of App. A, solved by block forward and back substitution:
        #     [ M_0   M_0B ] [ x_0 ]   [ v_0 ]      y_0 = M_0^-1 v_0
        #     [ M_B0  P_S* ] [ x_B ] = [ v_B ]      x_B = P_S^-1 (v_B - M_B0 y_0)
        #                                           x_0 = y_0 - M_0^-1 M_0B x_B
        # (P_S* = P_S + M_B0 M_0^-1 M_0B: P is M with M_S replaced by P_S.)
        _split_strip_bulk(v, self._strip32, self._bulk32, self._v_0, self._x_B)  # v_0, and v_B into x_B
        _strip_solve(*self._M_0_factors, self._v_0, self._y_0)                    # y_0 = M_0^-1 v_0
        # x_B = v_B - M_B0 y_0: only the interface rows of M_B0 are nonzero
        _subtract_M_B0_y_0(self._x_B, self._interface_rows, *self._M_B0_interface, self._y_0)
        if self.ilu is None:                                                      # x_B = P_S^-1 x_B
            _bulk_jacobi_solve(self._inverse_diagonal, self._x_B)
        else:
            _bulk_ilu_solve(self._Lp, self._Li, self._Lv, self._Up, self._Ui, self._Uv, self._inverse_pivots,
                            self._x_B)
        # z_0 = M_0^-1 M_0B x_B, a small dense product: M_0B has only the interface columns
        z_0 = self._M_0_inverse_M_0B @ self._x_B[self._interface_columns].astype(self.dtype, copy=False)
        _merge_strip_bulk(self._x_B, self._y_0, z_0, self._strip32, self._bulk32, out)   # x_0 = y_0 - z_0
        return out

    def apply(self, v):
        """P^-1 v, in a new array."""
        return self.apply_into(np.ascontiguousarray(v, np.complex128), np.empty(self.n, np.complex128))

    @property
    def memory_mb(self) -> float:
        """Stored factors: SuperLU of M_0 (complex128), M_0^-1 M_0B, and P_S (its values, or the diagonal)."""
        item = np.dtype(self.dtype).itemsize
        bulk = self._inverse_diagonal.nbytes if self.ilu is None else self.ilu.nnz * item
        return ((self.M_0_lu.L.nnz + self.M_0_lu.U.nnz) * 16 + self._M_0_inverse_M_0B.size * item + bulk) / 2**20
