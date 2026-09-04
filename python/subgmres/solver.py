# -*- coding: utf-8 -*-
"""
Generalized Minimum Residual Method with residual subvector monitoring.

Python port of src/subgmres/subgmres.m, implementing the algorithm described
in

  Herzog, Soodhalter: A Unified View of Residual Norm Minimizing Krylov
  Subspace Methods (in preparation)

Implemented by Roland Herzog, from a first implementation and the
subvector-monitoring derivation of Kirk M. Soodhalter.
Released under the MIT License; see LICENSE.
If this code is used in a scientific publication, please cite as

  Roland Herzog, Kirk M. Soodhalter:
  A Unified View of Residual Norm Minimizing Krylov Subspace Methods

Departures from subgmres.m, both driven by Python not having a `nargout`
equivalent:
  * What used to be output-count-gated ("need_XHIST" etc., derived from
    nargout) is now a set of explicit `compute_*` keyword arguments. XHIST
    and RHIST genuinely gate extra work (an extra preconditioner/matrix
    application every iteration); the others only gate whether an
    already-available array is attached to the result.
  * All outputs are attached to a single SubGMRESResult object rather than
    a positional tuple, so there is no arity to size by intent.
  * varargin passthrough to AFUN/PFUN/MFUN is replaced by ordinary Python
    closures: bake any extra parameters into the callable (or LinearOperator)
    you pass for A, P, or M.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg

from .operators import coerce_operator

_RELTOL = np.sqrt(np.finfo(float).eps)


class SubGMRESNotHermitianError(ValueError):
    """Raised when a quantity that must be real (a squared M-inverse-norm)
    has a non-negligible imaginary part, indicating that M is not Hermitian."""


class SubGMRESNotPositiveDefiniteError(ValueError):
    """Raised when a quantity that must be nonnegative (a squared
    M-inverse-norm) is negative beyond roundoff, indicating that M is not
    positive definite."""


@dataclass
class SubGMRESResult:
    """Result of a subgmres() call. Fields below the first block are `None`
    unless the matching `compute_*` keyword argument was set to True."""

    x: np.ndarray
    flag: int  # 0 = converged, 1 = maxiter reached
    iter: int
    sub_norms: np.ndarray  # M-inverse-norms of the final residual subvectors (+ total)
    ix: list  # the subvector index partition actually used
    tol_table: np.ndarray  # [rtol, atol] columns, one row per subvector plus the total

    sub_norms_hist: Optional[np.ndarray] = None
    x_hist: Optional[np.ndarray] = None
    r_hist: Optional[np.ndarray] = None
    v_hist: Optional[np.ndarray] = None
    w_hist: Optional[np.ndarray] = None
    z_hist: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None
    rh: Optional[np.ndarray] = None


def _sign(z):
    # MATLAB's sign(z) for complex z is z/abs(z) (0 at z == 0), unlike
    # numpy.sign which for complex input returns the sign of the real part.
    az = abs(z)
    return z / az if az != 0 else 0 * z


def _assert_real(z, description):
    # Return the real part of z after verifying that its imaginary part is
    # negligible relative to its magnitude; see subgmres.m's assertReal.
    z = np.atleast_1d(z)
    if np.any(np.abs(z.imag) > _RELTOL * np.abs(z)):
        raise SubGMRESNotHermitianError(
            f"{description} has a non-negligible imaginary part; "
            "the inner product M does not appear to be Hermitian."
        )
    return z.real.copy()


def _assert_nonnegative(z, scale, description):
    # Return max(z, 0), verifying z is not negative beyond a roundoff-level
    # tolerance relative to scale; see subgmres.m's assertNonnegative.
    z = np.atleast_1d(z)
    scale = np.atleast_1d(scale)
    if np.any(z < -_RELTOL * scale):
        raise SubGMRESNotPositiveDefiniteError(
            f"{description} is negative; the inner product M does not appear to be positive definite."
        )
    return np.maximum(z, 0.0)


def _parse_subvectors(ix, n):
    # Returns a list of 1-D integer index arrays (0-based) partitioning
    # range(n). ix may be None (no subvectors), a 1-D array of ascending
    # 0-based subvector-starting indices, or an explicit list of index
    # arrays; see subgmres.m's IX parsing.
    if ix is None or len(ix) == 0:
        return [np.arange(n)]
    first = ix[0]
    if isinstance(first, (np.ndarray, list, tuple)):
        return [np.asarray(sub, dtype=int) for sub in ix]
    starts = np.asarray(ix, dtype=int).ravel()
    bounds = np.concatenate([starts, [n]])
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(len(starts))]


def _parse_tolerance(tol, n_sub, default_last, name):
    # Returns an array of length n_sub+1: componentwise subvector tolerances
    # followed by the total-vector tolerance; see subgmres.m's rtol/atol
    # parsing.
    if tol is None:
        return np.concatenate([np.full(n_sub, np.inf), [default_last]])
    tol = np.atleast_1d(np.asarray(tol, dtype=float))
    if tol.size == 1:
        return np.concatenate([np.full(n_sub, np.inf), tol])
    if tol.size == n_sub:
        return np.concatenate([tol, [np.inf]])
    if tol.size == n_sub + 1:
        return tol
    raise ValueError(f"{name} must have length 1, {n_sub}, or {n_sub + 1}.")


def _infer_dtype(*items):
    # Combine the dtypes of whichever of b, x0, A, P, M carry one (a
    # LinearOperator built from a plain callable has dtype=None unless the
    # caller specified it). Falls back to float64. Passing a complex-valued
    # b, x0, or an explicitly complex LinearOperator forces complex
    # arithmetic throughout.
    dtypes = [getattr(item, "dtype", None) for item in items if item is not None]
    dtypes = [dt for dt in dtypes if dt is not None]
    return np.result_type(*dtypes) if dtypes else np.float64


def subgmres(
    A,
    b,
    ix=None,
    rtol=None,
    atol=None,
    maxiter=None,
    P=None,
    M=None,
    x0=None,
    *,
    compute_sub_norms_hist=False,
    compute_x_hist=False,
    compute_r_hist=False,
    compute_v_hist=False,
    compute_w_hist=False,
    compute_z_hist=False,
):
    """Solve A x = b with (sub)GMRES, monitoring residual subvector norms.

    Parameters
    ----------
    A : ndarray, sparse matrix, LinearOperator, or callable
        The (n, n) system matrix, applied forward. A callable/LinearOperator
        must compute A @ x.
    b : ndarray
        Right-hand side, shape (n,).
    ix : None, 1-D array of ints, or list of 1-D arrays of ints, optional
        Subvector partition. None (default): no subvectors, only the total
        residual is monitored. A 1-D array is interpreted as ascending
        0-based starting indices of contiguous subvectors. A list of arrays
        gives the subvector indices explicitly. If M is given, it must act
        on each subvector individually (block-diagonal w.r.t. ix).
    rtol, atol : None, scalar, or array, optional
        Relative/absolute stopping tolerances. See subgmres.m for the
        componentwise semantics (length 1, n_sub, or n_sub+1). Defaults:
        rtol=1e-6 (total residual only), atol=inf.
    maxiter : int, optional
        Maximum number of iterations. Default: min(n, 100).
    P : ndarray, sparse matrix, LinearOperator, or callable, optional
        Preconditioner. A raw matrix is solved against (P^{-1} x via direct
        factorization); a LinearOperator/callable must already compute the
        desired preconditioning action.
    M : ndarray, sparse matrix, LinearOperator, or callable, optional
        Inner product matrix inducing the norm used for the residual and
        the Arnoldi orthogonalization. A raw matrix is solved against
        (M^{-1} x); a LinearOperator/callable must already compute that
        action. None (default): the standard inner product.
    x0 : ndarray, optional
        Initial guess. Default: zero vector.
    compute_sub_norms_hist, compute_x_hist, compute_r_hist, compute_v_hist,
    compute_w_hist, compute_z_hist : bool, optional
        Attach the corresponding history to the result (see
        SubGMRESResult). compute_x_hist and compute_r_hist genuinely add
        work (an extra preconditioner application / matrix application per
        iteration); the others are free bookkeeping that is otherwise
        discarded.

        v_hist, w_hist and z_hist are the three bases of the Arnoldi
        process, in that order: the dual basis V (M-inverse-orthonormal),
        its primal companions W = M^{-1} V (M-orthonormal), and the primal
        Krylov basis Z = P^{-1} V (orthonormal w.r.t. P^H M^{-1} P). W is
        formed by the iteration in any case -- each column is reused in
        every subsequent orthogonalization -- whereas each column of Z is
        consumed immediately and is only kept when asked for.

    Returns
    -------
    SubGMRESResult
    """
    b = np.asarray(b).ravel()
    n = b.shape[0]

    A_op = coerce_operator(A, n, "A", mode="matvec")
    P_op = coerce_operator(P, n, "P", mode="solve")
    M_op = coerce_operator(M, n, "M", mode="solve")
    have_preconditioner = P_op is not None
    have_inner_product = M_op is not None

    sub_ix = _parse_subvectors(ix, n)
    n_sub = len(sub_ix)

    rtol_full = _parse_tolerance(rtol, n_sub, 1e-6, "rtol")
    atol_full = _parse_tolerance(atol, n_sub, np.inf, "atol")

    if maxiter is None:
        maxiter = round(min(n, 100))

    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.asarray(x0).ravel()
        if x0.shape != (n,):
            raise ValueError(f"x0 must be a vector of length {n}.")

    dtype = _infer_dtype(b, x0, A_op, P_op, M_op)

    need_sub_norms_hist = compute_sub_norms_hist
    need_x_hist = compute_x_hist
    need_r_hist = compute_r_hist
    need_z_hist = compute_z_hist

    ## Initialize the initial residual and the dual Krylov subspace basis
    ## (lines 1-2). V and W are preallocated to their maximum possible size
    ## to avoid the O(n*maxiter^2) cost of growing them one column at a
    ## time; they are truncated to the actual iteration count at the end.
    x = x0.astype(dtype, copy=True)
    r = (b - A_op.matvec(x)).astype(dtype, copy=False)
    V = np.zeros((n, maxiter + 1), dtype=dtype)
    V[:, 0] = r
    W = np.zeros((n, maxiter + 1), dtype=dtype)
    if have_inner_product:
        W[:, 0] = M_op.matvec(V[:, 0])
    else:
        W[:, 0] = V[:, 0]

    ## Compute the initial partial duality pairings (line 3). The pairing is
    ## <xi, x> = xi^T conj(x), antilinear in the primal argument, so the
    ## conjugate sits on W rather than on the companion. mu holds a raw pairing
    ## below, only guaranteed real once _assert_real has checked it, so it needs
    ## a complex-capable dtype or a genuinely complex value would be truncated.
    mu = np.zeros(n_sub, dtype=dtype)
    mu_scale = np.zeros(n_sub)
    for i, idx in enumerate(sub_ix):
        mu[i] = V[idx, 0] @ np.conj(W[idx, 0])
        mu_scale[i] = np.abs(V[idx, 0]) @ np.abs(W[idx, 0])
    mu = _assert_real(mu, "A squared M-inverse-norm of an initial residual subvector")
    mu = _assert_nonnegative(mu, mu_scale, "A squared M-inverse-norm of an initial residual subvector")

    ## Evaluate the M-inverse-norm of the initial residual and initialize g
    ## (lines 4-5). The squared norm is the sum of the (real, nonnegative)
    ## subvector pairings, so no separate full-vector pairing is needed.
    gamma2 = np.sum(mu)
    gamma = np.sqrt(gamma2)
    g = np.zeros(maxiter + 1, dtype=dtype)
    g[0] = gamma

    ## Normalize v and w, and rescale the partial pairings (lines 6-7). When
    ## gamma == 0 the initial residual is exactly zero; skip normalization
    ## to avoid 0/0, the convergence check below then fires immediately.
    if gamma > 0:
        V[:, 0] /= gamma
        W[:, 0] /= gamma
        mu = mu / gamma2

    ## Further initializations (lines 8-10).
    m = V[:, 0].copy()
    eta = np.concatenate([np.sqrt(mu) * gamma, [gamma]])
    eta0 = eta.copy()

    if need_x_hist:
        x_hist = np.zeros((n, maxiter + 1), dtype=dtype)
        x_hist[:, 0] = x
    if need_r_hist:
        r_hist = np.zeros((n, maxiter + 1), dtype=dtype)
        r_hist[:, 0] = r
    if need_sub_norms_hist:
        sub_norms_hist = np.zeros((n_sub + 1, maxiter + 1))
        sub_norms_hist[:, 0] = eta0
    if need_z_hist:
        Z = np.zeros((n, maxiter), dtype=dtype)

    n_iter = 0
    flag = None
    H = np.zeros((maxiter + 1, maxiter), dtype=dtype)
    RH = np.zeros((maxiter + 1, maxiter), dtype=dtype)
    c = np.zeros(maxiter, dtype=dtype)
    s = np.zeros(maxiter, dtype=dtype)
    theta = np.zeros(n_sub, dtype=dtype)

    ## Assemble the stopping tolerances (line 12). Fixed for the run, so
    ## formed once here rather than inside the loop.
    ##
    ## A subvector whose initial residual is exactly zero has nothing to
    ## reduce, so a relative criterion on it is vacuous. When its rtol is inf
    ## as well -- what a scalar rtol leaves behind, the caller having asked
    ## only for the total -- the product is inf*0 = nan. Map that to inf, no
    ## relative requirement, keeping any absolute one. numpy propagates nan
    ## through minimum where MATLAB's min drops it, so leaving the nan here
    ## made this implementation disagree with subgmres.m on the same input:
    ## `eta <= nan` is False forever, and the iteration ran until the Krylov
    ## space was exhausted and the triangular solve raised. The case is
    ## reached by the most ordinary of right hand sides -- a Stokes system
    ## with no source in the continuity equation starts with an exactly zero
    ## pressure residual.
    with np.errstate(invalid="ignore"):     # inf*0 is intended here
        relative_tol = rtol_full * eta0
    relative_tol[np.isnan(relative_tol)] = np.inf
    tol = np.minimum(relative_tol, atol_full)

    ## Main loop (lines 11-43). k is the 0-based column index of the vector
    ## being extended this iteration (k = MATLAB's `iter` - 1): this
    ## iteration reads V[:, k] and fills V[:, k+1] from it.
    while True:

        ## Check total residual M-inverse-norm for convergence (line 12).
        if np.all(eta <= tol):
            flag = 0
            break

        ## Check for maximum number of iterations reached.
        if n_iter >= maxiter:
            flag = 1
            break

        k = n_iter
        n_iter += 1

        ## Apply the preconditioner and the matrix (lines 13-14).
        vnew = V[:, k]
        if have_preconditioner:
            vnew = P_op.matvec(vnew)
        ## vnew is now z_{k+1} = P\v_{k+1}, the primal Krylov basis vector of
        ## line 13. It costs nothing extra, the matrix application of line 14
        ## needing it anyway, but it is consumed there and then, so it is kept
        ## only on request. Contrast W, whose columns are reused by every later
        ## orthogonalization and are therefore always stored.
        if need_z_hist:
            Z[:, k] = vnew
        vnew = A_op.matvec(vnew)

        ## Modified Gram-Schmidt orthogonalization (lines 15-18).
        for j in range(k + 1):
            H[j, k] = vnew @ np.conj(W[:, j])
            vnew = vnew - H[j, k] * V[:, j]

        ## Apply M-inverse (line 19).
        if have_inner_product:
            wnew = M_op.matvec(vnew)
        else:
            wnew = vnew

        ## Subvector norms of the new Arnoldi vector (line 20). psi needs a
        ## complex-capable dtype for the same reason mu does above: it holds
        ## a raw pairing before _assert_real has checked it.
        psi = np.zeros(n_sub, dtype=dtype)
        psi_scale = np.zeros(n_sub)
        for i, idx in enumerate(sub_ix):
            psi[i] = vnew[idx] @ np.conj(wnew[idx])
            psi_scale[i] = np.abs(vnew[idx]) @ np.abs(wnew[idx])
        psi = _assert_real(psi, "A squared M-inverse-norm of an Arnoldi subvector")
        psi = _assert_nonnegative(psi, psi_scale, "A squared M-inverse-norm of an Arnoldi subvector")

        ## Norm of the new Arnoldi vector (line 21).
        H[k + 1, k] = np.sqrt(np.sum(psi))
        RH[: k + 2, k] = H[: k + 2, k]

        ## Check for an invariant subspace: a (numerically) zero Arnoldi
        ## vector means the dual Krylov subspace is invariant under A P^-1,
        ## i.e. GMRES has converged.
        if abs(H[k + 1, k]) < 10 * np.finfo(float).eps:

            ## New Arnoldi vector is numerically zero (lines 22-25). V[:,
            ## k+1] and W[:, k+1] are already zero in the preallocated
            ## arrays.
            psi = np.zeros(n_sub)
            theta = np.zeros(n_sub)

            ## Apply previous Givens rotations (lines 26-30).
            for j in range(k):
                xi1 = RH[j, k]
                xi2 = RH[j + 1, k]
                RH[j, k] = c[j] * xi1 + s[j] * xi2
                RH[j + 1, k] = -np.conj(s[j]) * xi1 + np.conj(c[j]) * xi2

            ## New Givens rotation (lines 31-34).
            c[k] = 1
            s[k] = 0
            RH[k + 1, k] = 0

        else:

            ## Normalize the new Arnoldi vector (lines 22-25).
            psi = psi / np.sum(psi)
            V[:, k + 1] = vnew / H[k + 1, k]
            W[:, k + 1] = wnew / H[k + 1, k]
            for i, idx in enumerate(sub_ix):
                theta[i] = m[idx] @ np.conj(W[idx, k + 1])

            ## Apply previous Givens rotations (lines 26-30).
            for j in range(k):
                xi1 = RH[j, k]
                xi2 = RH[j + 1, k]
                RH[j, k] = c[j] * xi1 + s[j] * xi2
                RH[j + 1, k] = -np.conj(s[j]) * xi1 + np.conj(c[j]) * xi2

            ## New Givens rotation (lines 31-34).
            alpha = np.sqrt(np.abs(RH[k, k]) ** 2 + np.abs(RH[k + 1, k]) ** 2)
            sign_x1 = _sign(RH[k, k])
            c[k] = np.abs(RH[k, k]) / alpha
            s[k] = sign_x1 * np.conj(RH[k + 1, k]) / alpha
            RH[k, k] = sign_x1 * alpha
            RH[k + 1, k] = 0

        ## Update the (partial) residual norms (lines 35-41). g_iter holds
        ## g[k] as it stood before this iteration's update; both g[k] and
        ## g[k+1] below derive from it, so it must be read before either
        ## write.
        g_iter = g[k]
        g[k + 1] = -np.conj(s[k]) * g_iter
        g[k] = c[k] * g_iter
        m = -s[k] * m + c[k] * V[:, k + 1]
        for i, idx in enumerate(sub_ix):
            mu[i] = (
                np.abs(s[k]) ** 2 * mu[i]
                + np.abs(c[k]) ** 2 * psi[i]
                - 2 * np.real(s[k] * np.conj(c[k]) * theta[i])
            )
            eta[i] = np.abs(g[k + 1]) * np.sqrt(max(mu[i], 0))
        eta[-1] = np.abs(g[k + 1])

        ## Store the iteration history.
        if need_sub_norms_hist:
            sub_norms_hist[:, k + 1] = eta
        ## Reconstruct the current iterate (lines 42-43). This is needed
        ## whenever the iterate history OR the residual history is requested,
        ## since the residual r = b - A x below depends on the current x. Only
        ## the storing is gated on need_x_hist; requesting r_hist alone must
        ## not leave x at its initial value. subgmres.m keeps the same shape,
        ## the two having become independent there as well once the optional
        ## outputs were gated on isargout.
        if need_x_hist or need_r_hist:
            y = scipy.linalg.solve_triangular(RH[: k + 1, : k + 1], g[: k + 1], lower=False)
            update = V[:, : k + 1] @ y
            if have_preconditioner:
                update = P_op.matvec(update)
            x = x0 + update
            if need_x_hist:
                x_hist[:, k + 1] = x
        ## Evaluate and store the current residual. A genuine evaluation of
        ## b - A x, not the residual propagated alongside the Givens
        ## rotations: the Arnoldi relation makes the latter available for
        ## free, as g[k + 1] times the vector m updated above for the
        ## subvector norms, and its norm is already reported as the last row
        ## of sub_norms_hist. Taking r_hist from there would make it the
        ## recursion's own account of itself, where its whole use is to be the
        ## independent check. The two also drift apart in practice, agreeing
        ## only to about eps divided by the relative residual reached --
        ## worst in the regime a convergence study cares about.
        if need_r_hist:
            r = b - A_op.matvec(x)
            r_hist[:, k + 1] = r

    ## Truncate the preallocated arrays to the number of iterations
    ## actually performed.
    V = V[:, : n_iter + 1]
    H = H[: n_iter + 1, :n_iter]
    RH = RH[: n_iter + 1, :n_iter]
    g = g[: n_iter + 1]
    W = W[:, : n_iter + 1]
    if need_z_hist:
        Z = Z[:, :n_iter]
    if need_x_hist:
        x_hist = x_hist[:, : n_iter + 1]
    if need_r_hist:
        r_hist = r_hist[:, : n_iter + 1]
    if need_sub_norms_hist:
        sub_norms_hist = sub_norms_hist[:, : n_iter + 1]

    ## Reconstruct the final iterate and residual (lines 42-43), unless
    ## already up to date from the history bookkeeping above.
    if n_iter > 0 and not (need_x_hist or need_r_hist):
        y = scipy.linalg.solve_triangular(RH[:n_iter, :n_iter], g[:n_iter], lower=False)
        update = V[:, :n_iter] @ y
        if have_preconditioner:
            update = P_op.matvec(update)
        x = x0 + update
    if n_iter > 0 and not need_r_hist:
        r = b - A_op.matvec(x)

    return SubGMRESResult(
        x=x,
        flag=flag,
        iter=n_iter,
        sub_norms=eta,
        ix=sub_ix,
        tol_table=np.column_stack([rtol_full, atol_full]),
        sub_norms_hist=sub_norms_hist if need_sub_norms_hist else None,
        x_hist=x_hist if need_x_hist else None,
        r_hist=r_hist if need_r_hist else None,
        v_hist=V if compute_v_hist else None,
        w_hist=W if compute_w_hist else None,
        z_hist=Z if need_z_hist else None,
        h=H,
        rh=RH,
    )
