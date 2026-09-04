# -*- coding: utf-8 -*-
"""
Coercion of user-supplied A, P, M arguments into scipy LinearOperators.

This mirrors the branching in subgmres.m: a plain numeric array is treated
as itself, while a callable (function handle in the MATLAB code) is trusted
to already implement whatever action is needed. The one subtlety carried
over from subgmres.m is that A is applied forward (A @ x), whereas a numeric
P or M is *solved against* (P \\ x, M \\ x) rather than multiplied -- P and M
play the role of an already-inverted operator (a preconditioner action, or
the Riesz map induced by an inner product), matching the convention scipy
itself uses for the M= preconditioner argument of gmres/cg.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from ._direct import make_solve


def _is_array_like(op):
    # ndarray and any scipy.sparse matrix/array both count as "a matrix the
    # user handed us", as opposed to a LinearOperator or a bare callable.
    return isinstance(op, np.ndarray) or sp.issparse(op)


def _factorized_solve_operator(matrix, n, name):
    # Build a LinearOperator representing matrix^{-1} via a single direct
    # factorization, matching subgmres.m's `P\x` / `M\x` semantics for
    # matrix-valued P and M. The factorization backend (MKL PARDISO for large
    # real sparse systems, else scipy) and the complex-right-hand-side handling
    # both live in subgmres._direct.make_solve.
    solve = make_solve(matrix)
    # Leave dtype unset (None): the output dtype follows the right-hand side,
    # so a real operator can still return a complex result (Helmholtz) without
    # scipy's LinearOperator refusing the cast.
    return LinearOperator((n, n), matvec=solve)


def coerce_operator(op, n, name, mode):
    """Coerce `op` into a LinearOperator of shape (n, n), or return None.

    `mode='matvec'` (used for A): a raw ndarray/sparse matrix is applied
    directly, i.e. the resulting operator computes A @ x.

    `mode='solve'` (used for P and M): a raw ndarray/sparse matrix is
    factorized once and solved against on each application, i.e. the
    resulting operator computes A^{-1} @ x -- P and M are supplied as the
    forward operator, but subgmres needs their inverse action.

    A LinearOperator or a plain callable is always trusted to already
    compute the desired action (the forward matvec for A, the solve/apply
    action for P or M) and is used as-is regardless of `mode`.
    """
    if op is None:
        return None
    if isinstance(op, LinearOperator):
        result = op
    elif _is_array_like(op):
        if mode == "matvec":
            result = aslinearoperator(op)
        elif mode == "solve":
            result = _factorized_solve_operator(op, n, name)
        else:
            raise ValueError(f"Unknown mode {mode!r}.")
    elif callable(op):
        result = LinearOperator((n, n), matvec=op)
    else:
        raise TypeError(f"{name} must be an array, a LinearOperator, or a callable, got {type(op)!r}.")
    if result.shape != (n, n):
        raise ValueError(f"{name} must act on vectors of length {n}, but has shape {result.shape}.")
    return result
