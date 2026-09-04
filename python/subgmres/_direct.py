# -*- coding: utf-8 -*-
"""
Direct sparse-solve backend for the ``P \\ x`` / ``M \\ x`` actions.

The default backend is scipy's SuperLU (via ``scipy.sparse.linalg.factorized``),
which needs nothing beyond scipy. SuperLU is, however, single-threaded and
32-bit-indexed, and it fails on the largest 3-D saddle-point factorizations
("Can't expand MemType"). When the optional ``pypardiso`` package is installed
it is used instead for *real* sparse matrices: it wraps Intel MKL PARDISO, which
is 64-bit-indexed and multithreaded (thread count from ``MKL_NUM_THREADS``,
independent of Firedrake's ``OMP_NUM_THREADS=1``). This keeps the solver
backend-agnostic -- pypardiso is an auto-detected accelerator, never required.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg

# Optional accelerator: MKL PARDISO. Absence simply falls back to SuperLU.
try:
    from pypardiso import PyPardisoSolver
    _HAVE_PYPARDISO = True
except Exception:
    _HAVE_PYPARDISO = False


def have_parallel_direct_solver():
    """Whether the parallel MKL PARDISO backend (pypardiso) is available."""
    return _HAVE_PYPARDISO


def _pardiso_base_solve(matrix):
    # Factorize a real sparse matrix once with MKL PARDISO and return a callable
    # applying its inverse to a real (1-D or multi-column) right-hand side.
    # PARDISO wants a sorted CSR float64 matrix; keep it alive in the closure so
    # pypardiso reuses the factorization instead of refactorizing per solve.
    A = sp.csr_matrix(matrix)
    if A.dtype != np.float64:
        A = A.astype(np.float64)
    A.sort_indices()
    solver = PyPardisoSolver(mtype=11)   # real, structurally nonsymmetric (general)
    solver.factorize(A)

    def base_solve(b):
        return solver.solve(A, np.asarray(b, dtype=np.float64))

    return base_solve


def make_solve(matrix):
    """Return a callable ``solve(b)`` computing ``matrix^{-1} @ b`` from a single
    direct factorization, for dense or sparse ``matrix`` and 1-D or 2-D ``b``.

    A real sparse matrix uses MKL PARDISO when pypardiso is installed (parallel,
    64-bit -- required for the largest 3-D problems), otherwise scipy's SuperLU.
    Dense or complex matrices always use scipy. A complex right-hand side against
    a real factorization is handled by solving the real and imaginary parts
    separately, matching MATLAB's ``\\`` (e.g. a real inner-product matrix
    applied to the complex Helmholtz residual)."""
    if sp.issparse(matrix):
        matrix_is_real = not np.issubdtype(matrix.dtype, np.complexfloating)
        if _HAVE_PYPARDISO and matrix_is_real:
            base_solve = _pardiso_base_solve(matrix)
        else:
            base_solve = scipy.sparse.linalg.factorized(matrix.tocsc())
    else:
        matrix = np.asarray(matrix)
        matrix_is_real = not np.iscomplexobj(matrix)
        lu_piv = scipy.linalg.lu_factor(matrix)
        base_solve = lambda b: scipy.linalg.lu_solve(lu_piv, b)

    def solve(b):
        b = np.asarray(b)
        # Real factorization, complex right-hand side: solve the parts separately.
        if matrix_is_real and np.iscomplexobj(b):
            return base_solve(b.real) + 1j * base_solve(b.imag)
        return base_solve(b)

    return solve
