# -*- coding: utf-8 -*-
"""
A complex-valued Helmholtz problem, solved with a preconditioner.

Two things at once, because they belong together. A preconditioner P is passed
as one more argument, and subgmres uses its inverse action -- given a matrix it
factorizes and solves against it for you (demo_operator_forms covers the other
form, where you supply the solve yourself). And everything works over the
complex field: A, b, the Krylov basis and the iterates are complex here, while
the residual norms the solver reports remain real, as norms must.

The system is a one-dimensional Helmholtz problem on (0, 1) -- u'' - k^2 u = f,
Dirichlet at the left end and a first-order Sommerfeld radiation condition at
the right, imposed through a ghost point so that the last row stays second
order. This is the standard hard case for Krylov methods: the operator is
indefinite, and unpreconditioned GMRES gets nowhere within any sensible budget.
The preconditioner is the complex-shifted Laplacian, the same discretization
with k^2 replaced by (1 + i/2) k^2, which is the textbook remedy.
"""

import numpy as np
import scipy.sparse as sp

from subgmres import subgmres


def helmholtz(n, k, shift=0.0):
	"""The Helmholtz operator above, or its complex-shifted variant."""
	h = 1.0 / n
	wavenumber = (1 + 1j * shift) * k**2
	A = sp.diags([-np.ones(n - 1) / h**2,
	              2.0 / h**2 - wavenumber * np.ones(n, dtype=complex),
	              -np.ones(n - 1) / h**2], [-1, 0, 1], dtype=complex).tolil()
	# Sommerfeld at x = 1, via the ghost value u[n] = u[n-2] + 2ikh u[n-1].
	A[n - 1, n - 2] = -2.0 / h**2
	A[n - 1, n - 1] = 2.0 / h**2 - wavenumber - 2j * k / h
	return A.tocsc()


n = 400
k = 40.0
A = helmholtz(n, k)
b = np.zeros(n, dtype=complex)
b[n // 4] = float(n)                    # a point source a quarter of the way in
relative_residual = lambda x: np.linalg.norm(b - A @ x) / np.linalg.norm(b)
print(f"1-D Helmholtz: {n} unknowns, k = {k:.0f},"
      f" {2 * np.pi * n / k:.0f} points per wavelength")
print(f"A is complex: {np.iscomplexobj(A.data)}")

# Unpreconditioned, with a budget a quarter the size of the problem.
plain = subgmres(A, b, rtol=1e-8, maxiter=100)
print(f"\n   no preconditioner:   flag {plain.flag}, {plain.iter} iterations,"
      f" ||r||/||b|| = {relative_residual(plain.x):.2e}")

# The same solve with the shifted Laplacian passed as P. Nothing else changes.
P = helmholtz(n, k, shift=0.5)
preconditioned = subgmres(A, b, rtol=1e-8, maxiter=100, P=P)
print(f"   shifted Laplacian:   flag {preconditioned.flag},"
      f" {preconditioned.iter} iterations,"
      f" ||r||/||b|| = {relative_residual(preconditioned.x):.2e}")

# The solution is genuinely complex -- a travelling wave, not a standing one --
# and the reported residual norms are real regardless.
x = preconditioned.x
print(f"\n   solution is complex:   max |Re| = {np.abs(x.real).max():.3e},"
      f"  max |Im| = {np.abs(x.imag).max():.3e}")
print(f"   reported norms are real: {np.isrealobj(preconditioned.sub_norms)},"
      f" final total {preconditioned.sub_norms[-1]:.3e}")
