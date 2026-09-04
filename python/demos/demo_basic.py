# -*- coding: utf-8 -*-
"""
The simplest call, and how to read what comes back.

subgmres(A, b) with nothing else solves in the standard norm, without a
preconditioner and without partitioning the residual -- the GMRES most users
reach for first. This demo does that, then looks at the three things worth
understanding before using anything fancier: what the defaults actually were,
how to tell convergence from exhaustion, and what the returned iterate is
worth when the iteration stopped short.
"""

import numpy as np

from subgmres import subgmres

rng = np.random.default_rng(0)
n = 200
A = rng.standard_normal((n, n)) + 3 * n**0.5 * np.eye(n)   # well conditioned
b = rng.standard_normal(n)
relative_residual = lambda x: np.linalg.norm(b - A @ x) / np.linalg.norm(b)

## 1. The simplest call there is.
result = subgmres(A, b)
print("1. subgmres(A, b), everything left at its default")
print(f"   flag {result.flag}, {result.iter} iterations,"
      f" ||b - A x|| / ||b|| = {relative_residual(result.x):.2e}")

## 2. What the defaults were.
# tol_table reports the tolerances that were actually in force, which is more
# useful than remembering them: one row per subvector, then one for the total
# residual, with the relative tolerance first and the absolute second. With no
# partition requested there is a single subvector -- the whole vector -- so
# there are two rows, and only the total carries the default rtol of 1e-6.
# maxiter defaults to min(n, 100), which is 100 here, not 200.
print("\n2. the tolerances that were in force")
for row, name in zip(result.tol_table, ("whole vector", "total       ")):
	print(f"   {name}  rtol {row[0]:.1e}   atol {row[1]:.1e}")

## 3. Convergence, and stopping short.
# flag is the one output to check before using x. 0 means the tolerances were
# met; 1 means the iteration budget ran out first, and then x is simply the
# best iterate found so far -- a perfectly good vector, just not yet accurate.
# It is not a failure code and x is not garbage: the residual below is the
# honest measure of what was achieved, and picking up where it left off is a
# matter of passing that x back as x0 (see demo_restarts).
print("\n3. what happens when the budget runs out")
short = subgmres(A, b, rtol=1e-14, maxiter=8)
print(f"   rtol 1e-14 in at most 8 iterations: flag {short.flag},"
      f" {short.iter} iterations, ||r||/||b|| = {relative_residual(short.x):.2e}")
resumed = subgmres(A, b, rtol=1e-14, maxiter=8, x0=short.x)
print(f"   eight more from there:              flag {resumed.flag},"
      f" {resumed.iter} iterations, ||r||/||b|| = {relative_residual(resumed.x):.2e}")
finished = subgmres(A, b, rtol=1e-14, maxiter=n)
print(f"   or simply a larger budget:          flag {finished.flag},"
      f" {finished.iter} iterations, ||r||/||b|| = {relative_residual(finished.x):.2e}")
