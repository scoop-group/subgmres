# -*- coding: utf-8 -*-
"""
How far the monitored residual norm can be trusted.

subgmres reports residual norms it never computes a residual for. They come
out of the Givens recursion, at no cost -- which is what makes subvector
monitoring cheap enough to be worth having -- and the last row of the history
is the norm of the residual the recursion carries along, not of b - A x.

In exact arithmetic the two agree. In floating point they drift apart, and
the drift is not random: the relative gap between them grows like the
reciprocal of the relative residual reached. The last column below is that
statement as an experiment -- the product of the two stays at roughly machine
epsilon over eleven orders of magnitude, while the gap itself climbs from
1e-16 to a factor of two.

The practical reading: a stopping tolerance of 1e-6 or 1e-8 is measured
faithfully, which is the range convergence criteria live in. Ask for 1e-14 and
the number reported is no longer the residual you have. Nothing here is a
defect -- it is the reason the solver also offers RHIST, which pays one
application of A per iteration to ask the system rather than the recursion.
"""

import numpy as np

from subgmres import subgmres

rng = np.random.default_rng(5)
n = 120
A = rng.standard_normal((n, n)) + 3 * n**0.5 * np.eye(n)
b = rng.standard_normal(n)

# Driven past any sensible stopping tolerance, to where the two disagree:
# rtol and atol are both zero, so only the iteration budget stops it.
result = subgmres(A, b, rtol=0.0, atol=0.0, maxiter=32,
                  compute_sub_norms_hist=True, compute_r_hist=True)
monitored = result.sub_norms_hist[-1, :]
true = np.linalg.norm(result.r_hist, axis=0)
relative = true / true[0]
gap = np.abs(monitored - true) / true

print(f"condition number of A: {np.linalg.cond(A):.2f},"
      f" machine epsilon: {np.finfo(float).eps:.1e}")
print(f"\n{'k':>4} {'monitored':>13} {'true':>13} {'rel.residual':>13}"
      f" {'gap':>10} {'gap x rel':>11}")
for k in range(0, result.iter + 1, 2):
	print(f"{k:>4} {monitored[k]:>13.4e} {true[k]:>13.4e} {relative[k]:>13.2e}"
	      f" {gap[k]:>10.1e} {gap[k] * relative[k]:>11.1e}")

# Where a stopping criterion would actually have fired, the monitored norm is
# exact to within a few ulp; it is only past that point that it stops meaning
# anything.
for tolerance in (1e-6, 1e-8, 1e-12):
	k = int(np.argmax(relative <= tolerance))
	print(f"\nstopping at rtol {tolerance:.0e} would fire at iteration {k},"
	      f" where the monitored norm is off by {gap[k]:.1e}")
