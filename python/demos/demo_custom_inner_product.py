# -*- coding: utf-8 -*-
"""
Canonical GMRES in a non-standard inner product -- the paper's central point:
the norm in which the residual is minimized need not be the standard one, and
choosing it is separate from choosing a preconditioner.

Passing M changes which residual is made smallest at every step, and so the
whole path taken to the solution, without changing the system A x = b at all.
The two runs below differ in exactly one argument. What they produce is not
"better" and "worse" convergence but convergence measured in two different
currencies: each run is optimal in its own norm at every iteration and is
beaten by the other in the other norm. The table shows both histories in both
norms, which is the only fair way to look at them.
"""

import numpy as np

from subgmres import subgmres

rng = np.random.default_rng(1)
n = 100
A = rng.standard_normal((n, n)) + 3 * n**0.5 * np.eye(n)
b = rng.standard_normal(n)

# A diagonal weighting spanning six orders of magnitude: an extreme choice, so
# that the divergence between the two histories is unmistakable.
weights = np.geomspace(1e-3, 1e3, n)
M = np.diag(weights)

# Both runs ask for the residual history. The norms each run monitors for
# itself are in its own norm alone, and the comparison needs both residuals in
# both norms, so the residual vectors are what is wanted here.
standard = subgmres(A, b, rtol=1e-10, maxiter=n, compute_r_hist=True)
weighted = subgmres(A, b, rtol=1e-10, maxiter=n, M=M, compute_r_hist=True)
print(f"standard inner product: {standard.iter} iterations, flag {standard.flag}")
print(f"M-weighted:             {weighted.iter} iterations, flag {weighted.flag}")

standard_norm = lambda R: np.linalg.norm(R, axis=0)
m_inverse_norm = lambda R: np.sqrt(np.einsum("ij,ij->j", R, R / weights[:, None]))

# Compare over the iterations both runs reached.
common = min(standard.iter, weighted.iter) + 1
histories = {
	"standard": (standard_norm(standard.r_hist[:, :common]),
	             m_inverse_norm(standard.r_hist[:, :common])),
	"weighted": (standard_norm(weighted.r_hist[:, :common]),
	             m_inverse_norm(weighted.r_hist[:, :common])),
}

print(f"\n{'':>5} | {'measured in the standard norm':>29} |"
      f" {'measured in the M-inverse norm':>30}")
print(f"{'k':>5} | {'standard run':>14} {'M-weighted run':>14} |"
      f" {'standard run':>14} {'M-weighted run':>15}")
for k in list(range(0, common - 1, max(1, common // 7))) + [common - 1]:
	print(f"{k:>5} | {histories['standard'][0][k]:>14.3e}"
	      f" {histories['weighted'][0][k]:>14.3e} |"
	      f" {histories['standard'][1][k]:>14.3e}"
	      f" {histories['weighted'][1][k]:>15.3e}")

# Neither run is uniformly better. Each is the smaller one in the norm it was
# asked to minimize, at every single iteration -- which is what "minimizing the
# residual" means once one has said in which norm.
standard_wins = np.all(histories["standard"][0] <= histories["weighted"][0])
weighted_wins = np.all(histories["weighted"][1] <= histories["standard"][1])
print(f"\nstandard run never beaten in the standard norm:   {standard_wins}")
print(f"M-weighted run never beaten in the M-inverse norm: {weighted_wins}")

# Both solve the same system, so both end with a small true residual; the
# choice of norm decides the route, not the destination.
for name, x in (("standard  ", standard.x), ("M-weighted", weighted.x)):
	print(f"  [{name}] ||b - A x|| / ||b|| = "
	      f"{np.linalg.norm(b - A @ x) / np.linalg.norm(b):.3e}")
