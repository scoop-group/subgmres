# -*- coding: utf-8 -*-
"""
Restarted GMRES.

subgmres does not implement restarting, and deliberately so: because canonical
GMRES is formulated for an arbitrary initial guess, a restarted method needs no
machinery of its own. One re-invokes the solver every m steps with x0 set to
the iterate the previous cycle produced. That loop is the whole feature, and it
is written out below in eight lines.

Restarting caps what the method has to store and orthogonalize against. Full
GMRES holds the entire Krylov basis and orthogonalizes each new vector against
all of it, so both grow without bound; GMRES(m) never holds more than m
vectors. The price is paid in iterations, and if m is too small it is paid in
convergence itself -- the last part below shows a restart length that never
gets there at all.

The system is a convection-diffusion operator on (-1,1)^2, discretized by
centred finite differences, with the recirculating wind of the paper's Oseen
problems. Nothing beyond numpy and scipy is needed.
"""

import numpy as np
import scipy.sparse as sp

from subgmres import subgmres


def convection_diffusion(n, epsilon):
	"""-epsilon*Laplace(u) + w.grad(u) on (-1,1)^2, n x n interior points.
	Returns (A, P, mesh_peclet) with P the diffusion part alone, a textbook
	preconditioner for this operator."""
	h = 2.0 / (n + 1)
	axis = -1.0 + h * np.arange(1, n + 1)
	X, Y = np.meshgrid(axis, axis, indexing="ij")
	# The recirculating wind of the paper's Oseen cavity, divergence free and
	# tangential to the boundary.
	wind_x = (2 * Y * (1 - X**2)).ravel(order="F")
	wind_y = (-2 * X * (1 - Y**2)).ravel(order="F")
	I = sp.identity(n)
	second = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)],
	                  [-1, 0, 1]) / h**2
	first = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [-1, 1]) / (2 * h)
	laplacian = sp.kron(I, second) + sp.kron(second, I)
	A = (epsilon * laplacian
	     + sp.diags(wind_x) @ sp.kron(I, first)
	     + sp.diags(wind_y) @ sp.kron(first, I))
	wind_max = np.max(np.hypot(wind_x, wind_y))
	return A.tocsc(), (epsilon * laplacian).tocsc(), wind_max * h / (2 * epsilon)


def restarted_gmres(A, b, restart_length, target, max_cycles=200, **kwargs):
	"""GMRES(restart_length): re-invoke the solver from where it left off.

	The stopping criterion is absolute. It has to be: each cycle measures its
	own relative tolerance against its own starting residual, so a relative
	criterion would ask for a further reduction by that factor every cycle
	rather than the one reduction that was wanted overall. Passing rtol = inf
	and atol = target states the global goal once."""
	x = np.zeros(A.shape[0])
	iterations = 0
	for cycle in range(1, max_cycles + 1):
		result = subgmres(A, b, rtol=np.inf, atol=target,
		                  maxiter=restart_length, x0=x, **kwargs)
		x = result.x
		iterations += result.iter
		if result.flag == 0:
			return x, iterations, cycle, True
	return x, iterations, max_cycles, False


n = 24
A, P, mesh_peclet = convection_diffusion(n, epsilon=1 / 20)
b = np.ones(A.shape[0])
target = 1e-8 * np.linalg.norm(b)
relative_residual = lambda x: np.linalg.norm(b - A @ x) / np.linalg.norm(b)
print(f"convection-diffusion on {n} x {n} interior points: {A.shape[0]} unknowns")
print(f"mesh Peclet number {mesh_peclet:.1f}")

## 1. What restarting costs, and what it buys.
# The Krylov dimension held at once is the memory: full GMRES ends up holding
# one vector per iteration, GMRES(m) never more than m. (subgmres stores both
# the dual basis V and its primal companions W, so the vectors in flight are
# twice the figure below in each case.)
full = subgmres(A, b, rtol=np.inf, atol=target, maxiter=A.shape[0])
print(f"\n1. no preconditioner"
      f"\n   {'method':>12} {'iterations':>12} {'cycles':>8} {'basis held':>12}"
      f" {'||r||/||b||':>13}")
print(f"   {'full GMRES':>12} {full.iter:>12} {'-':>8} {full.iter + 1:>12}"
      f" {relative_residual(full.x):>13.1e}")
for restart_length in (40, 20, 10):
	x, iterations, cycles, converged = restarted_gmres(A, b, restart_length, target)
	print(f"   {'GMRES(%d)' % restart_length:>12} {iterations:>12} {cycles:>8}"
	      f" {restart_length + 1:>12} {relative_residual(x):>13.1e}")

## 2. The preconditioner and inner product carry over unchanged.
# Nothing in the loop above knows about P: it is passed straight through to
# each cycle, exactly as an unrestarted call would pass it. An M would go the
# same way. With a preconditioner that actually works, the iteration counts
# collapse and restarting stops costing anything much -- which is the usual
# reason one can afford a short restart length in practice.
full_p = subgmres(A, b, rtol=np.inf, atol=target, maxiter=A.shape[0], P=P)
print(f"\n2. same runs, preconditioned by the diffusion part alone")
print(f"   {'full GMRES':>12} {full_p.iter:>12} {'-':>8} {full_p.iter + 1:>12}"
      f" {relative_residual(full_p.x):>13.1e}")
for restart_length in (20, 10):
	x, iterations, cycles, converged = restarted_gmres(A, b, restart_length,
	                                                   target, P=P)
	print(f"   {'GMRES(%d)' % restart_length:>12} {iterations:>12} {cycles:>8}"
	      f" {restart_length + 1:>12} {relative_residual(x):>13.1e}")

## 3. Restarting can also fail.
# Halve the diffusion and the same GMRES(10) no longer converges at all. The
# information a cycle would have needed is exactly what the restart discarded,
# and no number of cycles recovers it -- full GMRES on this operator still
# reaches the target.
A_hard, _, peclet_hard = convection_diffusion(n, epsilon=1 / 40)
b_hard = np.ones(A_hard.shape[0])
target_hard = 1e-8 * np.linalg.norm(b_hard)
hard_full = subgmres(A_hard, b_hard, rtol=np.inf, atol=target_hard,
                     maxiter=A_hard.shape[0])
x, iterations, cycles, converged = restarted_gmres(
	A_hard, b_hard, 10, target_hard, max_cycles=50)
print(f"\n3. the same operator at mesh Peclet {peclet_hard:.1f}, where a short"
      f" restart is not enough")
print(f"   full GMRES  converged in {hard_full.iter} iterations")
print(f"   GMRES(10)   {'converged' if converged else 'did NOT converge'} after"
      f" {cycles} cycles, {iterations} iterations,"
      f" stuck at ||r||/||b|| = {np.linalg.norm(b_hard - A_hard @ x) / np.linalg.norm(b_hard):.1e}")
