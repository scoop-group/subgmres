# -*- coding: utf-8 -*-
"""
Subvector residual monitoring -- the paper's namesake feature -- on a Stokes
saddle point discretized by finite differences.

The system is assembled here, in a dozen lines, on a staggered (MAC) grid:
velocities on cell faces, pressures at cell centres, giving [[A, B'], [B, 0]]
with A the vector Laplacian and B the discrete divergence. Nothing beyond
numpy and scipy is needed -- the drivers that reproduce the paper's actual
experiments use Firedrake, but a demo should not.

The demo shows three things:
  1. the residual subvectors converging under their own relative tolerances,
     in the inner product induced by M = blkdiag(A, Mp) -- the H1 seminorm on
     velocity, the L2 mass on pressure, which is the paper's Stokes choice;
  2. why one would bother: in the standard inner product the total residual
     norm meets the same tolerance while the pressure block is left far less
     converged, and monitoring the total alone cannot see it;
  3. that the subvectors need not be contiguous -- the same solve, with the
     unknowns interleaved and the partition passed as explicit index sets.

The pressure is left unpinned, so the system is singular: a constant pressure
is a nullvector. That is deliberate. The right hand side is consistent by
construction, and the iterates never leave the range of the operator, so the
computed pressure comes out with zero mean by itself -- the space the paper
works in, obtained for free rather than imposed.
"""

import numpy as np
import scipy.sparse as sp

from subgmres import subgmres


def second_difference(m, at_wall=False):
	# Dirichlet second difference on m unknowns. At a wall the boundary sits
	# half a cell from the first unknown rather than a whole one, which raises
	# the diagonal of the two end rows from 2 to 3.
	diagonal = 2.0 * np.ones(m)
	if at_wall:
		diagonal[0] = diagonal[-1] = 3.0
	return sp.diags([-np.ones(m - 1), diagonal, -np.ones(m - 1)], [-1, 0, 1])


def stokes(n):
	"""Stokes on the unit square, n x n cells. Returns (K, M, n_velocity,
	n_pressure) with M = blkdiag(A, Mp) serving as both inner product and
	preconditioner. These are pointwise difference operators, so the Schur
	complement is O(1) and the pressure mass matrix is the identity; the h^2 of
	a finite element mass matrix would be wrong here by exactly h^-2."""
	h = 1.0 / n
	I = sp.identity
	# u lives on the (n-1) x n vertical faces, v on the n x (n-1) horizontal
	# ones; each is walled in one direction and Dirichlet in the other.
	Lu = (sp.kron(I(n), second_difference(n - 1))
	      + sp.kron(second_difference(n, True), I(n - 1))) / h**2
	Lv = (sp.kron(I(n - 1), second_difference(n, True))
	      + sp.kron(second_difference(n - 1), I(n))) / h**2
	# Divergence: each cell differences the two faces bracketing it.
	ahead = sp.diags([np.ones(n - 1)], [0], shape=(n, n - 1))
	behind = sp.diags([-np.ones(n - 1)], [-1], shape=(n, n - 1))
	B = sp.hstack([sp.kron(I(n), ahead + behind),
	               sp.kron(ahead + behind, I(n))]).tocsr() / h
	A = sp.block_diag([Lu, Lv])
	K = sp.bmat([[A, B.T], [B, None]]).tocsc()
	M = sp.block_diag([A, I(B.shape[0])]).tocsc()
	return K, M, A.shape[0], B.shape[0]


K, M, n_velocity, n_pressure = stokes(16)
n = n_velocity + n_pressure

# A manufactured right hand side. Taking b = K x_exact makes it consistent by
# construction -- the alternative, a physical b with no source in the
# continuity equation, has an exactly zero pressure residual to begin with,
# against which a *relative* tolerance would be vacuous.
rng = np.random.default_rng(0)
x_exact = rng.standard_normal(n)
x_exact[n_velocity:] -= x_exact[n_velocity:].mean()
b = K @ x_exact

constant_pressure = np.concatenate([np.zeros(n_velocity), np.ones(n_pressure)])
print(f"Stokes, 16 x 16 cells: {n} unknowns "
      f"({n_velocity} velocity, {n_pressure} pressure)")
print(f"  a constant pressure is a nullvector: ||K z|| / ||z|| = "
      f"{np.linalg.norm(K @ constant_pressure) / np.linalg.norm(constant_pressure):.1e}")
print(f"  the right hand side is consistent:   |z.b| / ||b||   = "
      f"{abs(constant_pressure @ b) / np.linalg.norm(b):.1e}")

## 1. Monitoring the two blocks, each under its own relative tolerance.
# ix holds the starting index of each subvector; the tolerance vector carries
# one entry per subvector plus a final one for the total residual.
ix = [0, n_velocity]
rtol = [1e-9, 1e-7, 1e-8]
result = subgmres(K, b, ix=ix, rtol=rtol, P=M, M=M, compute_sub_norms_hist=True)
history = result.sub_norms_hist
print(f"\n1. proper inner product M = blkdiag(A, Mp)"
      f"   ->  {result.iter} iterations, flag {result.flag}")
print("   requested [rtol, atol] per subvector, then the total:")
for row, name in zip(result.tol_table, ("velocity", "pressure", "total   ")):
	print(f"      {name}  rtol {row[0]:.1e}   atol {row[1]:.1e}")
print(f"   {'iteration':>10} {'velocity':>12} {'pressure':>12} {'total':>12}")
for k in list(range(0, result.iter, max(1, result.iter // 5))) + [result.iter]:
	print(f"   {k:>10} {history[0, k]:>12.3e} {history[1, k]:>12.3e} "
	      f"{history[2, k]:>12.3e}")
pressure = result.x[n_velocity:]
print(f"   pressure mean, never imposed: {pressure.mean():.1e}")
print(f"   error against the exact solution: "
      f"{np.linalg.norm(result.x - x_exact) / np.linalg.norm(x_exact):.2e}")

## 2. Why monitor the blocks at all.
# Two solves stopped on the total residual alone, differing only in the inner
# product that measures it. Both meet the tolerance they were given. In the
# standard inner product the two blocks are nowhere near equally converged,
# and a total norm cannot reveal that; in the M-inverse norm, which weighs the
# blocks commensurately, they come out together. This is the same knob as
# demo_custom_inner_product, seen per block rather than in aggregate.
def relative_block_errors(x):
	residual = b - K @ x
	return [np.linalg.norm(residual[s]) / np.linalg.norm(b[s])
	        for s in (slice(0, n_velocity), slice(n_velocity, n))]

print(f"\n2. what a total norm hides. Same solve, same tolerance 1e-8 on the")
print(f"   total residual and nothing else, only the inner product differs:")
for name, inner_product in (("M-inverse norm", M), ("standard norm", None)):
	total_only = subgmres(K, b, rtol=1e-8, P=M, M=inner_product)
	velocity_error, pressure_error = relative_block_errors(total_only.x)
	print(f"      {name:16s} {total_only.iter:3d} iterations   "
	      f"velocity {velocity_error:.2e}   pressure {pressure_error:.2e}")

## 3. Subvectors need not be contiguous.
# Real assemblies often number unknowns by cell rather than by field, leaving
# each field scattered through the vector. Interleave the two blocks and pass
# the partition as explicit index sets instead of starting indices; nothing
# else changes, and neither does the answer.
order = np.argsort(np.concatenate([np.linspace(0, 1, n_velocity),
                                   np.linspace(0, 1, n_pressure)]), kind="stable")
scattered = [np.flatnonzero(order < n_velocity), np.flatnonzero(order >= n_velocity)]
permuted = subgmres(K[order][:, order], b[order], ix=scattered, rtol=rtol,
                    P=M[order][:, order], M=M[order][:, order])
print(f"\n3. same system, unknowns interleaved, partition as index sets"
      f"   ->  {permuted.iter} iterations, flag {permuted.flag}")
print(f"   velocity indices begin {scattered[0][:5]} ... (not contiguous)")
print(f"   subvector norms agree with part 1 to "
      f"{np.max(np.abs(permuted.sub_norms - result.sub_norms)):.1e}")
