# -*- coding: utf-8 -*-
"""
Shared Firedrake helpers for the native (in-memory) matrix generation.

Replaces the legacy FEniCS-2019 export pipeline (assemble in FEniCS, write
PETSc binary, read into MATLAB). Here matrices are assembled with Firedrake
and converted straight to scipy in memory.

Note on FEniCS vs Firedrake: the UFL weak forms are essentially identical, but
the two apply Dirichlet boundary conditions differently. FEniCS's
assemble_system accumulates the constrained diagonal where boundary conditions
meet (corner dofs get 2 or 3), whereas Firedrake sets every constrained dof's
diagonal cleanly to 1. Both are valid discretizations; the resulting matrices
differ only at boundary dofs, which shifts discretization-dependent quantities
(iteration counts, equivalence constants) by a moderate amount. Results are
therefore regenerated from Firedrake rather than reproduced bit-for-bit.
"""

import numpy as np
import scipy.sparse as sp

from firedrake import (
	VectorFunctionSpace, FunctionSpace, TrialFunctions, TestFunctions,
	Constant, inner, grad, div, dot, dx, DirichletBC, assemble, Function, action,
)


def mat_to_scipy(assembled):
	"""Convert an assembled monolithic (mat_type='aij') Firedrake matrix to a
	scipy CSR matrix. The monolithic layout of a Taylor-Hood mixed space is
	field-contiguous ([all velocity dofs; all pressure dofs]), matching the
	block ordering the subvector monitoring expects."""
	petsc_mat = assembled.petscmat
	indptr, indices, data = petsc_mat.getValuesCSR()
	out = sp.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())
	out.eliminate_zeros()
	return out


def eliminate_dirichlet(A, dofs, diag):
	"""Symmetric Dirichlet elimination on a scipy matrix: zero the rows and
	columns of `dofs` and set their diagonal to `diag`. For homogeneous (u=0)
	conditions no right-hand-side lifting is needed. Used for the Helmholtz
	problem, whose partial Dirichlet boundary (and the choice to zero the
	Dirichlet diagonal of the mass/boundary-mass matrices) is applied at the
	scipy level for full control."""
	N = A.shape[0]
	keep = np.ones(N)
	keep[dofs] = 0.0
	D = sp.diags(keep)
	out = D @ A.tocsr() @ D            # zero rows and columns of the Dirichlet dofs
	if diag != 0.0:
		dvec = np.zeros(N)
		dvec[dofs] = diag
		out = out + sp.diags(dvec)    # set the Dirichlet diagonal
	return out.tocsr()


def vec_to_numpy(f):
	"""Flat numpy vector of a Firedrake Function/Cofunction in the same
	monolithic ordering as mat_to_scipy (verified consistent with the assembled
	operator)."""
	with f.dat.vec_ro as vec:
		return vec.array_r.copy()


def assemble_oseen_saddle_point(mesh, wind, driven_marker=None, noslip_markers=None,
                                driven_value=None, mu_val=1e-1, rho_val=1.0,
                                make_velocity_bcs=None):
	"""Assemble an Oseen saddle-point problem (works in 2D and 3D; the UFL forms
	are dimension-agnostic).

	Returns scipy matrices K (system), P (preconditioner), M (inner product),
	the numpy right-hand side b (with boundary lifting), the block dimensions
	n (velocity) and m (pressure), and x0, the discrete Dirichlet lifting to be
	used as the initial guess (see below). The forms mirror the FEniCS
	problems:

	  K = mu*(grad u : grad v) + rho*((grad u . wind) . v) - q div u - p div v
	  P = mu*(grad u : grad v) + rho*((grad u . wind) . v) - (1/mu) p q - p div v
	  M = mu*(grad u : grad v) + (1/mu) p q                (block-diagonal metric)

	WHY x0 IS RETURNED. Dirichlet dofs are not eliminated from the system; they
	are retained as trivial equations, as one does in practice, which deviates
	from the H^1_0 theory. Firedrake applies the conditions SYMMETRICALLY (rows
	AND columns), so those dofs are decoupled, with diagonal exactly 1 in K, P
	and M alike -- while every interior block scales with mu. Starting from the
	all-zero guess therefore leaves the boundary data sitting in the residual,
	weighted by 1 where the genuine unknowns are weighted by ~1/mu, and for
	small mu that fixed part dominates ||r||_{M^-1} and trips a relative
	stopping criterion early: Stokes with P = M then reports 49/45/35/29/19
	iterations for mu = 1e2 ... 1e-6, an apparent loss of parameter robustness
	that is not real.

	Starting instead from x0, which satisfies the boundary conditions, makes the
	residual of the trivial equations exactly zero -- bit-exact, since the
	diagonal is exactly 1 -- and the decoupling keeps it there for every
	iterate, so the monitored norms measure only the genuine V* x Q* residual.
	The counts above become 25 at every mu. For a problem with homogeneous
	Dirichlet data x0 is the zero vector and nothing changes.

	Boundary conditions on the velocity are either the driven-cavity default --
	no-slip on `noslip_markers` and velocity `driven_value` on `driven_marker`
	(the two-BC cavity case) -- or, when `make_velocity_bcs` is given, whatever
	that callable returns. `make_velocity_bcs(W)` receives the mixed space and
	returns a list of DirichletBC on W.sub(0); this supports problems with a
	spatially varying inflow profile and free (natural) outflow, such as the
	channel.
	"""
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	W = V * Q
	n, m = V.dim(), Q.dim()

	u, pr = TrialFunctions(W)
	v, qr = TestFunctions(W)
	mu, rho = Constant(mu_val), Constant(rho_val)
	# Velocity vector dimension (2 or 3) = number of components of the velocity
	# space, independent of the mesh dimension API.
	dim = V.value_size

	# Convection uses dot(grad(u), wind) = (wind . nabla) u, the FEniCS
	# grad(u)*wind term written explicitly.
	convection = rho * inner(dot(grad(u), wind), v) * dx
	diffusion = mu * inner(grad(u), grad(v)) * dx
	a_K = diffusion + convection - qr * div(u) * dx - pr * div(v) * dx
	a_P = diffusion + convection - (1.0 / mu) * inner(pr, qr) * dx - pr * div(v) * dx
	a_M = diffusion + (1.0 / mu) * inner(pr, qr) * dx
	L = inner(Constant((0.0,) * dim), v) * dx

	# Either use the caller's BC builder, or the default driven/no-slip pair.
	if make_velocity_bcs is not None:
		bcs = make_velocity_bcs(W)
	else:
		bcs = [DirichletBC(W.sub(0), Constant((0.0,) * dim), noslip_markers),
		       DirichletBC(W.sub(0), Constant(driven_value), (driven_marker,))]

	K = mat_to_scipy(assemble(a_K, bcs=bcs, mat_type="aij"))
	P = mat_to_scipy(assemble(a_P, bcs=bcs, mat_type="aij"))
	M = mat_to_scipy(assemble(a_M, bcs=bcs, mat_type="aij"))

	# Right-hand side with boundary lifting. g holds the boundary values; the
	# eliminated correction rhs (constrained entries homogenized to 0) plus g
	# gives the full-solution rhs, since K is symmetrically eliminated.
	g = Function(W)
	for bc in bcs:
		bc.apply(g)
	x0 = vec_to_numpy(g)
	b = vec_to_numpy(assemble(L - action(a_K, g), bcs=bcs)) + x0
	# g is zero away from the constrained dofs and equals b on them, so
	# K @ x0 agrees with b there and the trivial equations start solved.
	return K, b, P, M, n, m, x0
