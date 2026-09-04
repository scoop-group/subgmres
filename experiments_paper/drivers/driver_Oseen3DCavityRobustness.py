# -*- coding: utf-8 -*-
"""
Driver: Oseen 3D Cavity parameter-robustness study (native Firedrake pipeline).

Sweeps the two constants of the Oseen 3D cavity -- the viscosity ``mu`` and the
density ``rho`` -- over a range of mesh levels and records, at each (level, mu,
rho) grid point, the number of subGMRES iterations to convergence under two
inner products:

  * the proper M-inverse inner product (M = mu*diffusion + (1/mu) pressure mass),
  * the standard inner product (M = I).

What the experiment shows is *robustness*, not a head-to-head of counts. The
system depends on (mu, rho) only through gamma = rho/mu, and under the proper
M-inverse norm that dependence collapses to an overall scalar sqrt(mu) on the
residual norm, which the relative stopping criterion cancels: the proper count
is then constant along gamma = const, i.e. along the anti-diagonals of the
table. The standard norm weights the velocity and pressure blocks differently,
||r||^2 = mu^2 ||r_V||^2 + ||r_Q||^2, so no relative criterion can cancel it and
the standard count drifts. Since A and P are identical in the two solves, the
inner product is the only possible cause.

The counts themselves are NOT the comparison: the two solves apply a relative
criterion in two different norms, so reducing ||r||_{M^-1} by 1e-6 is a
different demand from reducing ||r||_2 by 1e-6 -- one meaningful to the problem,
the other not.

Efficiency note (from the structure of the forms): the inner-product matrix M
depends only on mu, so its expensive direct factorization is formed once per mu
and reused across the whole rho sweep. The preconditioner P depends on both
constants and is refactorized at every grid point.

Run inside the Firedrake virtual environment, e.g. locally via Docker:

  docker run --rm -v "$PWD":/work -w /work/src/python \\
    --user "$(id -u):$(id -g)" -e HOME=/tmp firedrakeproject/firedrake:latest \\
    python3 drivers/driver_Oseen3DCavityRobustness.py --levels 1 2
"""

import argparse
import os
import sys

# Make generation/ (oseen3d_cavity, firedrake_utils) importable; importing
# driver_common also puts src/python on the path for the subgmres module.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "generation"))

import driver_common as dc
import oseen3d_cavity
from subgmres import subgmres

from scipy.sparse.linalg import LinearOperator

# Parameter grid, log-spaced, so that gamma = rho/mu covers 1e-4 ... 1e2 with no
# cell to exclude. The range is capped at gamma = 1e2 for two independent
# reasons, which happen to bite at the same place.
#
# Physically: gamma is a Reynolds number, and gamma = 1e3 sits at a mesh Peclet
# number near 70 on level 3, far outside what this discretization resolves.
#
# Practically: at gamma = 1e3 the level-3 solves ran past 500 iterations without
# converging and were heading for a cap of 2000, at a cost quadratic in the
# iteration count for the orthogonalization alone.
#
# NOT for loss of orthogonality, which was the original motivation and does not
# survive measurement. At level 2, ||V^H M^-1 V - I||_2 came out 2.7e-10 after
# 42 iterations (gamma = 10) and 4.0e-10 after 191 (gamma = 1e2) -- essentially
# flat in the iteration count, and the monitored subvector norms tracked the
# recomputed ones to 8e-5 and 2e-7 respectively. This is what MGS-GMRES theory
# predicts (Paige, Rozloznik and Strakos, SIMAX 2006): the loss of orthogonality
# grows like the inverse of the relative residual achieved, not like the
# iteration count, and reaches O(1) only once the residual stagnates at the
# backward-stable level. Stopping at rtol = 1e-6 bounds it near eps/1e-6 ~ 1e-10,
# which is what is observed. A longer non-converged run would lose LESS, not
# more.
#
# The convection-dominated corner that remains (small mu, large rho) is still
# where the Stokes Schur approximation (1/mu) Mp in P is most stressed.
#
# The system depends on (mu, rho) only through gamma, up to the congruence
# A(mu,rho) = D1 Atilde(gamma) D2 with D1 = blkdiag(mu I, I), D2 = blkdiag(I,
# (1/mu) I); the grid is nevertheless run in 2D so that the collapse onto the
# anti-diagonals is something the table shows rather than assumes.
MU_GRID = [1.0, 1e-1, 1e-2, 1e-3]
RHO_GRID = [1e-4, 1e-3, 1e-2, 1e-1]


def write_robustness(filename, rows):
	"""Write the long-format robustness CSV: one line per (level, mu, rho) with
	the two iteration counts. A count equal to the maxiter cap means that solve
	did NOT converge -- the per-cell convergence flags are printed to stdout
	during the run (kept out of the CSV, which is meant to feed the manuscript
	tables), together with a convergence flag for each and the equivalence
	constants on the final dual Krylov subspace.

	The flags are in the CSV rather than only on stdout because a capped count
	is otherwise indistinguishable from a genuine one: the first sweep produced
	three cells reading exactly 500, which was the cap, and nothing in the file
	said so. The manuscript table has to mark those cells, so the information
	has to survive into it.

	The manuscript table is one mu x rho grid per inner product, each cell
	carrying the triple over the levels, so the level is a column here and the
	table is pivoted at typesetting time."""
	with open(filename, "w") as f:
		f.write("level,mu,rho,iter_proper,converged_proper,"
			"iter_standard,converged_standard,c,C\n")
		for level, mu, rho, ip, cp, isd, cs, c, C in rows:
			f.write(f"{level},{mu:.6e},{rho:.6e},{ip},{int(cp)},{isd},{int(cs)},"
				f"{c:.6e},{C:.6e}\n")


def run(levels, rtol, maxiter, results_dir, verbose=True):
	"""Sweep the (mu, rho) grid at each mesh level and write the CSV."""
	os.makedirs(results_dir, exist_ok=True)
	rows = []
	for level in levels:
		# Outer loop over mu so the mu-only metric M is factorized once per mu.
		for mu in MU_GRID:
			metric_op = None
			for rho in RHO_GRID:
				# Assemble the cavity at this (mu, rho): A = K system matrix,
				# r = rhs, P = preconditioner, M = inner-product metric, with
				# velocity/pressure block sizes n, m, and x0 the Dirichlet lifting.
				A, r, P, M, n, m, x0 = oseen3d_cavity.assemble_problem(
					level, mu=mu, rho=rho)
				# Subvector partition: velocity block [0:n], pressure block [n:],
				# so the stopping criterion tracks both residual subvectors.
				ix = [0, n]
				# M depends on mu alone, so it is identical across rho -> factorize
				# it once per mu and reuse the M-inverse action for every rho.
				if metric_op is None:
					metric_op = LinearOperator(A.shape, matvec=dc.make_solve(M))
				# P depends on both constants -> refactorize at every grid point.
				precond_op = LinearOperator(A.shape, matvec=dc.make_solve(P))
				# Proper solve (M-inverse inner product) and standard solve (M = I),
				# sharing the one P factorization. Only the iteration counts are
				# needed, so no history arrays are requested.
				#
				# x0 satisfies the Dirichlet conditions, so the retained trivial
				# equations contribute nothing to the monitored residual norms --
				# without it the counts drift with mu for that reason alone, which
				# is what made the first version of this study unreadable.
				# compute_v_hist is free -- the Arnoldi basis is formed anyway and
				# would otherwise be discarded -- and yields the equivalence
				# constants between the M-inverse and the standard norm on the
				# dual Krylov subspace. Their ratio is the factor by which the
				# standard norm can misjudge the meaningful one, and it grows
				# steeply as the viscosity leaves the value balancing the two
				# blocks of M: at level 2 it runs from 4.5 at mu = 1e-1 to 194
				# at mu = 1e-3. That is the same degradation the manuscript
				# shows under mesh refinement, on a second axis.
				res_proper = subgmres(A, r, ix=ix, rtol=rtol, maxiter=maxiter,
					P=precond_op, M=metric_op, x0=x0, compute_v_hist=True)
				c_vals, C_vals = dc.equivalence_constants(res_proper.v_hist)
				res_standard = subgmres(A, r, ix=ix, rtol=rtol, maxiter=maxiter,
					P=precond_op, M=None, x0=x0)
				# flag: 0 = converged, 1 = maxiter reached.
				if verbose:
					print(f"[Oseen3DCavityRobustness] level {level}  "
						f"mu={mu:.1e} rho={rho:.1e} gamma={rho / mu:.1e}: "
						f"proper {res_proper.iter} it (converged={res_proper.flag == 0}), "
						f"standard {res_standard.iter} it (converged={res_standard.flag == 0}), "
						f"C/c = {C_vals[-1] / c_vals[-1]:.3e}",
						flush=True)
				rows.append((level, mu, rho,
					res_proper.iter, res_proper.flag == 0,
					res_standard.iter, res_standard.flag == 0,
					c_vals[-1], C_vals[-1]))
	out = os.path.join(results_dir, "Oseen3DCavityRobustness.csv")
	write_robustness(out, rows)
	# Also write the pivoted equivalence table, so the manuscript can read it at
	# compile time instead of carrying transcribed numbers.
	piv = dc.write_robustness_equivalence(
		out, os.path.join(results_dir, "Oseen3DCavityRobustness_equivalence.csv"))
	if verbose:
		print(f"Wrote {out}")
		print(f"Wrote {piv}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Oseen 3D Cavity parameter-robustness study (Firedrake).")
	# Levels 1-3. Level 4 is out on cost: one (mu, rho) point there runs about
	# 17 minutes (184 s assembly + 207 s factorization + 650 s of solves, per the
	# committed timings), so a 16-point sweep would take some 4.6 hours against
	# roughly 20 minutes for levels 1-3 together. --levels rather than --level
	# also matches the other drivers, so run_experiments_Heidelberg.sh can call
	# this one through the same loop.
	parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
		help="Mesh levels to run. Default: %(default)s.")
	parser.add_argument("--results-dir", default=dc.DEFAULT_RESULTS_DIR,
		help="Directory for the output CSV. Default: ../results.")
	# dc.COMMAND_LINE, not sys.argv: driver_common took the arguments out of
	# PETSc's way when it was imported. See the note there.
	args = parser.parse_args(dc.COMMAND_LINE)
	# rtol matches the cavity experiment. With gamma capped at 1e2 the hardest
	# cell is level 3 there, which took 302 iterations for the proper inner
	# product and 336 for the standard one, so 500 is comfortable headroom
	# again -- the cap was briefly raised to 2000 while gamma = 1e3 was still in
	# the grid. The convergence flags in the CSV mean a capped count can no
	# longer be mistaken for a converged one.
	run(args.levels, rtol=1e-6, maxiter=500, results_dir=args.results_dir)
