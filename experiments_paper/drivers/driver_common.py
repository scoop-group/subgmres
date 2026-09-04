# -*- coding: utf-8 -*-
"""
Shared post-processing for the experiment drivers.

These helpers compute and write the ``*.csv`` files the manuscript plots.
They began as a port of the MATLAB drivers and their support functions
(``check_plot_residual_norms.m``, ``evaluate_equivalence_constants.m``) and
were validated against the CSVs those produced; both now sit in
``../../matlab/obsolete/``, the experiments having moved to Firedrake.
"""

import os
import sys
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

# Make this directory and the parent src/python directory (for the subgmres
# module) importable regardless of the caller's working directory. The third
# entry is where the subgmres package sits in the public release, whose layout
# puts the drivers in experiments_paper/ and the package in a sibling python/;
# in this repository it does not exist and adding it costs nothing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_PYTHON = os.path.abspath(os.path.join(_HERE, os.pardir))
_RELEASE_PYTHON = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "python"))
for _p in (_HERE, _SRC_PYTHON, _RELEASE_PYTHON):
	if _p not in sys.path:
		sys.path.insert(0, _p)

from subgmres import subgmres
from subgmres._direct import make_solve

# Take our own command line out of PETSc's way.
#
# Firedrake initializes PETSc when it is imported, and PETSc reads sys.argv at
# that moment, registering everything it finds as an option of its own. Ours are
# not options of its own, so at exit it prints
#
#   WARNING! There are options you set that were not used!
#   Option left: name:--levels value: 1 source: command line
#
# which looks like a failure and is not. Every driver imports this module before
# it imports anything that pulls in Firedrake, so stashing the arguments here
# and handing PETSc a bare command line settles it for all of them at once. The
# parsers read COMMAND_LINE instead of sys.argv; nothing else changes.
COMMAND_LINE = sys.argv[1:]
del sys.argv[1:]

# Default results location, relative to this file: the results/ directory beside
# the drivers' parent, which is the same place run_experiments.sh writes. In the
# public release that is release/results/, sitting next to the read-only
# release/results_published/, so a driver called with no --results-dir lands
# where the reader expects and overwrites nothing.
#
# CAUTION, in this repository: the same expression resolves to src/results/,
# which holds the committed CSVs the manuscript reads at compile time. A bare
# exploratory run therefore overwrites them. Pass --results-dir, or check
# `git diff src/results` before committing.
DEFAULT_RESULTS_DIR = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "results"))
# ParaView export target (src/paraview), where the render scripts read from.
DEFAULT_PARAVIEW_DIR = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "paraview"))


def _column_energy_norms(R, solve=None):
	# Return the per-column values sqrt(real(sum(conj(R[:,j]) * y[:,j]))),
	# where y = R if solve is None, else y = solve(R) (i.e. the metric
	# M^{-1} applied column-wise). This is the diagonal of R' * (M \ R),
	# matching the MATLAB `sqrt(diag(R' * (M \ R)))` / `sqrt(diag(R' * R))`.
	Y = R if solve is None else solve(R)
	quad = np.real(np.sum(np.conj(R) * Y, axis=0))
	# The quadratic form is a squared norm; clamp roundoff-level negatives.
	return np.sqrt(np.maximum(quad, 0.0))


def _metric_solver(M_block):
	# Return a callable solving M_block x = rhs for dense multi-column rhs, via a
	# single direct factorization (the analogue of MATLAB's decomposition(M)).
	# Uses the shared subgmres._direct.make_solve backend, so the large real
	# inner-product blocks go through MKL PARDISO when available (else SuperLU),
	# with the complex-rhs-against-real-factorization case handled there too.
	return make_solve(sp.csc_matrix(M_block))


def write_robustness_equivalence(long_csv, out_csv):
	"""Pivot the robustness sweep into the shape its manuscript table has:
	one row per viscosity, one column per mesh level, holding the ratio
	C/c of the equivalence constants averaged over the four densities.

	The averaging is legitimate because the ratio turns out not to depend on
	rho -- across the four values at fixed mu it varies by about 15% -- so the
	table reports mu against the level rather than repeating four near-equal
	columns. Written so that the table can be read from a file at compile time,
	like the other two equivalence tables, rather than transcribed by hand: the
	hand-written version silently kept squared values through a change of
	convention.
	"""
	d = np.genfromtxt(long_csv, delimiter=",", names=True)
	mus = sorted(set(d["mu"]), reverse=True)
	levels = sorted(set(d["level"].astype(int)))
	with open(out_csv, "w") as f:
		f.write("mu," + ",".join("level%d" % lv for lv in levels) + "\n")
		for mu in mus:
			vals = []
			for lv in levels:
				m = (d["level"] == lv) & (d["mu"] == mu)
				vals.append(float(np.mean(d["C"][m] / d["c"][m])))
			f.write("%.6e," % mu + ",".join("%.6e" % v for v in vals) + "\n")
	return out_csv


def saddle_point_true_subnorms(R, M, n):
	"""True M-inverse-norms of the velocity (rows 0:n) and pressure (rows
	n:) residual subvectors, column by column. If M is None the standard
	(standard) norm is used. Mirrors ``check_plot_residual_norms.m``."""
	R1 = R[:n, :]
	R2 = R[n:, :]
	if M is None:
		true1 = _column_energy_norms(R1)
		true2 = _column_energy_norms(R2)
	else:
		M = sp.csc_matrix(M)
		solve1 = _metric_solver(M[:n, :n])
		solve2 = _metric_solver(M[n:, n:])
		true1 = _column_energy_norms(R1, solve1)
		true2 = _column_energy_norms(R2, solve2)
	return true1, true2


def helmholtz_true_norm(R, MM):
	"""True norm of the (single-block) residual, column by column: the
	MM-inverse-norm if MM is given, else the standard norm. Mirrors the
	Helmholtz driver's ``sqrt(real(diag(R' * (MM \\ R))))``."""
	if MM is None:
		return _column_energy_norms(R)
	return _column_energy_norms(R, _metric_solver(sp.csc_matrix(MM)))


def equivalence_constants(V):
	"""For each prefix V[:, :j] (j = 1..nV), the smallest and largest
	eigenvalues of V[:, :j]' * V[:, :j] -- whose SQUARE ROOTS are the
	equivalence constants c, C between the M-inverse and standard norms on
	that subspace (the columns of V being M-inverse-orthonormal). Mirrors
	``evaluate_equivalence_constants.m``, but forms the full Gram matrix once
	and eigendecomposes its leading principal submatrices.

	At a happy breakdown -- the GMRES residual reaches zero, so the Krylov
	subspace stops growing -- the final column of V is not a genuine basis
	vector: the Gram matrix turns numerically singular and its smallest
	eigenvalue underflows to a meaningless value at or below machine zero (it
	can even come out slightly negative, though a Gram matrix is positive
	semidefinite). Such rank-deficient prefixes are dropped, so the returned
	sequences end at the last iteration whose basis is numerically full rank."""
	nV = V.shape[1]
	# Gram matrix (Hermitian for complex V, symmetric for real V).
	G = V.conj().T @ V
	c_vals = np.empty(nV)
	C_vals = np.empty(nV)
	# Relative rank tolerance (as in numpy.linalg.matrix_rank): a smallest
	# eigenvalue at or below this counts as numerical zero, i.e. the basis has
	# stopped growing. Legitimate small constants (~1e-3 relative even at the
	# finest level) sit far above it; a happy-breakdown value (~1e-16 relative)
	# sits far below.
	eps = np.finfo(float).eps
	kept = 0
	for j in range(1, nV + 1):
		evals = np.linalg.eigvalsh(G[:j, :j])
		# Stop at the first rank-deficient prefix: its newest column is a
		# happy-breakdown artifact, not a basis vector, so disregard it (and all
		# that would follow -- the subspace does not grow again).
		if evals[0] <= j * eps * evals[-1]:
			break
		c_vals[j - 1] = evals[0]
		C_vals[j - 1] = evals[-1]
		kept = j
	# Return the constants themselves, not their squares. For r = V w with V
	# M-inverse-orthonormal one has ||r||_{M^-1}^2 = w^H w and ||r||_I^2 =
	# w^H (V^H V) w, so the eigenvalues above are the extreme values of the
	# SQUARED quotient. The manuscript defines c and C through the norms,
	#     c ||r||_{M^-1} <= ||r||_I <= C ||r||_{M^-1},
	# because it is that quotient C/c, and not its square, which bounds how far
	# a criterion applied in the standard norm can fall short in the M-inverse
	# norm. Hence the square root here.
	return np.sqrt(c_vals[:kept]), np.sqrt(C_vals[:kept])


def write_history_saddle_point(filename, true1, true2, norm1, norm2):
	"""Write the saddle-point convergence history CSV with columns
	iter,trueNormR1,trueNormR2,normR1,normR2 (0-based iter). Matches the
	MATLAB ``fprintf(fid,'%d,%.6e,%.6e,%.6e,%.6e\\n',...)``."""
	with open(filename, "w") as f:
		f.write("iter,trueNormR1,trueNormR2,normR1,normR2\n")
		for k in range(len(true1)):
			f.write(f"{k},{true1[k]:.6e},{true2[k]:.6e},{norm1[k]:.6e},{norm2[k]:.6e}\n")


def write_history_single(filename, true_norm, norm):
	"""Write the single-block (Helmholtz) history CSV with columns
	iter,trueNorm,normR (0-based iter)."""
	with open(filename, "w") as f:
		f.write("iter,trueNorm,normR\n")
		for k in range(len(true_norm)):
			f.write(f"{k},{true_norm[k]:.6e},{norm[k]:.6e}\n")


def write_timings(filename, problem, level, dims, phases):
	"""Write a per-level wall-clock timing report for the run. ``dims`` is a
	dict of dimension name -> value (e.g. ``{"n": n, "m": m}``); ``phases`` is
	a list of ``(label, seconds, iters_or_None)`` tuples in execution order
	(matrix assembly and each solve). The format is a small, human-readable
	text file with the phase labels aligned; solves also report their iteration
	count. This is the Python counterpart of the MATLAB drivers' ``_timings.txt``
	and is the baseline against which a threaded direct solver is measured."""
	# Pad the phase labels to a common width for a tidy column layout.
	width = max((len(label) for label, _, _ in phases), default=0)
	lines = [f"problem: {problem}", f"level: {level}"]
	# Append each recorded dimension (velocity/pressure block sizes etc.).
	for name, value in dims.items():
		lines.append(f"{name}: {value}")
	lines.append("")
	# One line per timed phase; solves carry an iteration count, assembly does not.
	for label, seconds, iters in phases:
		if iters is None:
			lines.append(f"{label.ljust(width)}  {seconds:10.3f} s")
		else:
			lines.append(f"{label.ljust(width)}  {iters:>4d} iters, {seconds:10.3f} s")
	with open(filename, "w") as f:
		f.write("\n".join(lines) + "\n")


def write_equivalence(filename, c_vals, C_vals):
	"""Write the equivalence-constant CSV with columns iter,c,C (1-based
	iter, matching the MATLAB ``for j = 1:nV``)."""
	with open(filename, "w") as f:
		f.write("iter,c,C\n")
		for j in range(len(c_vals)):
			f.write(f"{j + 1},{c_vals[j]:.6e},{C_vals[j]:.6e}\n")


def parse_common_args(default_levels, description, add_solution_export=False):
	"""Parse the shared driver command-line arguments: --levels (space-
	separated ints) and --results-dir. When ``add_solution_export`` is set, also
	accept --export-solution and --paraview-dir (used by the Helmholtz driver to
	export the solution field for the Figure 8.4 ParaView rendering)."""
	import argparse

	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--levels", type=int, nargs="+", default=list(default_levels),
		help="Mesh levels to run. Default: %(default)s.")
	parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
		help="Directory to write the result CSVs into. Default: ../results.")
	if add_solution_export:
		# Opt-in solution-field export to ParaView PVD/VTU (Figure 8.4).
		parser.add_argument("--export-solution", action="store_true",
			help="Also export the solution field to ParaView PVD/VTU (for Figure 8.4).")
		parser.add_argument("--paraview-dir", default=DEFAULT_PARAVIEW_DIR,
			help="Directory for the exported ParaView files. Default: ../paraview.")
	return parser.parse_args(COMMAND_LINE)


def run_saddle_point_experiment(
	basename, generate, levels, rtol, maxiter,
	results_dir=DEFAULT_RESULTS_DIR, write_history=True, verbose=True,
	export_solution=None,
):
	"""Run an Oseen saddle-point experiment over the given mesh levels and
	write the result CSVs.

	`generate(level)` supplies the system as ``(A, r, P, M, n, m, x0)``; the
	generators live in ``generation/``.

	x0 is the discrete Dirichlet lifting, used as the initial guess so that the
	retained trivial equations of the constrained dofs start (and stay) at zero
	residual and do not enter the monitored norms; see
	``firedrake_utils.assemble_oseen_saddle_point``. Passing None instead means
	the all-zero guess, under which those equations do enter the norms.

	For each level: a "proper" solve using the M-inverse inner product
	(P preconditioner, M inner product) produces the equivalence-constant
	CSV and -- if ``write_history`` -- the history CSV; a second "standard"
	solve (same P, standard inner product) produces the history_standard
	CSV. Oseen2DCavity in the MATLAB code only emits equivalence, so it
	passes ``write_history=False``.

	If ``export_solution`` is given (a callable ``(level, x)``), it is invoked
	with the solution of the proper solve after each level -- used to write the
	wind, velocity and pressure fields for the ParaView rendering.
	"""
	os.makedirs(results_dir, exist_ok=True)
	for level in levels:
		# Time the matrix assembly (Firedrake) / read (legacy) separately.
		t0 = perf_counter()
		A, r, P, M, n, m, x0 = generate(level)
		t_gen = perf_counter() - t0
		# Subvector partition: velocity block [0:n], pressure block [n:].
		# subgmres takes 0-based starting indices (MATLAB used [1 n+1]).
		ix = [0, n]
		out = os.path.join(results_dir, f"{basename}_{level:02d}")
		if verbose:
			print(f"[{basename}] level {level}: n={n} m={m}, factorizing P...", flush=True)

		# Factorize the preconditioner P once and reuse it across BOTH solves:
		# the proper and standard solves use the same P and differ only in the
		# inner product M. Passing a ready LinearOperator makes subgmres skip its
		# internal re-factorization, halving the dominant cost at the large levels
		# (subgmres.coerce_operator passes a LinearOperator through unchanged).
		t0 = perf_counter()
		P_op = LinearOperator(A.shape, matvec=make_solve(P))
		t_factP = perf_counter() - t0
		if verbose:
			print(f"[{basename}] level {level}: proper solve...", flush=True)

		# Proper solve: reused P, M inner product (timed).
		t0 = perf_counter()
		res = subgmres(
			A, r, ix=ix, rtol=rtol, maxiter=maxiter, P=P_op, M=M, x0=x0,
			compute_r_hist=True, compute_sub_norms_hist=True, compute_v_hist=True,
		)
		t_proper = perf_counter() - t0
		norm1 = res.sub_norms_hist[0, :]
		norm2 = res.sub_norms_hist[1, :]
		if write_history:
			true1, true2 = saddle_point_true_subnorms(res.r_hist, M, n)
			write_history_saddle_point(out + "_history.csv", true1, true2, norm1, norm2)
		c_vals, C_vals = equivalence_constants(res.v_hist)
		write_equivalence(out + "_equivalence.csv", c_vals, C_vals)

		# Optionally export the solution fields for the ParaView rendering.
		if export_solution is not None:
			export_solution(level, res.x)

		# Collect the timing phases; the standard solve is added below if run.
		phases = [
			("generate", t_gen, None),
			("factorize P", t_factP, None),
			("solve (proper M)", t_proper, res.iter),
		]

		if write_history:
			if verbose:
				print(f"[{basename}] level {level}: standard solve...", flush=True)
			# Standard solve: same reused P, standard inner product (timed).
			t0 = perf_counter()
			res_std = subgmres(
				A, r, ix=ix, rtol=rtol, maxiter=maxiter, P=P_op, M=None, x0=x0,
				compute_r_hist=True, compute_sub_norms_hist=True,
			)
			t_std = perf_counter() - t0
			norm1s = res_std.sub_norms_hist[0, :]
			norm2s = res_std.sub_norms_hist[1, :]
			true1s, true2s = saddle_point_true_subnorms(res_std.r_hist, None, n)
			write_history_saddle_point(out + "_history_standard.csv", true1s, true2s, norm1s, norm2s)
			phases.append(("solve (standard M)", t_std, res_std.iter))

		write_timings(out + "_timings.txt", basename, level, {"n": n, "m": m}, phases)
		if verbose:
			# Report the proper-solve iteration count, and the standard-solve
			# count too when that solve was run (Oseen2DCavity skips it).
			iters_msg = f"proper {res.iter} iters"
			if write_history:
				iters_msg += f", standard {res_std.iter} iters"
			print(f"[{basename}] level {level}: done ({iters_msg}, flag {res.flag}).", flush=True)


def run_helmholtz_experiment(
	basename, generate, levels, k, alpha, beta, rtol, maxiter,
	results_dir=DEFAULT_RESULTS_DIR, verbose=True, export_solution=None,
):
	"""Run the complex Helmholtz experiment over the given mesh levels,
	mirroring ``driver_Helmholtz2D.m``. `generate(level)` supplies the real
	blocks ``(K, M, Mbdry, b, n)``; the complex system, preconditioner, and
	inner-product matrix are formed here with the wavenumber k and
	preconditioner parameters affixed.

	If ``export_solution`` is given (a callable ``(level, u)``), it is invoked
	with the complex solution of the proper solve after each level -- used to
	write the ParaView field for Figure 8.4."""
	os.makedirs(results_dir, exist_ok=True)
	for level in levels:
		# Time the real-block assembly (Firedrake) / read (legacy); the complex
		# system is then formed from those blocks with the constants affixed.
		t0 = perf_counter()
		K, M, Mbdry, b, n = generate(level)
		# Affix the constants: complex shifted system, shifted-Laplacian
		# preconditioner, and the Graham-Spence-Vainikko inner product.
		A = K - k**2 * M - 1j * k * Mbdry
		P = K + (alpha + 1j * beta) * k**2 * M - 1j * k * Mbdry
		MM = K + k**2 * M
		t_gen = perf_counter() - t0
		out = os.path.join(results_dir, f"{basename}_{level:02d}")
		if verbose:
			print(f"[{basename}] level {level}: n={n}, factorizing P...", flush=True)

		# Factorize the (complex) preconditioner P once and reuse it across both
		# the proper and standard solves, which differ only in the inner product.
		t0 = perf_counter()
		P_op = LinearOperator(A.shape, matvec=make_solve(P))
		t_factP = perf_counter() - t0
		if verbose:
			print(f"[{basename}] level {level}: proper solve...", flush=True)

		# Proper solve: reused P, MM-inverse inner product (timed).
		t0 = perf_counter()
		res = subgmres(
			A, b, ix=None, rtol=rtol, maxiter=maxiter, P=P_op, M=MM,
			compute_r_hist=True, compute_sub_norms_hist=True, compute_v_hist=True,
		)
		t_proper = perf_counter() - t0
		norm = res.sub_norms_hist[0, :]
		true_norm = helmholtz_true_norm(res.r_hist, MM)
		write_history_single(out + "_history.csv", true_norm, norm)
		c_vals, C_vals = equivalence_constants(res.v_hist)
		write_equivalence(out + "_equivalence.csv", c_vals, C_vals)

		# Optionally export the solution field (real/imag) for the Figure 8.4
		# ParaView rendering.
		if export_solution is not None:
			export_solution(level, res.x)

		# Standard solve: same reused P, standard inner product (timed).
		t0 = perf_counter()
		res_std = subgmres(
			A, b, ix=None, rtol=rtol, maxiter=maxiter, P=P_op, M=None,
			compute_r_hist=True, compute_sub_norms_hist=True,
		)
		t_std = perf_counter() - t0
		norm_s = res_std.sub_norms_hist[0, :]
		true_norm_s = helmholtz_true_norm(res_std.r_hist, None)
		write_history_single(out + "_history_standard.csv", true_norm_s, norm_s)

		write_timings(out + "_timings.txt", basename, level, {"n": n}, [
			("generate", t_gen, None),
			("factorize P", t_factP, None),
			("solve (proper M)", t_proper, res.iter),
			("solve (standard M)", t_std, res_std.iter),
		])
		if verbose:
			# Helmholtz always runs both solves; report both iteration counts.
			print(f"[{basename}] level {level}: done (proper {res.iter} iters, "
				f"standard {res_std.iter} iters, flag {res.flag}).", flush=True)
