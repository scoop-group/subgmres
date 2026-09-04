# -*- coding: utf-8 -*-
"""
Driver: Helmholtz 2D (complex, single block), native Firedrake pipeline.

Assembles the real K/M/Mbdry blocks with Firedrake
(src/python/generation/helmholtz2d.py); the complex shifted system, the
shifted-Laplacian preconditioner, and the Graham-Spence-Vainikko inner product
are formed in the runner with the wavenumber k and preconditioner parameters
(alpha, beta) affixed here. Must run inside the Firedrake virtual environment.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "generation"))

import driver_common as dc
import helmholtz2d

if __name__ == "__main__":
	args = dc.parse_common_args(range(1, 8), "Helmholtz 2D experiment (Firedrake).",
		add_solution_export=True)
	# With --export-solution, also write the solution field to ParaView PVD/VTU
	# (for the Figure 8.4 amplitude/phase rendering via
	# src/paraview/Helmholtz2D_images.py).
	sink = None
	if args.export_solution:
		sink = lambda level, u: helmholtz2d.export_solution(level, u, args.paraview_dir)
	dc.run_helmholtz_experiment(
		"Helmholtz2D", helmholtz2d.assemble_problem, args.levels,
		k=20.0, alpha=0.0, beta=1.0, rtol=1e-6, maxiter=100,
		results_dir=args.results_dir, export_solution=sink)
