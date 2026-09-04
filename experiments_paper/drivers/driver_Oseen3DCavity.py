# -*- coding: utf-8 -*-
"""
Driver: Oseen 3D Cavity (saddle-point), native Firedrake pipeline.

Assembles with Firedrake (src/python/generation/oseen3d_cavity.py) and runs
subGMRES end-to-end, writing the convergence-history, standard-history, and
equivalence-constant CSVs. Must run inside the Firedrake virtual environment.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "generation"))

import driver_common as dc
import oseen3d_cavity

if __name__ == "__main__":
	# Levels 1-3 by default, matching the cap in run_experiments.sh. The paper
	# goes to level 5, but levels 4 and 5 need a large shared-memory machine --
	# so they are asked for explicitly, never reached by leaving --levels off.
	args = dc.parse_common_args(range(1, 4), "Oseen 3D Cavity experiment (Firedrake).",
		add_solution_export=True)
	# With --export-solution, also write the wind, velocity and pressure fields
	# to ParaView PVD/VTU for the flow figure. The manuscript uses the finest
	# level; the export runs at every level requested, which costs little.
	sink = None
	if args.export_solution:
		sink = lambda level, x: oseen3d_cavity.export_solution(level, x, args.paraview_dir)
	dc.run_saddle_point_experiment(
		"Oseen3DCavity", oseen3d_cavity.assemble_problem, args.levels,
		rtol=1e-6, maxiter=100, results_dir=args.results_dir, write_history=True,
		export_solution=sink)
