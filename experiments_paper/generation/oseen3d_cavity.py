# -*- coding: utf-8 -*-
"""
Firedrake generation for the Oseen 3D Cavity problem.

Port of the FEniCS src/python/Oseen3DCavity.py: 3D Oseen driven-cavity flow on
[-1,1]^3 with Taylor-Hood elements, a rotational wind (zero in z), no-slip on
five walls, and a driven lid on top (y = 1).
"""

import numpy as np

from firedrake import (
	BoxMesh, SpatialCoordinate, as_vector, VectorFunctionSpace, FunctionSpace,
	Function, assemble, interpolate,
)

import vtu
from firedrake_utils import assemble_oseen_saddle_point


def _mesh_and_wind(level):
	"""The level-`level` mesh and its rotational wind, built identically for the
	assembly and for the ParaView export, so the dof numbering matches -- it is
	deterministic for identical construction."""
	# FEniCS mesh: BoxMesh([-1,-1,-1],[1,1,1], 4, 4, 4) then refine (level-1)x.
	ncells = 4 * 2 ** (level - 1)
	mesh = BoxMesh(ncells, ncells, ncells, 2.0, 2.0, 2.0)
	mesh.coordinates.dat.data[:] -= 1.0
	x, y, z = SpatialCoordinate(mesh)
	wind = as_vector([2 * y * (1 - x * x), -2 * x * (1 - y * y), 0.0])
	return mesh, wind


def assemble_problem(level, mu=1e-1, rho=1.0):
	"""Assemble the level-`level` Oseen 3D Cavity system. Returns
	(K, b, P, M, n, m, x0); see firedrake_utils.assemble_oseen_saddle_point."""
	mesh, wind = _mesh_and_wind(level)
	# BoxMesh markers: 1:x=0, 2:x=Lx, 3:y=0, 4:y=Ly, 5:z=0, 6:z=Lz.
	# The driven lid (y=1) is marker 4; the other five faces are no-slip.
	return assemble_oseen_saddle_point(
		mesh, wind, driven_marker=4, noslip_markers=(1, 2, 3, 5, 6),
		driven_value=(1.0, 0.0, 0.0), mu_val=mu, rho_val=rho)


def export_solution(level, x, out_dir, basename="Oseen3DCavity"):
	"""Write the wind, velocity and pressure fields of the level-`level`
	solution to ``<basename>_<LL>.vtu`` (plus its ``.pvd``) in ``out_dir``, as
	the point-data arrays ``wind``, ``velocity`` and ``pressure``.

	``x`` is the flat solution vector in the monolithic ordering of
	:func:`assemble_problem` -- the velocity block first, its components
	interleaved per node, then the pressure block.

	All three fields are written on the CG1 mesh vertices: the pressure lives
	there already, while the CG2 velocity and the analytic wind are interpolated
	down. That costs the mid-edge detail but keeps a single, small topology for
	all three arrays, which is what a ParaView glyph/slice pipeline wants; the
	fields are smooth on this mesh, so the visual difference is negligible.

	Written as portable ASCII VTU rather than through Firedrake's VTKFile, whose
	raw appended binary older ParaView cannot parse; see :mod:`vtu`."""
	mesh, wind = _mesh_and_wind(level)
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	n = V.dim()
	x = np.asarray(x)
	# Scatter the flat vector back into the two fields. The velocity block is
	# node-major with the three components interleaved, matching dat.data's
	# (n_nodes, 3) layout.
	uh = Function(V, name="velocity")
	uh.dat.data[:] = np.real(x[:n]).reshape(uh.dat.data.shape)
	ph = Function(Q, name="pressure")
	ph.dat.data[:] = np.real(x[n:])
	# Visualization space: vector CG1 on the same mesh. It shares Q's node
	# numbering (a VectorFunctionSpace is the scalar space with three components
	# per node), so Q's cell-node map indexes all three arrays alike.
	V1 = VectorFunctionSpace(mesh, "CG", 1)
	u1 = assemble(interpolate(uh, V1))
	w1 = assemble(interpolate(wind, V1))
	# Take the node coordinates from the same space, rather than from
	# mesh.coordinates, so they are aligned with the field values by construction.
	pts = assemble(interpolate(SpatialCoordinate(mesh), V1))
	return vtu.write_fields(
		out_dir, basename, level,
		points=np.asarray(pts.dat.data_ro),
		cells=np.asarray(Q.cell_node_map().values),
		arrays=[("wind", np.asarray(w1.dat.data_ro)),
		        ("velocity", np.asarray(u1.dat.data_ro)),
		        ("pressure", np.asarray(ph.dat.data_ro))],
		cell_type=vtu.VTK_TETRA)
