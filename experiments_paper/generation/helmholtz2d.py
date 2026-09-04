# -*- coding: utf-8 -*-
"""
Firedrake generation for the Helmholtz 2D problem.

Port of the FEniCS src/python/Helmholtz2D.py: a 2D Helmholtz problem on the
unit square with a point source, a Dirichlet (sound-soft) segment on the
southern boundary, and a Sommerfeld (impedance) condition on the other three
sides. The problem is naturally constant-decoupled: the real stiffness K, mass
M and Sommerfeld boundary-mass Mbdry are assembled here, and the driver forms
the complex system

    A  = K - k^2 M - i k Mbdry
    P  = K + (alpha + i beta) k^2 M - i k Mbdry     (shifted-Laplacian)
    MM = K + k^2 M                                   (inner product)

with the wavenumber k and preconditioner parameters affixed at solve time.
The complex arithmetic happens in numpy/scipy, so a real (default) PETSc scalar
type suffices -- no complex Firedrake build is needed.
"""

import io
import os

import numpy as np

from firedrake import (
	RectangleMesh, FunctionSpace, Function, TrialFunction, TestFunction,
	inner, grad, dx, ds, assemble,
)

from firedrake_utils import mat_to_scipy, eliminate_dirichlet


def assemble_problem(level):
	"""Assemble the level-`level` Helmholtz 2D components. Returns
	(K, M, Mbdry, b, n): real scipy matrices K (stiffness), M (mass), Mbdry
	(Sommerfeld boundary mass), the numpy right-hand side b (unit point source),
	and the dimension n. The driver combines these with the wavenumber."""
	ncells = 4 * 2 ** (level - 1)
	# Unit square [0,1]^2. RectangleMesh markers: 1:x=0 (west), 2:x=1 (east),
	# 3:y=0 (south), 4:y=1 (north).
	mesh = RectangleMesh(ncells, ncells, 1.0, 1.0)
	V = FunctionSpace(mesh, "CG", 1)
	n = V.dim()
	u = TrialFunction(V)
	v = TestFunction(V)

	K = mat_to_scipy(assemble(inner(grad(u), grad(v)) * dx, mat_type="aij"))
	M = mat_to_scipy(assemble(inner(u, v) * dx, mat_type="aij"))
	# Sommerfeld (impedance) boundary = west + east + north (not the southern
	# Dirichlet segment).
	Mbdry = mat_to_scipy(assemble(inner(u, v) * ds((1, 2, 4)), mat_type="aij"))

	# Dirichlet (sound-soft) dofs: the southern boundary (marker 3) minus the
	# node carrying the point source at (0.5, 0), which must stay free.
	coords = mesh.coordinates.dat.data_ro
	source = int(np.argmin(np.sum((coords - np.array([0.5, 0.0])) ** 2, axis=1)))
	south = V.boundary_nodes(3)
	dirichlet = np.array([d for d in south if d != source])

	# Clean homogeneous Dirichlet: K keeps a unit diagonal, while M and Mbdry
	# have their Dirichlet rows/columns zeroed, so the combined complex system
	# A = K - k^2 M - i k Mbdry has clean unit-diagonal Dirichlet rows.
	K = eliminate_dirichlet(K, dirichlet, 1.0)
	M = eliminate_dirichlet(M, dirichlet, 0.0)
	Mbdry = eliminate_dirichlet(Mbdry, dirichlet, 0.0)

	# Unit point source at (0.5, 0).
	b = np.zeros(n)
	b[source] = 1.0
	return K, M, Mbdry, b, n


def _ascii_block(array, fmt):
	# Render a numpy array as whitespace-separated ASCII text for embedding in a
	# VTU <DataArray>: np.savetxt writes one line per row (2D) or per element (1D).
	buf = io.StringIO()
	np.savetxt(buf, array, fmt=fmt)
	return buf.getvalue()


def _write_ascii_vtu(path, points, cells, array_name, values):
	"""Write a single-scalar-field, uncompressed ASCII VTU. Unlike Firedrake's
	own VTKFile -- which emits raw appended binary that older ParaView (<= 5.11)
	fails to parse ("not well-formed (invalid token)") -- a plain ASCII VTU is
	readable by every ParaView version, keeping the Figure 8.4 rendering
	reproducible across machines."""
	n_points = points.shape[0]
	n_cells = cells.shape[0]
	# VTU cell bookkeeping: the running end-offset into the connectivity array per
	# cell, and the VTK cell type (5 = triangle) for the mesh's triangular cells.
	offsets = np.arange(1, n_cells + 1) * cells.shape[1]
	types = np.full(n_cells, 5, dtype=np.uint8)
	with open(path, "w") as f:
		# XML header: an unstructured grid with a single piece.
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
		f.write('  <UnstructuredGrid>\n')
		f.write('    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (n_points, n_cells))
		# Node coordinates (x y z per line; the 2D mesh is padded with z = 0).
		f.write('      <Points>\n')
		f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
		f.write(_ascii_block(points, "%.17g"))
		f.write('        </DataArray>\n')
		f.write('      </Points>\n')
		# Cell connectivity, end offsets, and per-cell VTK type codes.
		f.write('      <Cells>\n')
		f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
		f.write(_ascii_block(cells, "%d"))
		f.write('        </DataArray>\n')
		f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
		f.write(_ascii_block(offsets, "%d"))
		f.write('        </DataArray>\n')
		f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
		f.write(_ascii_block(types, "%d"))
		f.write('        </DataArray>\n')
		f.write('      </Cells>\n')
		# The scalar point-data field, named u_R or u_I for the render script.
		f.write('      <PointData Scalars="%s">\n' % array_name)
		f.write('        <DataArray type="Float64" Name="%s" format="ascii">\n' % array_name)
		f.write(_ascii_block(values, "%.17g"))
		f.write('        </DataArray>\n')
		f.write('      </PointData>\n')
		f.write('    </Piece>\n')
		f.write('  </UnstructuredGrid>\n')
		f.write('</VTKFile>\n')


def _write_pvd(path, vtu_name):
	# Minimal PVD collection pointing at a single flat, same-directory VTU, so the
	# render script's PVDReader finds it -- matching the old flat FEniCS layout.
	with open(path, "w") as f:
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="Collection" version="0.1">\n')
		f.write('  <Collection>\n')
		f.write('    <DataSet timestep="0" part="0" file="%s" />\n' % vtu_name)
		f.write('  </Collection>\n')
		f.write('</VTKFile>\n')


def export_solution(level, u, out_dir, basename="Helmholtz2D"):
	"""Write the complex Helmholtz solution ``u`` (a numpy array of length
	V.dim(), in the same dof order as :func:`assemble_problem`'s right-hand side)
	to ParaView files ``<basename>_<LL>_real.pvd`` and ``_imag.pvd`` (each with a
	flat, same-directory ASCII ``.vtu``), with the point-data arrays named
	``u_R`` and ``u_I`` -- the exact inputs expected by
	``src/paraview/Helmholtz2D_images.py``, which renders manuscript Figure 8.4.

	The files are written as plain ASCII VTU rather than via Firedrake's VTKFile
	(whose raw appended binary older ParaView cannot parse), so the figure stays
	reproducible on any ParaView version.

	The mesh and CG1 space are rebuilt exactly as in :func:`assemble_problem`;
	Firedrake's dof numbering is deterministic for identical construction, so
	``u`` maps back onto the field node-for-node."""
	# Rebuild the identical mesh and space so the dof order matches ``u``.
	ncells = 4 * 2 ** (level - 1)
	mesh = RectangleMesh(ncells, ncells, 1.0, 1.0)
	V = FunctionSpace(mesh, "CG", 1)
	u = np.asarray(u)
	# Real and imaginary parts as separate CG1 fields, named for the render script.
	uR = Function(V, name="u_R")
	uR.dat.data[:] = np.real(u)
	uI = Function(V, name="u_I")
	uI.dat.data[:] = np.imag(u)
	# Shared topology for both parts, taken from the same CG1 space so it aligns
	# with the field values node-for-node: node coordinates padded to 3D, and the
	# triangle connectivity as CG1 dof indices.
	coords = np.asarray(mesh.coordinates.dat.data_ro)
	points = np.column_stack([coords, np.zeros(coords.shape[0])])
	cells = np.asarray(V.cell_node_map().values)
	# Write a portable ASCII VTU (+ PVD) for each part into out_dir.
	os.makedirs(out_dir, exist_ok=True)
	tag = "%02d" % level
	for part_name, array_name, field in (("real", "u_R", uR), ("imag", "u_I", uI)):
		vtu_name = f"{basename}_{tag}_{part_name}.vtu"
		_write_ascii_vtu(os.path.join(out_dir, vtu_name), points, cells,
			array_name, np.asarray(field.dat.data_ro))
		_write_pvd(os.path.join(out_dir, f"{basename}_{tag}_{part_name}.pvd"), vtu_name)
