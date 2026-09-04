# -*- coding: utf-8 -*-
"""
Portable ASCII VTU/PVD writing.

Firedrake's own VTKFile emits raw appended binary, which ParaView 5.11 and
older fail to parse ("not well-formed (invalid token)"). Everything the
manuscript renders is therefore written as plain ASCII VTU, readable by every
ParaView version, which keeps the figures reproducible across machines.

This module is the general form of the writer first used for the Helmholtz
figure: several point-data arrays per file, scalar or vector, and a choice of
cell type, so that the 2D triangle meshes and the 3D tetrahedral ones share one
implementation.
"""

import io
import os

import numpy as np

# VTK cell type codes for the linear simplices we export on.
VTK_TRIANGLE = 5
VTK_TETRA = 10


def _ascii_block(array, fmt):
	# Render a numpy array as whitespace-separated ASCII text for embedding in a
	# VTU <DataArray>: np.savetxt writes one line per row (2D) or per element (1D).
	buf = io.StringIO()
	np.savetxt(buf, array, fmt=fmt)
	return buf.getvalue()


def orient_tets(points, cells):
	"""Reorder each tetrahedron's nodes so that its signed volume is positive.

	Firedrake's cell-node map is not oriented for VTK, and on a BoxMesh exactly
	half the tetrahedra come out inverted. Rendering mostly survives that, but
	anything computing a signed volume does not: ParaView's IntegrateVariables
	then reports a total volume of zero, the positive and negative cells having
	cancelled, and every integral over the mesh with it. Swapping the last two
	nodes of an inverted cell flips its sign and changes nothing else."""
	cells = np.array(cells, copy=True)
	a, b, c, d = (points[cells[:, i]] for i in range(4))
	inverted = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0.0
	cells[inverted, 2], cells[inverted, 3] = cells[inverted, 3], cells[inverted, 2].copy()
	return cells


def write_ascii_vtu(path, points, cells, arrays, cell_type=VTK_TETRA):
	"""Write an uncompressed ASCII VTU with one or more point-data arrays.

	``points`` is (n_points, 3), ``cells`` is (n_cells, nodes_per_cell) holding
	node indices into ``points``, and ``arrays`` is a sequence of (name, values)
	pairs whose values are (n_points,) for a scalar or (n_points, 3) for a
	vector. The first scalar and the first vector name the piece's default
	Scalars/Vectors attributes, which is what ParaView colours by on load."""
	n_points, n_cells = points.shape[0], cells.shape[0]
	# VTU cell bookkeeping: the running end-offset into the connectivity array
	# per cell, and one type code per cell.
	offsets = np.arange(1, n_cells + 1) * cells.shape[1]
	types = np.full(n_cells, cell_type, dtype=np.uint8)
	# Default colouring: first scalar and first vector field, if present.
	scalars = next((nm for nm, v in arrays if np.asarray(v).ndim == 1), None)
	vectors = next((nm for nm, v in arrays if np.asarray(v).ndim == 2), None)
	attrs = "".join(f' {k}="{v}"' for k, v in
		(("Scalars", scalars), ("Vectors", vectors)) if v is not None)
	with open(path, "w") as f:
		# XML header: an unstructured grid with a single piece.
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
		f.write('  <UnstructuredGrid>\n')
		f.write('    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (n_points, n_cells))
		# Node coordinates, x y z per line.
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
		# Point data: one <DataArray> per field, vectors declaring 3 components.
		f.write('      <PointData%s>\n' % attrs)
		for name, values in arrays:
			values = np.asarray(values)
			ncomp = 1 if values.ndim == 1 else values.shape[1]
			comp = '' if ncomp == 1 else ' NumberOfComponents="%d"' % ncomp
			f.write('        <DataArray type="Float64" Name="%s"%s format="ascii">\n'
				% (name, comp))
			f.write(_ascii_block(values, "%.17g"))
			f.write('        </DataArray>\n')
		f.write('      </PointData>\n')
		f.write('    </Piece>\n')
		f.write('  </UnstructuredGrid>\n')
		f.write('</VTKFile>\n')


def write_pvd(path, vtu_name):
	# Minimal PVD collection pointing at a single flat, same-directory VTU, so a
	# render script's PVDReader finds it -- matching the old flat FEniCS layout.
	with open(path, "w") as f:
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="Collection" version="0.1">\n')
		f.write('  <Collection>\n')
		f.write('    <DataSet timestep="0" part="0" file="%s" />\n' % vtu_name)
		f.write('  </Collection>\n')
		f.write('</VTKFile>\n')


def write_fields(out_dir, basename, level, points, cells, arrays, cell_type=VTK_TETRA):
	"""Write ``<basename>_<LL>.vtu`` plus its ``.pvd`` into ``out_dir`` and
	return the VTU path. Flat, same-directory layout, as the render scripts
	expect."""
	os.makedirs(out_dir, exist_ok=True)
	if cell_type == VTK_TETRA:
		cells = orient_tets(points, cells)
	tag = "%02d" % level
	vtu_name = "%s_%s.vtu" % (basename, tag)
	vtu_path = os.path.join(out_dir, vtu_name)
	write_ascii_vtu(vtu_path, points, cells, arrays, cell_type=cell_type)
	write_pvd(os.path.join(out_dir, "%s_%s.pvd" % (basename, tag)), vtu_name)
	return vtu_path
