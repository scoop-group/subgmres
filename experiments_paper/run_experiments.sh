#!/bin/bash
#
# Reproduce the paper's numerical experiments, locally.
#
# Each driver assembles its problem with Firedrake, solves it with subgmres over
# a range of mesh levels, and writes the CSV files the manuscript's figures and
# tables are built from. See INSTALL.md for the Firedrake environment, which is
# pinned; nothing else in the repository needs it.
#
#   ./run_experiments.sh                          run every driver below
#   ./run_experiments.sh driver_Helmholtz2D       run only the named ones
#
# Results land in ../results, next to this directory.

set -u

# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------
# Two different things want threads, and they are configured separately.
#
# OMP_NUM_THREADS stays at 1 throughout, on Firedrake's own recommendation: its
# assembly is parallelized across processes, not threads, and letting OpenMP
# oversubscribe the cores slows it down.
#
# MKL_NUM_THREADS controls the direct solves inside subgmres, which dominate the
# cost at the finer levels. They are threaded only if the optional `pypardiso`
# package is installed, which routes them through MKL PARDISO; without it the
# solver falls back to serial SuperLU and this setting has no effect at all.
# The two do not collide, Firedrake not using MKL.
#
# It defaults to 1, so nothing here assumes anything about your machine, but a
# value you set yourself is kept -- if you have pypardiso and cores to spare:
#
#   MKL_NUM_THREADS=8 ./run_experiments.sh
#
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RESULTS_DIR="../results"

# ---------------------------------------------------------------------------
# Drivers and mesh levels
# ---------------------------------------------------------------------------
# Two of the three drivers run the full range behind the published figures; they
# are cheap enough to. Only driver_Oseen3DCavity is capped, at level 3, so that
# this script finishes on an ordinary workstation. The commented line under it is
# the range the manuscript used. Its last two levels are out of reach of a
# workstation: the preconditioner is factorized directly and the fill grows
# faster than the problem size, so memory, rather than time, is what runs out.
# Raise the list one level at a time and watch the memory.
declare -A LEVELS=(
	[driver_Oseen3DCavity]="1 2 3"
	# manuscript: [driver_Oseen3DCavity]="1 2 3 4 5"

	[driver_Helmholtz2D]="1 2 3 4 5 6 7"

	[driver_Oseen3DCavityRobustness]="1 2 3"
	# This one sweeps a 4x4 grid of (mu, rho) at every level, so each level
	# costs sixteen solve pairs rather than one.
)

DRIVERS="${!LEVELS[*]}"
if [ $# -gt 0 ]; then
	for DRIVER in "$@"; do
		if [ -z "${LEVELS[$DRIVER]+set}" ]; then
			echo "ERROR: unknown driver '${DRIVER}'. Known: ${!LEVELS[*]}" >&2
			exit 1
		fi
	done
	DRIVERS="$*"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# A driver that fails -- most likely by running out of memory on a level too
# large for the machine -- is reported and skipped, so the remaining drivers
# still produce their results.
mkdir -p "${RESULTS_DIR}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}, MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "Writing results to ${RESULTS_DIR}"

FAILED=""
for DRIVER in ${DRIVERS}; do
	echo
	echo "=== ${DRIVER}, levels ${LEVELS[$DRIVER]} ==="
	if python3 "drivers/${DRIVER}.py" --levels ${LEVELS[$DRIVER]} \
			--results-dir "${RESULTS_DIR}"; then
		echo "--- ${DRIVER} done"
	else
		echo "--- WARNING: ${DRIVER} failed (exit $?); continuing" >&2
		FAILED="${FAILED} ${DRIVER}"
	fi
done

echo
if [ -n "${FAILED}" ]; then
	echo "Finished, with failures:${FAILED}" >&2
	exit 1
fi
echo "Finished. All drivers completed."
