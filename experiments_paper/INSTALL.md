# Reproducing the paper's experiments

Three drivers live here. Each assembles its finite-element system, solves it
with `subgmres` over a range of mesh levels, and writes the CSV files behind the
manuscript's figures and tables. No matrix files are read or written; everything
is assembled in memory.

| Driver | Problem |
|---|---|
| `driver_Oseen3DCavity` | Oseen equations, three-dimensional lid-driven cavity (a saddle-point system) |
| `driver_Oseen3DCavityRobustness` | the same cavity, swept over a grid of viscosity and reaction parameters |
| `driver_Helmholtz2D` | Helmholtz equation, two-dimensional, complex-valued |

The drivers need [Firedrake](https://www.firedrakeproject.org/) to assemble
finite element matrices and vectors.

## Quick start

Two commands, run from this directory `experiments_paper/`. The first fetches
Firedrake, pinned to the version the paper used; the second reproduces every
experiment in the paper inside it.

```bash
docker pull firedrakeproject/firedrake:2026.4.1

docker run --rm -v "$PWD/..":/work -w /work/experiments_paper \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    firedrakeproject/firedrake:2026.4.1 \
    ./run_experiments.sh
```

The pull moves about 2.5 GiB over the network and occupies about 5.5 GiB once
unpacked. `2026.4.1` is the Firedrake version the published results were
computed with. To be precise,
```bash
docker image inspect firedrakeproject/firedrake:2026.4.1 \
    --format '{{index .RepoDigests 0}}'
```
should show `firedrakeproject/firedrake@sha256:0a94880f8abb6b5c82d06cd43ba9303a02159775d3d1161d1dff9c6245d428b2`.

The CSV files appear in `../results`. The parameter sweep is much the longest of
the three drivers.

If Docker is not available to you, see "Installing Firedrake without Docker"
below. With Firedrake installed natively, the whole reproduction is the second
command without its `docker run` prefix — just `./run_experiments.sh`.

## Running the experiments

Everything below is a command to run *inside* the Firedrake environment: either
appended to the `docker run` line above in place of `./run_experiments.sh`, or
typed directly if you have a native installation. To avoid repeating that long
line, open a shell in the container once and work inside it:

```bash
docker run --rm -it -v "$PWD/..":/work -w /work/experiments_paper \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    firedrakeproject/firedrake:2026.4.1 \
    bash
```

### With `run_experiments.sh`

The script is the reproduction itself, and takes an optional list of drivers:

```bash
./run_experiments.sh                        # all three: the whole paper
./run_experiments.sh driver_Helmholtz2D     # only the ones you name
```

Beyond looping over the drivers it does two things worth having. It pins the
thread environment, described under "Threads" below, so a run does not depend on
what your shell happened to export. And it treats a driver that dies — most
often by running out of memory on a level too large for the machine — as a
warning rather than the end: the remaining drivers still run and still write
their results, and the script exits non-zero at the end to say that something
failed.

### Calling a driver directly

Use this to vary something the `run_experiments.sh` script fixes, most
obviously the mesh levels:

```bash
python3 drivers/driver_Helmholtz2D.py --levels 1 2
python3 drivers/driver_Helmholtz2D.py --help
```

A driver invoked this way behaves as the script would in every respect but the
two above. Its own default mesh levels are exactly the levels the script would
have given it, and it writes to `../results` either way, so

```bash
python3 drivers/driver_Helmholtz2D.py
```

and `./run_experiments.sh driver_Helmholtz2D` write the same files over the same
mesh levels. What differs is that the thread environment is inherited from your
shell rather than pinned — which can move the final digits of the residual norms,
for the reason given under "Comparing against the published results" — and that
there is no progress or exit-status reporting around the run.

Each driver runs in a single process. There is no MPI parallelism to arrange.

## What comes out

Everything lands in `../results` by default, one set of files per problem and mesh level.

| File | Contents |
|---|---|
| `<Problem>_<level>_history.csv` | the residual subvector norms at every iteration, measured in the inner product the paper proposes — the convergence plots |
| `<Problem>_<level>_history_standard.csv` | the same problem solved again in the standard inner product, which is what those plots are compared against |
| `<Problem>_<level>_equivalence.csv` | the norm-equivalence constants between the two, tabulated in the paper |
| `<Problem>_<level>_timings.txt` | problem sizes and wall-clock time per phase |

The robustness sweep writes two files instead: `Oseen3DCavityRobustness.csv`,
one row per parameter pair, and `Oseen3DCavityRobustness_equivalence.csv`, the
pivoted table the manuscript prints.

## Comparing against the published results

The CSV files behind the manuscript are in `../results_published`. Nothing
writes there — a run of your own goes to `../results`, so the two sit side by
side and the comparison is yours to make however you prefer.

Two things are worth knowing before you make it. **Iteration counts should
reproduce exactly, residual norms may not reproduce exactly**:
floating-point summation orders differ between machines, and the files are
written at fixed precision, so expect disagreement in the final digits.

`../results_published` also holds levels 4 and 5 of `driver_Oseen3DCavity`,
which `run_experiments.sh` does not run; see "Mesh levels" below for why.

## Mesh levels

Two of the three drivers run the paper's full range of mesh levels. Only
`driver_Oseen3DCavity` is capped, so that a run finishes on an ordinary machine.
The cap is in both places that choose levels — the driver's own default and the
list in `run_experiments.sh` — so it holds however you invoke it; the paper's
range sits beside the script's, commented out.

| Driver | Levels run | Paper uses |
|---|---|---|
| `driver_Oseen3DCavity` | 1–3 | 1–5 |
| `driver_Oseen3DCavityRobustness` | 1–3 | 1–3 |
| `driver_Helmholtz2D` | 1–7 | 1–7 |

Going beyond level 3 with `driver_Oseen3DCavity` is what gets expensive. The
preconditioner is factorized directly, and the fill grows faster than the
problem size. The paper's two finest levels were run on a large shared-memory
machine and are out of reach of a workstation. Run `python3
drivers/driver_Oseen3DCavity.py --levels 4` to try out level 4 on your machine.
A driver that is killed without an error message ran out of memory.
`run_experiments.sh` reports it and carries on with the remaining drivers.

## Threads

Two independent settings, both handled by `run_experiments.sh`.

`OMP_NUM_THREADS` is held at 1, on Firedrake's own recommendation: its assembly
is parallelized across processes rather than threads, and letting OpenMP
oversubscribe the cores makes it slower.

`MKL_NUM_THREADS` controls the direct solves inside `subgmres`, which dominate
the cost at the finer levels. These are threaded only if the optional
[`pypardiso`](https://github.com/haasad/PyPardiso) package is installed, which
routes them through MKL PARDISO; without it the solver falls back to SciPy's
SuperLU, which is serial, and the setting does nothing at all. The two do not
collide, since Firedrake does not use MKL. The default is `MKL_NUM_THREADS=1`,
assuming nothing about your machine, but a value you set yourself is kept:

```bash
MKL_NUM_THREADS=8 ./run_experiments.sh
```

The script echoes both thread settings as its first line of output, so you can
see what it resolved them to.

The pinned container does *not* carry `pypardiso`, so on that path the setting
has no effect and the factorizations are serial. If desired, install `pypardiso` via

```bash
pip install --user pypardiso
python3 -c "import pypardiso; print('MKL PARDISO available')"
```

from the running Docker container (see "Running the experiments"). To make the
`pypardiso` installation persistent, create a file named `Dockerfile` with the
following content:
```
FROM firedrakeproject/firedrake:2026.4.1
RUN pip install pypardiso && ldconfig
```

Build the image once, from the directory holding that `Dockerfile`:

```bash
docker build -t subgmres-firedrake:2026.4.1 .
```

Then name `subgmres-firedrake:2026.4.1` wherever the instructions above name
`firedrakeproject/firedrake:2026.4.1`. From this directory, and now with a
thread count worth setting:

```bash
docker run --rm -v "$PWD/..":/work -w /work/experiments_paper \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e MKL_NUM_THREADS=8 \
    subgmres-firedrake:2026.4.1 \
    ./run_experiments.sh
```

The published results were computed with `pypardiso` at version 0.4.7.

## Installing Firedrake without Docker

Firedrake's own `firedrake-configure` script decides the system packages and
the PETSc build options, so pinning the Firedrake version pins the rest. The
steps below are the ones from [Firedrake's installation
guide](https://www.firedrakeproject.org/install.html), with the version fixed
at 2026.4.1 throughout. They are written for Ubuntu and need `sudo` for the
system packages; on any other distribution, add `--os unknown` to each
`firedrake-configure` call and PETSc will build its external packages from
source instead of linking against yours. Consult that guide if a step fails —
it carries a list of the problems people actually hit.

```bash
# 0. Choose an install root, and fetch firedrake-configure into it. Everything
#    below is run from this directory, which is what --show-env assumes.
export FIREDRAKE_DIR=~/firedrake
mkdir -p "$FIREDRAKE_DIR" && cd "$FIREDRAKE_DIR"
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/2026.4.1/scripts/firedrake-configure

# 1. System packages.
sudo apt install $(python3 firedrake-configure --show-system-packages)

# 2. PETSc, built from source. This is the long step.
git clone --branch $(python3 firedrake-configure --show-petsc-version) \
    https://gitlab.com/petsc/petsc.git
cd petsc
python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure
make PETSC_DIR="$PWD" PETSC_ARCH=arch-firedrake-default all
make check
cd ..

# 3. Firedrake, into a virtual environment.
python3 -m venv venv-firedrake
. venv-firedrake/bin/activate
pip cache purge
export $(python3 firedrake-configure --show-env)
pip install --no-binary h5py 'firedrake[check]==2026.4.1'

# 4. Check, and add the optional parallel factorization.
firedrake-check
pip install pypardiso
```

With that virtual environment active, come back to this directory and run the
experiments as above, without the `docker run` prefix:

```bash
./run_experiments.sh
```

## Versions behind the published results

| Component | Version |
|---|---|
| Firedrake | 2026.4.1 |
| PETSc, petsc4py | 3.25.0 |
| Python | 3.13.5 (Firedrake requires 3.10 or newer) |
| NumPy, SciPy | 2.5.2, 1.18.0 |
| pypardiso | 0.4.7 |
| OS | Debian GNU/Linux 13 (trixie), x86_64 |

The pinned container carries the same Firedrake 2026.4.1 and petsc4py 3.25.0,
which are the two components that fix the discretization and the linear algebra.
The rest of it is its own: Ubuntu 24.04, Python 3.12.3, NumPy 2.4.4, SciPy
1.17.1, and no `pypardiso`, so the direct solves inside it run serially through
SuperLU.
