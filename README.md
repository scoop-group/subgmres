# subgmres

GMRES that solves real or complex linear systems `A x = b` by minimizing the
residual in a norm you choose, and reports the norms of the residual's
*subvectors* — for instance the velocity and pressure blocks of a saddle-point
problem — as it goes.

This is the code accompanying

> Roland Herzog, Kirk M. Soodhalter,
> *A Unified View of Residual Norm Minimizing Krylov Subspace Methods.*
> <!-- publication details to follow -->

Two things distinguish it from a textbook GMRES.

**The norm is a choice, separate from the preconditioner.** Most GMRES
implementations minimize the standard ("Euclidean") norm of the residual, and
leave the preconditioner as the only handle on the iteration. Here the inner
product `M`, whose inverse measures the residual `||r||_{M^{-1}}`, and the
preconditioner `P` are independent arguments. `P` governs how the Krylov
subspaces are built, namely from `A P^{-1}`. On the other hand, `M` decides
which norm of the residual is minimized, namely the one induced by `M^{-1}`.
That lets you state a stopping criterion in a norm that means something for
your problem, and it makes the method behave identically when the data undergo
a change of basis.

**Residual subvectors are monitored separately.** Given a partition of the
unknowns and equations — velocity and pressure, say, in a fluid flow problem,
with the balance of forces and the incompressibility constraint that go with
them — `subgmres` tracks the norm of each block of the residual, induced by a
block-diagonal `M`, and can be told to converge each to its own tolerance in
addition to a joint one. The norms fall out of the Givens recursion at no extra
cost: no residual vector is formed to obtain them.

The `subgmres` solver exists in MATLAB/Octave and Python implementations. The
two take the same arguments and return the same quantities.

## What is here

```
matlab/            the solver (subgmres.m), its tests and its demos
python/            the same, as an importable package
experiments_paper/ the manuscript's numerical experiments
results_published/ the CSV files behind the manuscript's figures and tables
results/           where your own run of experiments_paper/ writes
CITATION.cff       citation metadata
LICENSE            MIT
```

## Getting started

There is nothing to build and nothing to install: both implementations are
plain source files with no dependencies beyond a working numerical stack.

```bash
git clone https://github.com/scoop-group/subgmres.git
cd subgmres
```

For Python, work from the `python/` directory (or put it on your `PYTHONPATH`);
`import subgmres` then finds the package. NumPy and SciPy are all it needs.

```python
import numpy as np
import scipy.sparse as sp
from subgmres import subgmres

n = 200
A = sp.diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format="csr")
b = A @ np.ones(n)

result = subgmres(A, b, rtol=1e-8)
print(result.flag, result.iter)      # 0 = converged, in 13 iterations
```

For MATLAB or Octave, work from the `matlab/` directory (or `addpath('matlab');` to put it on the path):

```matlab
n = 200;
A = spdiags(ones(n,1)*[-1 4 -1], -1:1, n, n);
b = A*ones(n,1);

[x, flag, ~, iter] = subgmres(A, b, [], 1e-8);
```

Both report flag 0 after 13 iterations, and agree on `x` to a few units in
the last place.

`A`, and the optional `P` and `M`, may each be given as a dense or sparse
matrix, as a function, or — in Python — as a `LinearOperator`. One asymmetry is
worth knowing before you go further: `A` is applied forward, but a *matrix* `P`
or `M` is solved against for you, while a *function* must return the inverse
action itself — `P \ x`, not `P * x`. Getting that backwards raises no error;
the iteration merely stops converging.

Everything beyond this call — a partition of the unknowns and a tolerance for
each block, a non-standard inner product, a preconditioner, restarts — is shown
in a demo you can run, on a problem chosen to make the point. What each demo
covers is listed in [`matlab/demos/README.md`](matlab/demos/README.md) and
[`python/demos/README.md`](python/demos/README.md).

## Demos

The demos are documentation of what `subgmres` can do; the tests further down
are the separate question of whether it does it. There are seven demos in each
language, on matching subjects, and they run in seconds:

```bash
python3 demos/run_all_demos.py                 # from python/
octave demos/run_all_demos.m                   # from matlab/
matlab -batch "run('demos/run_all_demos.m')"   # from matlab/
```

Start with `demo_basic`. Then `demo_subvector_monitoring` and
`demo_custom_inner_product`, which are the two that show what this solver does
that others do not.

An individual Python demo runs from `python/`, as `python3 demos/demo_basic.py`.
An individual MATLAB-language demo has to be run from inside `matlab/demos/`,
because several of them call helper files that live there.

## Reproducing the paper's experiments

See [`experiments_paper/INSTALL.md`](experiments_paper/INSTALL.md). In short:
get [Firedrake](https://www.firedrakeproject.org/), most simply as the pinned container, then

```bash
cd experiments_paper && ./run_experiments.sh
```

The mesh levels it runs by default are coarser than the published ones, so that
it finishes in reasonable time on an ordinary machine; `INSTALL.md` lists both
ranges.

The CSV files behind the manuscript's figures and tables are in
`results_published/`, and a run of your own goes to `results/`, so you can hold
the two against each other. Iteration counts should agree exactly; residual
norms may differ in the final digits, machines summing floating-point
numbers in their own order.

## Tests

The Python suite, from `python/`:

```bash
python3 run_all_tests.py
```

The MATLAB-language suite, from `matlab/`:

```bash
octave run_all_tests.m                 # Octave
matlab -batch run_all_tests            # MATLAB
```

Both exit non-zero on failure. They check relations that hold in theory: the
Arnoldi relation, the orthonormality of the three bases the method builds, the
Pythagorean identity relating the subvector norms to the total, the documented
defaults, and — the property the method exists for — that each iterate minimizes
the residual over the affine Krylov space it was drawn from.

## Requirements

| | |
|---|---|
| MATLAB implementation | MATLAB (tested on R2021a) or Octave (tested on 8.4.0) |
| Python implementation | NumPy and SciPy (tested on Python 3.12.3, NumPy 1.26.4, SciPy 1.11.4) |
| Python implementation, optional | [`pypardiso`](https://github.com/haasad/PyPardiso), `pip install pypardiso`. A real sparse `P` or `M` is factorized once; with `pypardiso` that factorization uses MKL PARDISO in parallel, otherwise SciPy's SuperLU, serially. |
| The paper's experiments | Firedrake; see [`experiments_paper/INSTALL.md`](experiments_paper/INSTALL.md) |

## Citing

If you use this code in scientific work, please cite the paper:

```bibtex
@ONLINE{HerzogSoodhalter:2026:1,
  AUTHOR = {Herzog, Roland and Soodhalter, Kirk M.},
  DATE = {2026},
  EPRINT = {XXXX.XXXXX},
  EPRINTTYPE = {arXiv},
  TITLE = {A Unified View of Residual Norm Minimizing Krylov Subspace Methods},
}
```

To cite the software itself rather than the method — a particular archived
release of this code — use its Zenodo record:

```bibtex
@SOFTWARE{HerzogSoodhalter:2026:2,
  AUTHOR = {Herzog, Roland and Soodhalter, Kirk M.},
  DATE = {2026},
  DOI = {10.5281/zenodo.XXXXXXX},
  TITLE = {subgmres. GMRES with residual subvector norms and a configurable inner product},
}
```

<!-- Two placeholders. EPRINT is filled in on arXiv submission; the Zenodo DOI
     once the first release is archived, here and in CITATION.cff. -->
`EPRINT` is filled in once the manuscript is on arXiv. The `DOI` is Zenodo's
concept DOI, which always resolves to the most recent release, so the entry does
not go stale when a later version is archived. `CITATION.cff` carries the same
information in machine-readable form.

## License

MIT, © Roland Herzog and Kirk M. Soodhalter. See `LICENSE`.
