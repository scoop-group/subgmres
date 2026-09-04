# Demos (Python)

Seven short scripts, one per subject, each self-contained and running in a few
seconds. They are documentation rather than verification: they show how the
solver is used and what its arguments mean.

The MATLAB package has the same seven under `../../matlab/demos/`, on matching
subjects and with matching names.

| Demo | What it shows |
| --- | --- |
| `demo_basic` | The simplest call, `subgmres(A, b)`: no preconditioner, no partition, the standard norm. What the defaults actually were, what `flag` means, and how to read the result object. Start here. |
| `demo_custom_inner_product` | A non-standard inner product `M`. The residual is minimized in the `M^{-1}`-norm rather than the standard norm. |
| `demo_Helmholtz` | A complex-valued one-dimensional Helmholtz problem, discretized by finite differences with a Sommerfeld radiation condition, solved with a preconditioner `P`: the complex-shifted Laplacian. |
| `demo_operator_forms` | The forms `A`, `P` and `M` may take: dense arrays, sparse matrices, a `LinearOperator`, or a plain callable. If a *matrix* is passed as `P` (or `M`), it will be factorized. If a *callable* is passed instead, it must return the inverse action, `P \ x` rather than `P @ x`. This demo also shows a parameterized operator carried by `functools.partial`. |
| `demo_residual_accuracy` | How far the monitored norms can be trusted. They come out of the Givens recursion at no cost, which is what makes subvector monitoring cheap, and are compared here against a genuine `b - A @ x`. |
| `demo_restarts` | Restarted GMRES. `subgmres` has no restarting of its own and needs none: because canonical GMRES admits an arbitrary initial guess, re-invoking it every `m` steps with `x0` set to the previous iterate is the whole feature. Uses an absolute stopping criterion, which is more reasonable for a restart cycle. |
| `demo_subvector_monitoring` | The namesake feature, on a 2D Stokes saddle point on a staggered (MAC) grid, assembled here in a dozen lines. Velocity and pressure residuals are monitored separately, with their own tolerances, and the last part shows that the subvectors need not be contiguous. |

## Running them

All seven at once, from this directory's parent:

```bash
python3 demos/run_all_demos.py
```

Or any single one, from the same place:

```bash
python3 demos/demo_basic.py
```

Only NumPy and SciPy are needed.
