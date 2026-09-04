# Demos (MATLAB / Octave)

Seven short scripts, one per subject, each self-contained and running in a few
seconds. They are documentation rather than verification: they show how the
solver is used and what its arguments mean.

The Python package has the same seven under `../../python/demos/`, on matching
subjects and with matching names.

| Demo | What it shows |
| --- | --- |
| `demo_basic` | The simplest call, `subgmres(A, b)`: no preconditioner, no partition, the standard norm. What the defaults actually were, what `flag` means, and how to read the outputs. Start here. |
| `demo_custom_inner_product` | A non-standard inner product `M`. The residual is minimized in the `M^{-1}`-norm rather than the standard norm. |
| `demo_Helmholtz` | A complex-valued one-dimensional Helmholtz problem, discretized by finite differences with a Sommerfeld radiation condition, solved with a preconditioner `P`: the complex-shifted Laplacian. |
| `demo_operator_forms` | The forms `A`, `P` and `M` may take: dense or sparse matrices, or a function handle. If a *matrix* is passed as `P` (or `M`), it will be factorized. If a *function* `PFUN` is passed instead, it must return the inverse action, `P \ x` rather than `P * x`. This demo also shows extra arguments being forwarded to `AFUN`, `PFUN` and `MFUN`. |
| `demo_residual_accuracy` | How far the monitored norms can be trusted. They come out of the Givens recursion at no cost, which is what makes subvector monitoring cheap, and are compared here against a genuine `b - A*x`. |
| `demo_restarts` | Restarted GMRES. `subgmres` has no restarting of its own and needs none: because canonical GMRES admits an arbitrary initial guess, re-invoking it every `m` steps with `x0` set to the previous iterate is the whole feature. Uses an absolute stopping criterion, which is more reasonable for a restart cycle. |
| `demo_subvector_monitoring` | The namesake feature, on a 2D Stokes saddle point on a staggered (MAC) grid, assembled here in a dozen lines. Velocity and pressure residuals are monitored separately, with their own tolerances, and the last part shows that the subvectors need not be contiguous. |

## Running them

All seven at once, from this directory's parent:

```bash
octave demos/run_all_demos.m                   # Octave
matlab -batch "run('demos/run_all_demos.m')"   # MATLAB
```

An individual demo has to be run **from inside this directory**, because each
one does `addpath('..')` to find `subgmres.m` and its own helper files:

```bash
cd demos
octave demo_basic.m
```

## The other files here

`run_all_demos.m` and `runDemo.m` are the runner. The remaining four are helpers
that the demos call, not demos themselves: `convectionDiffusion.m` builds the
system for `demo_restarts` and `helmholtzOperator.m` the one for
`demo_Helmholtz`, `restartedGMRES.m` is the restart loop of `demo_restarts`, and
`applyOperatorShifted.m` is the parameterized operator of `demo_operator_forms`.
