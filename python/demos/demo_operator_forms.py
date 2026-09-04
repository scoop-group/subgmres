# -*- coding: utf-8 -*-
"""
The forms A, P and M may take, and what subgmres assumes about each.

Any of the three may be handed over as a dense array, a sparse matrix, a
scipy LinearOperator, or a plain callable. The one thing worth knowing is
that a matrix and a function do not mean the same thing for P and M:

  * as a matrix, P (or M) is the operator itself, and subgmres factorizes it
    once and solves against it -- what the iteration needs is the inverse
    action, and it obtains that for you;
  * as a function, you supply that inverse action yourself. The callable
    must return P \\ x, not P @ x.

Getting this backwards is a quiet mistake: the iteration still runs, and
merely fails to converge, so the demo checks the two forms against each
other rather than asserting that either looks plausible on its own.

The system is a one-dimensional convection-diffusion operator with a shift,
which gives the last part something to pass around: a parameter the operators
depend on. MATLAB's subgmres forwards trailing arguments to AFUN, PFUN and
MFUN for this; Python has no such mechanism and needs none, a closure or
functools.partial carrying the parameter instead.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from subgmres import subgmres

n = 400
h = 1.0 / (n + 1)
wind = 4.0
shift = 0.5
rtol = 1e-10

# The operator, as a sparse matrix: -u'' + wind*u' + shift*u.
diffusion = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)],
                     [-1, 0, 1]) / h**2
convection = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [-1, 1]) * (wind / (2 * h))
A_matrix = (diffusion + convection + shift * sp.identity(n)).tocsc()
b = np.ones(n)
relative_residual = lambda x: np.linalg.norm(b - A_matrix @ x) / np.linalg.norm(b)

## 1. A as a sparse matrix, as a callable, and as a LinearOperator.
# The callable never assembles anything: it applies the stencil in place.
def apply_operator(x, shift):
	y = (2 / h**2 + shift) * x
	y[:-1] += (-1 / h**2 + wind / (2 * h)) * x[1:]     # the super-diagonal
	y[1:] += (-1 / h**2 - wind / (2 * h)) * x[:-1]     # the sub-diagonal
	return y

forms = {
	"sparse matrix": A_matrix,
	"callable": lambda x: apply_operator(x, shift),
	"LinearOperator": spla.LinearOperator(
		(n, n), matvec=lambda x: apply_operator(x, shift)),
}
print("1. the same A in three forms, preconditioned identically")
reference = None
for name, A in forms.items():
	result = subgmres(A, b, rtol=rtol, maxiter=n, P=diffusion.tocsc())
	reference = result.x if reference is None else reference
	print(f"   {name:>16}: {result.iter:3d} iterations, flag {result.flag},"
	      f" ||r||/||b|| = {relative_residual(result.x):.2e},"
	      f" agrees to {np.linalg.norm(result.x - reference):.1e}")

## 2. P and M as matrices, and as functions supplying the inverse action.
# The diffusion part preconditions this operator well. Handed over as a matrix
# it is factorized internally; handed over as a function it must arrive already
# inverted, which is what the prefactorized solve below does. The weighting M
# is diagonal, so its inverse action is a division.
weights = np.geomspace(1e-2, 1e2, n)
M_matrix = sp.diags(weights).tocsc()
diffusion_solve = spla.factorized(diffusion.tocsc())
print("\n2. P and M as matrices, then as functions returning the inverse action")
as_matrices = subgmres(A_matrix, b, rtol=rtol, maxiter=n,
                       P=diffusion.tocsc(), M=M_matrix)
as_functions = subgmres(A_matrix, b, rtol=rtol, maxiter=n,
                        P=diffusion_solve, M=lambda x: x / weights)
for name, result in (("as matrices ", as_matrices), ("as functions", as_functions)):
	print(f"   {name}: {result.iter:3d} iterations, flag {result.flag},"
	      f" ||r||/||b|| = {relative_residual(result.x):.2e}")
print(f"   the two agree to {np.linalg.norm(as_matrices.x - as_functions.x):.1e}")

## 3. Operators that depend on a parameter.
# subgmres.m forwards any trailing arguments to AFUN, PFUN and MFUN, so that a
# parameterized operator needs no wrapper -- and every function-valued operator
# must then accept them, whether or not it uses them. In Python the same job is
# done by a closure, as above, or by functools.partial, and nothing is
# forwarded. The result is the same solve either way.
from functools import partial

parameterized = subgmres(partial(apply_operator, shift=shift), b,
                         rtol=rtol, maxiter=n, P=diffusion_solve)
plain = subgmres(A_matrix, b, rtol=rtol, maxiter=n, P=diffusion.tocsc())
print("\n3. a parameterized operator, the shift carried by functools.partial")
print(f"   partial(apply_operator, shift={shift}): {parameterized.iter} iterations,"
      f" flag {parameterized.flag}")
print(f"   the assembled matrix:                  {plain.iter} iterations,"
      f" flag {plain.flag}")
print(f"   the two agree to {np.linalg.norm(parameterized.x - plain.x):.1e}")
