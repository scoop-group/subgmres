# -*- coding: utf-8 -*-
"""
Test suite for the Python port of subgmres.m.

Runnable with the standard library alone (`python -m unittest discover -s
src/python/subgmres/tests`); also auto-discovered by pytest if installed.
Correctness of the port itself was additionally cross-validated against
src/subgmres/subgmres.m directly (identical random complex data, with a
preconditioner, a block-diagonal inner product, and a subvector partition)
to ~1e-14 relative agreement on x, flag, iter, all history outputs, H, RH,
VHIST and ZHIST; see review-todos.md sect. 3D for the cross-validation
scenario, not itself checked in since it depends on Octave.

This suite covers the paper identities (mirroring test_subgmres.m's
paperIdentityTests) and the operator-coercion behaviour that has no MATLAB
analogue (LinearOperator/sparse/callable inputs), but does not attempt to
re-port every case from test_subgmres.m one for one.
"""

import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from .. import (
    subgmres,
    SubGMRESNotHermitianError,
    SubGMRESNotPositiveDefiniteError,
)


class BasicSolveTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def _check_matches_direct_solve(self, A, b, **kwargs):
        x_exact = np.linalg.solve(A, b)
        result = subgmres(A, b, rtol=1e-12, atol=0.0, maxiter=A.shape[0], **kwargs)
        self.assertLessEqual(
            np.linalg.norm(result.x - x_exact) / np.linalg.norm(x_exact), 1e-9
        )
        return result

    def test_real_dense(self):
        n = 25
        A = self.rng.standard_normal((n, n)) + 5 * np.eye(n)
        b = self.rng.standard_normal(n)
        self._check_matches_direct_solve(A, b)

    def test_complex_dense(self):
        n = 25
        A = (
            self.rng.standard_normal((n, n))
            + 1j * self.rng.standard_normal((n, n))
            + 5 * np.eye(n)
        )
        b = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        self._check_matches_direct_solve(A, b)

    def test_sparse_input(self):
        n = 25
        A = self.rng.standard_normal((n, n)) + 5 * np.eye(n)
        b = self.rng.standard_normal(n)
        x_exact = np.linalg.solve(A, b)
        result = subgmres(sp.csr_matrix(A), b, rtol=1e-12, atol=0.0, maxiter=n)
        self.assertLessEqual(np.linalg.norm(result.x - x_exact) / np.linalg.norm(x_exact), 1e-9)

    def test_callable_and_linear_operator_input(self):
        n = 25
        A = self.rng.standard_normal((n, n)) + 5 * np.eye(n)
        b = self.rng.standard_normal(n)
        x_exact = np.linalg.solve(A, b)

        result_callable = subgmres(lambda x: A @ x, b, rtol=1e-12, atol=0.0, maxiter=n)
        self.assertLessEqual(
            np.linalg.norm(result_callable.x - x_exact) / np.linalg.norm(x_exact), 1e-9
        )

        A_op = LinearOperator((n, n), matvec=lambda x: A @ x)
        result_op = subgmres(A_op, b, rtol=1e-12, atol=0.0, maxiter=n)
        self.assertLessEqual(
            np.linalg.norm(result_op.x - x_exact) / np.linalg.norm(x_exact), 1e-9
        )

    def test_preconditioner_and_inner_product_as_callables(self):
        # coerce_operator treats an array and a callable differently for P and
        # M: an array is the operator itself, factorized once and solved
        # against, while a callable is trusted to compute the inverse action
        # already. Every other call site in this suite passes arrays, so the
        # callable branch went untested here; test_subgmres.m covers both at
        # its varargin test.
        #
        # A is nearly diagonal with a spread of 1e6, so the diagonal is an
        # excellent preconditioner and the run finishes far short of the
        # Krylov dimension. That matters twice over: it keeps the comparison
        # away from the point where the monitored norms stop meaning anything,
        # and it leaves room for the wrong answer to look wrong.
        n = 60
        scale = np.geomspace(1.0, 1e6, n)
        A = np.diag(scale) + 0.01 * self.rng.standard_normal((n, n))
        b = self.rng.standard_normal(n)
        P = np.diag(scale)
        M = np.diag(np.geomspace(1e-2, 1e2, n))
        solve_P = lambda v: np.linalg.solve(P, v)
        solve_M = lambda v: np.linalg.solve(M, v)
        options = dict(rtol=1e-10, maxiter=40)

        as_arrays = subgmres(A, b, P=P, M=M, **options)
        self.assertEqual(as_arrays.flag, 0)
        self.assertLess(as_arrays.iter, 40)          # converged, not exhausted

        # Both callable forms must reproduce it exactly.
        as_callables = subgmres(A, b, P=solve_P, M=solve_M, **options)
        as_operators = subgmres(
            A, b, P=LinearOperator((n, n), matvec=solve_P),
            M=LinearOperator((n, n), matvec=solve_M), **options)
        for other in (as_callables, as_operators):
            self.assertEqual(other.iter, as_arrays.iter)
            self.assertEqual(other.flag, as_arrays.flag)
            self.assertTrue(np.allclose(other.x, as_arrays.x,
                                        rtol=1e-9, atol=1e-12))

        # And the trap the distinction exists for: a callable supplying the
        # FORWARD action is a different preconditioner, not a mis-scaled one,
        # and nothing complains -- the iteration simply stops converging. If
        # this ever starts matching, the two branches have been conflated.
        forward = subgmres(A, b, P=lambda v: P @ v, M=M, **options)
        self.assertEqual(forward.flag, 1)
        self.assertFalse(np.allclose(forward.x, as_arrays.x,
                                     rtol=1e-3, atol=1e-6))

    def test_defaults_are_what_they_are_documented_to_be(self):
        # Each default asserted against its documented value, rather than
        # inferred from a run that happens to succeed. n exceeds 100 so that
        # the maxiter default is a real cap and not simply n.
        n = 120
        A = self.rng.standard_normal((n, n)) + 3 * n**0.5 * np.eye(n)
        b = self.rng.standard_normal(n)

        # rtol defaults to 1e-6 on the total residual only, atol to inf.
        defaults = subgmres(A, b)
        self.assertEqual(defaults.tol_table[-1, 0], 1e-6)
        self.assertTrue(np.all(np.isinf(defaults.tol_table[:-1, 0])))
        self.assertTrue(np.all(np.isinf(defaults.tol_table[:, 1])))

        # x0 defaults to the zero vector: passing it explicitly changes
        # nothing at all.
        explicit = subgmres(A, b, x0=np.zeros(n))
        self.assertEqual(explicit.iter, defaults.iter)
        self.assertTrue(np.array_equal(explicit.x, defaults.x))

        # maxiter defaults to min(n, 100). Given a tolerance it cannot meet,
        # the run must stop at exactly that many iterations.
        capped = subgmres(A, b, rtol=0.0)
        self.assertEqual(capped.iter, 100)
        self.assertEqual(capped.flag, 1)
        self.assertEqual(subgmres(A, b, rtol=0.0, maxiter=17).iter, 17)

    def test_r_hist_reconstructs_residuals_without_x_hist(self):
        # Regression: compute_r_hist must reconstruct the iterate x on its
        # own, not rely on compute_x_hist having done so. Previously,
        # requesting r_hist alone left every column equal to the initial
        # residual (x stayed at x0), which silently corrupted the residual
        # history used by the experiment drivers.
        n = 20
        A = self.rng.standard_normal((n, n)) + 5 * np.eye(n)
        b = self.rng.standard_normal(n)
        # Reference with BOTH histories on: here x is reconstructed for
        # x_hist, so r_hist is trustworthy.
        ref = subgmres(A, b, rtol=1e-10, atol=0.0, maxiter=n,
            compute_x_hist=True, compute_r_hist=True)
        # r_hist ALONE must produce the same residual history.
        res = subgmres(A, b, rtol=1e-10, atol=0.0, maxiter=n, compute_r_hist=True)
        self.assertTrue(np.allclose(res.r_hist, ref.r_hist, atol=1e-10))
        # Each column must be the true residual b - A x_k for the k-th iterate.
        true_R = b[:, None] - A @ ref.x_hist
        self.assertTrue(np.allclose(res.r_hist, true_R, atol=1e-10))
        # The bug's signature: the last column must NOT still equal the
        # initial residual.
        self.assertFalse(np.allclose(res.r_hist[:, -1], res.r_hist[:, 0]))

    def test_zero_initial_subvector_residual_does_not_block_stopping(self):
        # Regression: a subvector whose initial residual is exactly zero, with
        # a scalar rtol, leaves that entry's relative tolerance as inf*0 = nan.
        # numpy propagates nan through minimum where MATLAB's min drops it, so
        # `eta <= nan` was False forever: the iteration could not stop, ran
        # until the Krylov space was exhausted, and the triangular solve raised
        # LinAlgError. subgmres.m converged on the same input. The case is not
        # exotic -- a Stokes system with no source in the continuity equation
        # starts with an exactly zero pressure residual.
        n, n1 = 12, 8
        A = self.rng.standard_normal((n, n)) + 4 * np.eye(n)
        b = np.concatenate([self.rng.standard_normal(n1), np.zeros(n - n1)])
        res = subgmres(A, b, ix=[0, n1], rtol=1e-8, maxiter=200)
        self.assertEqual(res.flag, 0)
        self.assertLessEqual(res.iter, n)
        self.assertLessEqual(
            np.linalg.norm(b - A @ res.x) / np.linalg.norm(b), 1e-8)

    def test_absolute_tolerance_survives_on_a_zero_initial_subvector(self):
        # Only the vacuous relative requirement is dropped, never the whole
        # test: an absolute tolerance placed on that same subvector must still
        # bind. Its residual starts at exactly zero but does not stay there, so
        # an unreachable atol there alone has to prevent convergence.
        n, n1 = 40, 25
        A = self.rng.standard_normal((n, n)) + 3 * n**0.5 * np.eye(n)
        b = np.concatenate([self.rng.standard_normal(n1), np.zeros(n - n1)])
        loose = subgmres(A, b, ix=[0, n1], rtol=1e-8, maxiter=30,
                         compute_sub_norms_hist=True)
        self.assertEqual(loose.flag, 0)
        # It really does start at zero and really does leave it.
        self.assertEqual(loose.sub_norms_hist[1, 0], 0.0)
        self.assertGreater(loose.sub_norms_hist[1, 1], 0.0)
        # Same run, an unreachable atol on that subvector and nothing else.
        bound = subgmres(A, b, ix=[0, n1], rtol=1e-8,
                         atol=[np.inf, 1e-30, np.inf], maxiter=30)
        self.assertEqual(bound.flag, 1)
        self.assertEqual(bound.tol_table[1, 1], 1e-30)

    def test_real_inner_product_with_complex_system(self):
        # Regression: a real inner-product matrix M applied to the complex
        # residual of a complex system (the Helmholtz configuration) must
        # work -- the real factorization is applied to real and imaginary
        # parts separately. Previously scipy refused the complex-to-real
        # cast and raised.
        n = 24
        A = (self.rng.standard_normal((n, n)) + 1j * self.rng.standard_normal((n, n))
            + 5 * np.eye(n))
        b = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        # Real, symmetric positive definite inner product.
        Mroot = self.rng.standard_normal((n, n))
        M = Mroot @ Mroot.T + n * np.eye(n)
        x_exact = np.linalg.solve(A, b)
        result = subgmres(A, b, rtol=1e-11, atol=0.0, maxiter=n, M=M)
        self.assertLessEqual(np.linalg.norm(result.x - x_exact) / np.linalg.norm(x_exact), 1e-8)


class PaperIdentityTests(unittest.TestCase):
    """Mirrors test_subgmres.m's paperIdentityTests, checking the identities
    of section "Arnoldi Process" of content.tex: the Arnoldi relation
    eq:Arnoldi-relation, the Hessenberg entries
    eq:upper-Hessenberg-matrix-entries, the M^{-1}-orthonormality of the dual
    basis V, Z's relation to V through P
    (eq:relation-primal-and-dual-Krylov-bases) and the P^H M^{-1} P-
    orthonormality of Z (lemma:orthonormal-basis-of-primal-Krylov-subspace).

    All inner products follow the paper's convention: <r,s>_{M^{-1}} =
    r^H M^{-1} s, antilinear in the *first* argument."""

    def test_identities_with_preconditioner_and_inner_product(self):
        rng = np.random.default_rng(2)
        n = 16

        def rand_complex(n):
            return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

        A = rand_complex(n) + 4 * np.eye(n)
        P = rand_complex(n) + 4 * np.eye(n)
        # M must be Hermitian positive definite (no subvectors here, so no
        # block-diagonal requirement).
        Mraw = rand_complex(n)
        M = Mraw @ Mraw.conj().T + n * np.eye(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        x_exact = np.linalg.solve(A, b)
        result = subgmres(
            A, b, rtol=1e-11, atol=0.0, maxiter=n, P=P, M=M, x0=x0,
            compute_v_hist=True, compute_w_hist=True, compute_z_hist=True,
        )
        k = result.iter
        V = result.v_hist[:, :k]
        V_full = result.v_hist[:, : k + 1]  # includes v_{k+1}, needed for the Arnoldi relation below
        Z = result.z_hist[:, :k]
        H = result.h

        self.assertLess(np.linalg.norm(result.x - x_exact) / np.linalg.norm(x_exact), 1e-9)

        # Z = P^{-1} V (subgmres.m's `z_j = P \ v_j`).
        PinvV = np.linalg.solve(P, V)
        self.assertLessEqual(
            np.linalg.norm(Z - PinvV) / max(1.0, np.linalg.norm(PinvV)), 1e-9
        )

        # V is orthonormal w.r.t. the inner product <r,s>_{M^{-1}} = r^H
        # M^{-1} s of the antidual space. The conjugate belongs on the
        # *first* argument, i.e. outside the solve with M; conj(M^{-1} v)
        # and M^{-1} conj(v) differ whenever M is genuinely complex.
        Minv_V = np.linalg.solve(M, V)
        gram_V = V.conj().T @ Minv_V
        self.assertLessEqual(np.linalg.norm(gram_V - np.eye(k)), 1e-8)

        # W = M^{-1} V are the primal companions
        # (eq:primal-companions-of-the-dual-Krylov-basis); by the second
        # statement of lemma:orthonormal-basis-of-primal-Krylov-subspace they
        # are M-orthonormal. They are a different basis from Z: the two
        # coincide only when P == M.
        Wc = result.w_hist[:, :k]
        self.assertLessEqual(
            np.linalg.norm(Wc - np.linalg.solve(M, V)) / max(1.0, np.linalg.norm(Wc)), 1e-9
        )
        self.assertLessEqual(np.linalg.norm(Wc.conj().T @ M @ Wc - np.eye(k)), 1e-8)
        self.assertGreater(np.linalg.norm(Wc - Z), 1e-3)

        # Z is orthonormal w.r.t. P^H M^{-1} P -- the Hermitian inverse
        # throughout, matching lemma:orthonormal-basis-of-primal-Krylov-subspace.
        N = P.conj().T @ np.linalg.solve(M, P)
        gram_Z = Z.conj().T @ N @ Z
        self.assertLessEqual(np.linalg.norm(gram_Z - np.eye(k)), 1e-8)

        # Arnoldi relation A P^{-1} V_k = V_{k+1} H (the full (k+1)-row H,
        # subdiagonal entry included, matched against the (k+1)-column V).
        AP_inv_V = A @ np.linalg.solve(P, V)
        self.assertLessEqual(
            np.linalg.norm(AP_inv_V - V_full @ H) / max(1.0, np.linalg.norm(AP_inv_V)),
            1e-8,
        )

        # Hessenberg entries h_{i,j} = <v_i, A P^{-1} v_j>_{M^{-1}}
        # (eq:upper-Hessenberg-matrix-entries). The argument order matters:
        # the opposite one would yield the entrywise conjugate of H.
        Minv_AP_inv_V = np.linalg.solve(M, AP_inv_V)
        entries = V_full.conj().T @ Minv_AP_inv_V
        # Drop the last row after a happy breakdown: h_{k+1,k} is then O(eps)
        # and v_{k+1} = v/h_{k+1,k} is amplified roundoff noise, so that row
        # of the Gram matrix carries no information (it is still normalized,
        # so a test on the diagonal of the Gram matrix would not catch it).
        breakdown = abs(H[k, k - 1]) <= 1e-10 * np.linalg.norm(H)
        rows = np.arange(k + 1) < (k if breakdown else k + 1)
        self.assertLessEqual(
            np.linalg.norm(H[rows] - entries[rows]) / max(1.0, np.linalg.norm(H[rows])),
            1e-8,
        )
        # Guard against the conjugated variant passing by accident, which it
        # would for real data; with a complex M the two must differ.
        self.assertGreater(np.linalg.norm(H[rows] - entries[rows].conj()), 1e-3)


class SubvectorTests(unittest.TestCase):
    def test_subvector_tolerances_are_all_met(self):
        rng = np.random.default_rng(3)
        n = 24
        A = rng.standard_normal((n, n)) + 6 * np.eye(n)
        b = rng.standard_normal(n)
        ix = [0, 8, 16]  # three contiguous blocks of size 8
        blocks = [slice(0, 8), slice(8, 16), slice(16, 24)]
        rtol = 1e-9

        # x0 defaults to zero, so the initial residual is b itself, and with
        # no inner product M the relevant norm is the plain standard one:
        # convergence requires each block's residual to have shrunk by rtol
        # relative to that block of b. Note that atol=0 does not disable the
        # absolute check -- it is the strictest setting there is, since the
        # effective tolerance is min(rtol*eta0, atol); atol=inf, the default,
        # is what disables it. That is why this run cannot stop on tolerance
        # at all and the residual is checked directly below instead.
        # Not asserting flag == 0: with maxiter == n exactly, the Arnoldi
        # process is pushed one step past the Krylov space's actual
        # dimension on its last iteration, which can leave flag == 1 (see
        # test_x0_and_maxiter_defaults above) even though x itself is
        # accurate -- checked directly below instead.
        result = subgmres(A, b, ix=ix, rtol=rtol, atol=0.0, maxiter=n)
        self.assertEqual(len(result.ix), 3)
        self.assertEqual(result.sub_norms.shape, (4,))  # 3 subvectors + total

        r = b - A @ result.x
        for block in blocks:
            self.assertLessEqual(
                np.linalg.norm(r[block]), 10 * rtol * np.linalg.norm(b[block])
            )


class NonContiguousPartitionTests(unittest.TestCase):
    """Subvector partitions given as explicit index sets, which need not be
    contiguous. The starting-index form of ix implies contiguity; the index-set
    form does not, and an assembly numbering unknowns by node rather than by
    field leaves each field scattered through the vector. Until these tests
    only demo_subvector_monitoring exercised that, which is the wrong place for
    a correctness property."""

    def setUp(self):
        # A two-block problem built contiguously, then interleaved, so the
        # right answer is known independently: the scattered solve has to
        # reproduce the contiguous one index for index.
        #
        # Well conditioned and comfortably larger than the iteration count it
        # needs. On a system small enough to be solved by exhausting the Krylov
        # space, every run ends at the point where the monitored norms have
        # parted company with the true ones (see demo_residual_accuracy), and
        # no comparison against directly computed norms would mean anything.
        rng = np.random.default_rng(17)
        self.n, self.block = 40, 20
        n, block = self.n, self.block
        self.A = rng.standard_normal((n, n)) + 3 * n**0.5 * np.eye(n)
        self.b = rng.standard_normal(n)
        self.P = rng.standard_normal((n, n)) + 3 * n**0.5 * np.eye(n)
        # M must not couple the two blocks, or the subvector norms would not
        # split; block-diagonal in the contiguous ordering guarantees that, and
        # permuting preserves it with respect to the permuted sets.
        self.M = np.zeros((n, n))
        for start in (0, block):
            root = rng.standard_normal((block, block))
            self.M[start:start + block, start:start + block] = (
                root @ root.T + block * np.eye(block))
        # Even positions take the first block, odd positions the second.
        self.order = np.empty(n, dtype=int)
        self.order[0::2] = np.arange(block)
        self.order[1::2] = np.arange(block, n)
        self.sets = [np.flatnonzero(self.order < block),
                     np.flatnonzero(self.order >= block)]
        # Neither set is an interval, which is the whole point.
        self.assertTrue(np.any(np.diff(self.sets[0]) != 1))
        self.assertTrue(np.any(np.diff(self.sets[1]) != 1))

    def _permuted(self, matrix):
        return matrix[self.order][:, self.order]

    def _block_norm(self, vector, indices):
        block = vector[indices]
        M_block = self._permuted(self.M)[np.ix_(indices, indices)]
        return np.sqrt(np.real(block @ np.linalg.solve(M_block, block)))

    def test_scattered_partition_matches_the_contiguous_one(self):
        # Relabelling the unknowns must change nothing but the labels.
        contiguous = subgmres(self.A, self.b, ix=[0, self.block], rtol=1e-8,
                              maxiter=self.n, P=self.P, M=self.M)
        scattered = subgmres(self._permuted(self.A), self.b[self.order],
                             ix=self.sets, rtol=1e-8, maxiter=self.n,
                             P=self._permuted(self.P), M=self._permuted(self.M))
        self.assertEqual(scattered.iter, contiguous.iter)
        self.assertEqual(scattered.flag, contiguous.flag)
        self.assertLess(scattered.iter, self.n)     # not by exhaustion
        self.assertTrue(np.allclose(scattered.sub_norms, contiguous.sub_norms,
                                    rtol=1e-8, atol=1e-14))
        self.assertTrue(np.allclose(scattered.x, contiguous.x[self.order],
                                    rtol=1e-8, atol=1e-14))

    def test_scattered_subvector_norms_are_the_true_ones(self):
        # The reported norms must be the M-inverse norms of the residual
        # restricted to each scattered set -- an indexing claim, and the one a
        # partition that is not an interval could most easily get wrong.
        A, b = self._permuted(self.A), self.b[self.order]
        result = subgmres(A, b, ix=self.sets, rtol=1e-8, maxiter=self.n,
                          P=self._permuted(self.P), M=self._permuted(self.M))
        self.assertEqual(result.flag, 0)
        residual = b - A @ result.x
        for i, indices in enumerate(self.sets):
            self.assertAlmostEqual(
                result.sub_norms[i] / self._block_norm(residual, indices),
                1.0, places=6)

    def test_per_subvector_tolerances_on_scattered_sets(self):
        # The stopping test must apply each tolerance to its own scattered set.
        A, b = self._permuted(self.A), self.b[self.order]
        rtol = [1e-10, 1e-4, np.inf]
        result = subgmres(A, b, ix=self.sets, rtol=rtol, maxiter=self.n,
                          P=self._permuted(self.P), M=self._permuted(self.M))
        self.assertEqual(result.flag, 0)
        for i, indices in enumerate(self.sets):
            self.assertLessEqual(result.sub_norms[i],
                                 rtol[i] * self._block_norm(b, indices))


class OptimalityTests(unittest.TestCase):
    """The property the method exists for: the k-th iterate minimizes the
    M-inverse norm of the residual over the affine space
    x_0 + P^{-1} K_k(A P^{-1}, r_0).

    Everything else in this suite checks the Arnoldi machinery -- the Arnoldi
    relation, the orthonormality of the bases, the bookkeeping of the norms --
    and none of it checks that the machinery solves the minimization it was
    built for. A subgmres that assembled every basis correctly and then solved
    the wrong least-squares problem would pass all of those and fail these."""

    def test_iterate_minimizes_the_residual_over_the_krylov_space(self):
        rng = np.random.default_rng(2)
        n = 40

        def rand_complex():
            return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

        # Complex, preconditioned, non-standard inner product and a nonzero
        # initial guess, so that no term of the statement is trivially absent.
        # P is a decent preconditioner, which makes the run stop well short of
        # the Krylov dimension; at exhaustion the residual is machine zero and
        # the orthogonality below would be measured against nothing.
        A = rand_complex() + 8 * np.eye(n)
        P = A + 0.6 * rand_complex()
        Mroot = rand_complex()
        M = Mroot @ Mroot.conj().T + n * np.eye(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        result = subgmres(A, b, rtol=1e-8, maxiter=n, P=P, M=M, x0=x0,
                          compute_x_hist=True, compute_r_hist=True,
                          compute_z_hist=True)
        self.assertEqual(result.flag, 0)
        self.assertLess(result.iter, n)
        Z = result.z_hist

        def m_inverse_norm(v):
            return np.sqrt(np.real(np.vdot(v, np.linalg.solve(M, v))))

        initial = m_inverse_norm(result.r_hist[:, 0])

        for k in range(1, result.iter + 1):
            residual = result.r_hist[:, k]
            AZ = A @ Z[:, :k]
            # First-order optimality: the residual is M-inverse-orthogonal to
            # A P^{-1} K_k, the space the correction was drawn from. This is
            # the minimization written as a Galerkin condition, and it is an
            # identity rather than an approximation.
            galerkin = np.linalg.norm(AZ.conj().T @ np.linalg.solve(M, residual))
            self.assertLess(galerkin / initial, 1e-9)

            # The same statement in its literal form: no other point of the
            # affine space does better. Checked against random displacements
            # at two magnitudes, since a first-order condition alone would
            # also hold at a maximum or a saddle.
            optimal = m_inverse_norm(residual)
            for direction in (rng.standard_normal((3, k))
                              + 1j * rng.standard_normal((3, k))):
                for step in (1e-1, 1e-3):
                    moved = result.x_hist[:, k] + Z[:, :k] @ (step * direction)
                    self.assertGreaterEqual(
                        m_inverse_norm(b - A @ moved), optimal)


class StructuralInvariantTests(unittest.TestCase):
    """Structural invariants of the outputs, mirroring test_subgmres.m's
    standardTests: history shapes, the Pythagorean subvector-norm identity,
    monotonicity of the total residual norm, agreement of the final column
    with sub_norms, and the (quasi-)triangular structure of H and RH."""

    def test_invariants_with_preconditioner_inner_product_and_subvectors(self):
        rng = np.random.default_rng(11)
        n = 30
        block_sizes = [10, 10, 10]
        A = rng.standard_normal((n, n)) + 6 * np.eye(n)
        b = rng.standard_normal(n)
        P = rng.standard_normal((n, n)) + 6 * np.eye(n)
        # M must be block-diagonal w.r.t. the subvector partition and SPD, so
        # the M-inverse norm splits across blocks (Pythagorean identity).
        M = np.zeros((n, n))
        off = 0
        for size in block_sizes:
            root = rng.standard_normal((size, size))
            M[off:off + size, off:off + size] = root @ root.T + size * np.eye(size)
            off += size
        ix = [0, 10, 20]

        result = subgmres(
            A, b, ix=ix, rtol=1e-10, atol=0.0, maxiter=n, P=P, M=M,
            compute_x_hist=True, compute_r_hist=True,
            compute_sub_norms_hist=True, compute_v_hist=True,
        )
        k = result.iter
        n_sub = len(block_sizes)

        # History shapes: n_sub + 1 rows (subvectors + total), k + 1 columns.
        self.assertEqual(result.sub_norms_hist.shape, (n_sub + 1, k + 1))
        self.assertEqual(result.x_hist.shape, (n, k + 1))
        self.assertEqual(result.r_hist.shape, (n, k + 1))
        self.assertEqual(result.v_hist.shape, (n, k + 1))

        # Final column of the history equals the returned sub_norms.
        self.assertTrue(np.allclose(result.sub_norms, result.sub_norms_hist[:, -1]))

        # Pythagorean identity: the squared subvector norms sum to the squared
        # total norm, column by column.
        sub = result.sub_norms_hist[:-1, :]
        total = result.sub_norms_hist[-1, :]
        self.assertTrue(np.allclose(np.sum(sub**2, axis=0), total**2, atol=1e-8))

        # The total residual norm is monotonically non-increasing.
        self.assertTrue(np.all(np.diff(total) <= 1e-12))

        # H is upper Hessenberg (nothing below the first subdiagonal) and RH
        # is upper triangular (nothing below the diagonal).
        self.assertTrue(np.allclose(np.tril(result.h, -2), 0.0))
        self.assertTrue(np.allclose(np.tril(result.rh, -1), 0.0))

        # The progressive subvector norms match the true residual subvector
        # M-inverse-norms recomputed from the residual history.
        for i, (lo, size) in enumerate(zip([0, 10, 20], block_sizes)):
            block = slice(lo, lo + size)
            Mblock = M[block, block]
            Rblock = result.r_hist[block, :]
            true_sq = np.real(np.sum(np.conj(Rblock) * np.linalg.solve(Mblock, Rblock), axis=0))
            self.assertTrue(np.allclose(np.sqrt(np.maximum(true_sq, 0)),
                result.sub_norms_hist[i, :], atol=1e-7))


class ErrorHandlingTests(unittest.TestCase):
    def test_non_hermitian_inner_product_raises(self):
        rng = np.random.default_rng(4)
        n = 10
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)) + 5 * np.eye(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)) + 5 * np.eye(n)  # not Hermitian
        with self.assertRaises(SubGMRESNotHermitianError):
            subgmres(A, b, M=M, maxiter=3)

    def test_non_positive_definite_inner_product_raises(self):
        rng = np.random.default_rng(5)
        n = 10
        A = rng.standard_normal((n, n)) + 5 * np.eye(n)
        b = rng.standard_normal(n)
        M = np.diag(rng.standard_normal(n))  # symmetric but indefinite
        M = 0.5 * (M + M.T)
        with self.assertRaises((SubGMRESNotPositiveDefiniteError, SubGMRESNotHermitianError)):
            subgmres(A, b, M=M, maxiter=3)


if __name__ == "__main__":
    unittest.main()
