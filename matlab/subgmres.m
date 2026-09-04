function [x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,WHIST,ZHIST,H,RH,toltable,ix] = subgmres(A,b,ix,rtol,atol,maxiter,P,M,x0,varargin)
% SUBGMRES   Generalized Minimum Residual Method for Matlab.
%   This code implements the modified version of GMRES with residual
%   subvector norms as described in
%
%     Herzog, Soodhalter: A Unified View of Residual Norm Minimizing Krylov
%	    Subspace Methods (in preparation)
%
%   Implemented by Roland Herzog, from a first implementation and the
%   subvector-monitoring derivation of Kirk M. Soodhalter.
%   Released under the MIT License; see LICENSE.
%   If this code is used in a scientific publication, please cite as
%
%     Roland Herzog, Kirk M. Soodhalter:
%     A Unified View of Residual Norm Minimizing Krylov Subspace Methods
%
%   X = SUBGMRES(A,B) attempts to find a solution to the system of linear
%   equations A*X=B. The N-by-N coefficient matrix A must be non-singular.
%   The right hand side must be a column vector B of length N.
%
%   X = SUBGMRES(AFUN,B) accepts a function handle AFUN instead of
%   the matrix A. AFUN(X) accepts a vector input X and returns the matrix-
%   vector product A*X. In all of the following, you may replace
%   A by AFUN.
%
%   X = SUBGMRES(AFUN,B,IX) specifies the subvector indices.
%   IX can be either a vector or a cell array. If IX is a vector,
%   then it must contain ascending entries which are interpreted as
%   the starting indices of the subvectors. If IX is a cell array,
%   then each entry must contain the indices for one of the subvectors.
%   Note that the inner product matrix M, when given, must be compatible
%   with the parititioning into subvectors.
%
%   X = SUBGMRES(A,B,IX,RTOL) specifies the relative tolerance of the
%   method. If RTOL is [], then SUBGMRES uses the default 1e-6 as a
%   relative stopping criterion for the total residual norm. The norm
%   induced by the inverse of the inner product M (see below) is used. If
%   no inner product is specified, the standard norm is employed. If RTOL
%   is a single number, then this is interpreted as relative stopping
%   criterion for the total residual norm. If RTOL is a vector of length
%   equal to the number of subvectors, then its entries serve componentwise
%   as relative stopping criteria for the individual subvector norms, all
%   of which need to be satisfied. If RTOL is a vector of length equal to
%   the number of subvectors plus one, then in addition to the above the
%   final entry serves as a relative stopping criterion for the total
%   residual norm.
%
%   X = SUBGMRES(A,B,IX,RTOL,ATOL) specifies the absolute
%   tolerance of the method. The same options as for RTOL apply.
%   The default is inf.
%
%   The iteration will stop as soon as the norms of the each subvector
%   as well as the norm of the full residual vector meet their
%   respective relative and absolute tolerances simultaneously.
%
%   X = SUBGMRES(A,B,IX,RTOL,ATOL,MAXITER) specifies the maximum
%   number of iterations. If MAXITER is [] then SUBGMRES uses the
%   default, round(min(N,100)).
%
%   X = SUBGMRES(A,B,IX,RTOL,ATOL,MAXITER,P) specifies the non-singular P.
%   If P is [] then a preconditioner is not applied.  P may be a function
%   handle PFUN of a function returning P\X.

%   X = SUBGMRES(A,B,IX,RTOL,ATOL,MAXITER,P,M) specifies the inner product
%   (symmetric positive definite) matrix M whose inverse is used to measure
%   the residual norm and perform the Arnoldi orthogonalization. If M is []
%   then the standard inner product is used. M may be a function handle
%   MFUN of a function returning M\X. Note that in case subvector norms are
%   required, i.e., IX is nontrivial, then M must have block-diagonal
%   structure, i.e., M must act on each subvector individually.
%
%   X = SUBGMRES(A,B,IX,RTOL,ATOL,MAXITER,P,M,X0) specifies the
%   initial guess. If X0 is [] then SUBGMRES uses the default, an all
%   zero vector.
%
%   X = SUBGMRES(A,B,IX,RTOL,ATOL,MAXITER,P,M,X0,VARARGIN)
%   specifies further arguments which will be passed to all of AFUN,
%   PFUN and MFUN, provided that those are functions (not matrices).
%
%   [X,FLAG] = SUBGMRES(A,B,...) also returns a convergence FLAG, following the
%   convention of MATLAB's built-in GMRES:
%    0 SUBGMRES converged to the desired tolerance within MAXITER iterations.
%    1 SUBGMRES iterated MAXITER times but did not converge to the desired
%    tolerance.
%   Improper use raises an error rather than setting a flag: the inner product M
%   must be Hermitian positive definite and compatible with the subvector
%   partition IX, otherwise the errors subgmres:notHermitian or
%   subgmres:notPositiveDefinite are raised.

%   [X,FLAG,RSUBNORMS] = SUBGMRES(A,B,...) also returns the M-inverse-norm
%   of the residual subvectors at the final iterate X. This norm is
%   obtained by a progressive update formula and may be subject to
%   accumulating rounding error when many iterations are performed. When no
%   inner product matrix is specified, the standard norm is employed. The
%   final entry in RSUBNORMS is the total residual norm.
%
%   [X,FLAG,RSUBNORMS,ITER] = SUBGMRES(A,B,...) also returns
%   the number of iterations performed to reach X.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST] = SUBGMRES(A,B,...) returns the
%   complete history of the M-inverse-norm of the residual subvectors as
%   well as of the full residual vector throughout the iterations. The
%   final column will agree with RSUBNORMS.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST] = SUBGMRES(A,B,...)
%   will return also the entire history of iterates column-wise in the
%   matrix XHIST.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST] = SUBGMRES(A,B,...)
%   will also return the entire history of residuals column-wise in the matrix
%   RHIST. Each column is evaluated as B - A*X for the corresponding iterate,
%   at the price of one extra application of A per iteration. The price is the
%   point. The residual the recursion carries along is available for nothing,
%   its M-inverse-norm being the last row of RSUBNORMSHIST, but it is the
%   recursion's own account of itself; RHIST puts the question to the system
%   instead, which is what lets it serve as an independent check. In exact
%   arithmetic the two agree. In floating point they separate as the iteration
%   converges, by about eps divided by the relative residual reached.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST,VHIST] = SUBGMRES(A,B,...)
%   will also return the entire basis of the Krylov subspace for the residual
%   in the matrix VHIST. Its columns are orthonormal with respect to the inv(M)-
%   inner product.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST,VHIST,WHIST] = SUBGMRES(A,B,...)
%   will also return the primal companions of the dual Krylov subspace basis in
%   the matrix WHIST = M \ VHIST. Its columns are orthonormal with respect to
%   the inner product induced by M. These vectors are formed by the iteration
%   anyway: each one is reused in every subsequent orthogonalization, which is
%   why they are stored rather than recomputed.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST,VHIST,WHIST,ZHIST] = SUBGMRES(A,B,...)
%   will also return the primal Krylov subspace basis in the matrix ZHIST = P \
%   VHIST. Its columns are orthonormal with respect to the inner product
%   induced by P'*inv(M)'*P. Unlike WHIST, these vectors are consumed
%   immediately by the iteration and are stored only when requested here.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST,VHIST,WHIST,ZHIST,H] = SUBGMRES(A,B,...)
%   will return also the upper Hessenberg matrix H arising from the Arnoldi
%   process.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST,VHIST,WHIST,ZHIST,H,RH] = SUBGMRES(A,B,...)
%   will return also the upper triangular factor RH of H arising from the QR
%   factorization.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST,VHIST,WHIST,ZHIST,H,RH,TOLTABLE] = SUBGMRES(A,B,...)
%   returns also the relative tolerances (first columns) and absolute
%   tolerances (second column) that were in effect for the present run.
%   The rows pertain to the subvectors and total vector norms, respectively.
%   This feature helps the user verify the requested convergence tolerances.
%
%   [X,FLAG,RSUBNORMS,ITER,RSUBNORMSHIST,XHIST,RHIST,VHIST,WHIST,ZHIST,H,RH,TOLTABLE,IX] = SUBGMRES(A,B,...)
%   returns also the cell array ix of indices pertaining to the subvectors.
%   In case of no subvectors, ix will be 1:N.

%% Parse the output arguments.
% Decide which of the optional history outputs the caller actually wants, so
% that the ones it does not are never formed: skipping the iterate and residual
% histories saves both their storage and one application of A per iteration.
%
% Octave provides isargout, which reports false both for a position beyond
% nargout and for one the caller discarded with a ~ placeholder. That is the
% precise answer, and it lets [~,~,~,~,~,~,~,VHIST] = subgmres(...) reach past
% the histories without paying for them.
%
% MATLAB has no isargout and no supported equivalent: its nargout counts
% ~-discarded outputs, so a discarded output cannot be told from a requested
% one. There we fall back to a threshold on nargout. That is conservative
% rather than wrong -- a discarded output is computed and then thrown away, so
% the saving above is lost, but every returned value is still correct. The test
% suite checks the two behaviors separately.
%
% isargout has to be called here, in the body of the function whose outputs it
% reports on; it cannot be moved into a helper.
if (exist('isargout') ~= 0)
	wanted = [isargout(5), isargout(6), isargout(7), ...
		isargout(8), isargout(9), isargout(10)];
else
	wanted = (5:10) <= nargout;
end
need_RsubnormsHIST = wanted(1);
need_XHIST = wanted(2);
need_RHIST = wanted(3);
need_VHIST = wanted(4);
need_WHIST = wanted(5);
need_ZHIST = wanted(6);

%% Parse the input arguments.
if (nargin < 2)
	error('subgmres:Not enough input arguments (%d provided).\nAt least A and b need to be provided.\n',nargin);
end

%% Check the sizes of inputs A and b and wrap A into a function if necessary.
if (isnumeric(A))
	% Verify that the matrix and right hand side vector inputs have appropriate sizes.
	[a1,a2] = size(A);
	A = @(x) A*x;
	if (a1 ~= a2)
		error('subgmres:Matrix has size %d x %d but must be square.\n',a1,a2);
	end
	if (~isequal(size(b),[a1,1]))
		error('subgmres:Right hand side must be a column vector of length %d.\n',a1);
	end
else
	% Verify that the right hand side vector input has appropriate size.
	a1 = size(b,1);
	if (~iscolumn(b))
		error('subgmres:Right hand side must be a column vector of length %d.\n',a1);
	end
end
n = a1;


%% Parse subvector index sets into a cell array
if (nargin >= 3) && ~isempty(ix)
	if iscell(ix)
		% If ix is a cell array (of subvector indices) already, we use it directly.
		nSubvectors = length(ix);
	else
		% We assume that ix is a vector containing the starting indices of
		% each contiguous subvector into the full vector, in ascending order.
		% Make ix a colummn vector and create the subvector indices.
		ix = ix(:);
		nSubvectors = length(ix);
		ix = [ix; length(b)+1];
		ixx = cell(nSubvectors,1);
		for i = 1:nSubvectors
			ixx{i} = [ix(i):ix(i+1)-1]';
		end
		% Overwrite the given vector ix of starting indices by the cell array of
		% subvector indices.
		ix = ixx;
	end
else
	% We do not have any subvectors and work only with the total residual.
	nSubvectors = 1;
	ix = cell(nSubvectors,1);
	ix{1} = [1:n]';
end

%% Check and assign the relative tolerances.
if (nargin < 4) || isempty(rtol)
	rtol = [inf(nSubvectors,1); 1e-6];
elseif (length(rtol) == 1)
	rtol = [inf(nSubvectors,1); rtol];
elseif (length(rtol) == nSubvectors)
	rtol = [rtol(:); inf];
elseif (length(rtol) ~= nSubvectors+1)
	error('subgmres:rtol must be of length 1, %d or %d.\n',nSubvectors,nSubvectors+1);
end

%% Check and assign the absolute tolerances.
if (nargin < 5) || isempty(atol)
	atol = [inf(nSubvectors,1); inf];
elseif (length(atol) == 1)
	atol = [inf(nSubvectors,1); atol];
elseif (length(atol) == nSubvectors)
	atol = [atol(:); inf];
elseif (length(atol) ~= nSubvectors+1)
	error('subgmres:atol must be of length 1, %d or %d.\n',nSubvectors,nSubvectors+1);
end

%% Assign the inferred tolerances to the toltable output, if needed.
if (nargout >= 11)
	toltable = [rtol, atol];
end

%% Assign the maximum # of iterations.
if (nargin < 6) || isempty(maxiter)
	maxiter = round(min(n,100));
end

%% Check and assign the preconditioner P.
if ((nargin >= 7) && ~isempty(P))
	have_preconditioner = true;
	if (isnumeric(P))
		if (~isequal(size(P),[a1,a1]))
			error('subgmres:The preconditioner needs to be a matrix of size %d x %d.\n',a1,a1);
		end
		applyP = @(x) P\x;
	else
		applyP = P;
	end
else
	have_preconditioner = false;
end

%% Check and assign the inner product M.
if ((nargin >= 8) && ~isempty(M))
	have_inner_product = true;
	if (isnumeric(M))
		if (~isequal(size(M),[a1,a1]))
			error('subgmres:The inner product needs to be a matrix of size %d x %d.\n',a1,a1);
		end
		applyM = @(x) M\x;
	else
		applyM = M;
	end
else
	have_inner_product = false;
end

%% Check and assign the initial guess x0.
if ((nargin >= 9) && ~isempty(x0))
	if ~isequal(size(x0),[n,1])
		error('subgmres:The initial guess must of a column vector of length %d.\n', n);
	end
else
	x0 = zeros(n,1);
end

%% Initialize some vectors and scalar quantities (lines 1-2).
% Initialize the initial residual and the dual Krylov subspace basis. V and W are
% preallocated to their maximum possible size (one column per iteration, plus the
% initial column) to avoid the O(n*maxiter^2) cost of growing them by concatenation
% one column at a time; they are truncated to the actual number of iterations
% performed once the main loop terminates. Throughout, column j holds the vector
% denoted v_j (resp. w_j) in the paper, so explicit indices replace MATLAB's
% "(:,end)" idiom wherever the number of columns filled so far does not equal iter+1.
x = x0;
r  = b - A(x,varargin{:});
V = zeros(n, maxiter+1);
V(:,1) = r;
% Pull the initial residual into the primal space.
W = zeros(n, maxiter+1);
if have_inner_product
	W(:,1) = applyM(V(:,1),varargin{:});
else
	W(:,1) = V(:,1);
end

%% Compute the initial partial duality pairings (line 3).
mu = zeros(nSubvectors,1);
muScale = zeros(nSubvectors,1);
for i = 1:nSubvectors
	mu(i) = V(ix{i},1).' * conj(W(ix{i},1));
	muScale(i) = abs(V(ix{i},1)).' * abs(W(ix{i},1));
end
mu = assertReal(mu, 'A squared M-inverse-norm of an initial residual subvector');
mu = assertNonnegative(mu, muScale, 'A squared M-inverse-norm of an initial residual subvector');

%% Evaluate the M-inverse-norm of the initial residual and initialize g (lines 4-5).
% The squared norm is the sum of the (real, nonnegative) subvector pairings, so
% no separate full-vector pairing is needed.
gamma2 = sum(mu);
gamma = sqrt(gamma2);
g = zeros(maxiter+1, 1);
g(1) = gamma;

%% Normalize v and w, and rescale the partial pairings (lines 6-7).
% When gamma = 0, the initial residual is (exactly) zero, i.e., x0 is already
% the exact solution. Skip the normalization to avoid a 0/0 NaN; the
% convergence test below then returns immediately with flag = 0 and x = x0.
if (gamma > 0)
	V(:,1) = V(:,1) / gamma;
	W(:,1) = W(:,1) / gamma;
	mu = mu / gamma2;
end

%% Perform further initializations (lines 8-10)
% Initialize the m vector
m = V(:,1);

% Assign vector of M-inverse-norms of subvectors and total initial residual
eta = [sqrt(mu) * gamma; gamma];
eta0 = eta;

%% Initialize the sequence of iterates, residuals and residual subvector M-inverse-norms, if needed
if (need_XHIST), XHIST = zeros(n, maxiter+1); XHIST(:,1) = x; end
if (need_RHIST), RHIST = zeros(n, maxiter+1); RHIST(:,1) = r; end
if (need_RsubnormsHIST), RsubnormsHIST = zeros(nSubvectors+1, maxiter+1); RsubnormsHIST(:,1) = eta0; end
if (need_ZHIST), Z = zeros(n, maxiter); end

%% Assemble the stopping tolerances (line 12).
% The tolerance vector is fixed for the run, rtol, atol and eta0 all being
% known by now, so it is formed once here rather than inside the loop.
%
% A subvector whose initial residual is exactly zero has nothing to reduce, so
% a relative criterion on it is vacuous. When its rtol is Inf as well -- which
% is what a scalar RTOL leaves behind, the caller having asked only for the
% total -- the product is Inf*0 = NaN. Map that to Inf, no relative
% requirement, and leave any absolute one in place. Doing this explicitly
% matters: min() drops NaN in MATLAB but propagates it in numpy, so leaving
% the NaN in place made the two implementations disagree on the same input,
% one converging where the other iterated until the Krylov space ran dry and
% the triangular solve failed. That case is reached by the most ordinary of
% right hand sides -- a Stokes system with no source in the continuity
% equation has an exactly zero pressure residual to begin with.
relativeTol = rtol .* eta0;
relativeTol(isnan(relativeTol)) = inf;
tol = min(relativeTol, atol);

%% Initialize counters and flags
iter = 0;
done = 0;
H = zeros(maxiter+1, maxiter);
RH = zeros(maxiter+1, maxiter);
c = zeros(maxiter, 1);
s = zeros(maxiter, 1);
theta = zeros(nSubvectors,1);

%% Main loop (lines 11-43)
while (~done)

	% Check total residual M-inverse-norm for convergence (line 12).
	% Convergence is achieved when the M-inverse-norms of the total residual vector
	% and all of its subvectors verify both their relative and absolute tolerances.
	if (all(eta <= tol))
		flag = 0;
		done = 1;
		break
	end

	%% Check for maximum # of iterations reached.
	if (iter >= maxiter)
		flag = 1;
		done = 1;
		break
	end

	%% Increase the iteration counter.
	iter = iter + 1;

	%% Apply the preconditioner and the matrix (lines 13-14)
	vnew = V(:,iter);
	if have_preconditioner
		vnew = applyP(vnew,varargin{:});
	end
	% At this point, vnew is z_iter = P\v_iter, the primal Krylov basis vector of
	% line 13 (cf. eq:relation-primal-and-dual-Krylov-bases). It costs nothing
	% extra, the matrix application of line 14 needing it anyway, but it is
	% consumed there and then, so it is kept only when ZHIST was asked for.
	% Contrast W, whose columns are reused by every later orthogonalization and
	% are therefore always stored.
	if need_ZHIST
		Z(:,iter) = vnew;
	end
	vnew = A(vnew,varargin{:});

	%% Perform a modified Gram-Schmidt orthogonalization (lines 15-18)
	for j = 1:iter
		H(j,iter) = vnew.' * conj(W(:,j));
		vnew = vnew - H(j,iter) * V(:,j);
	end

	%% Apply M-inverse (line 19)
	if have_inner_product
		wnew = applyM(vnew,varargin{:});
	else
		wnew = vnew;
	end

	%% Obtain subvector norms of the new Arnoldi vector (line 20)
	psi = zeros(nSubvectors,1);
	psiScale = zeros(nSubvectors,1);
	for i = 1:nSubvectors
		psi(i) = vnew(ix{i}).' * conj(wnew(ix{i}));
		psiScale(i) = abs(vnew(ix{i})).' * abs(wnew(ix{i}));
	end
	psi = assertReal(psi, 'A squared M-inverse-norm of an Arnoldi subvector');
	psi = assertNonnegative(psi, psiScale, 'A squared M-inverse-norm of an Arnoldi subvector');

	%% Obtain the norm of the new Arnoldi vector (line 21)
	H(iter+1,iter) = sqrt(sum(psi));

	%% Extend the Hessenberg matrix and its triangular factor
	RH(1:iter+1,iter) = H(1:iter+1,iter);

	% Check for an invariant subspace
	% If Arnoldi returned a (numerically) zero vector, the dual Krylov subspace was
	% a (numerically) invariant subspace of A P^{-1}, meaning GMRES has converged
	if (abs(H(iter+1,iter)) < 10*eps)

		%% Observe that the new Arnoldi vector is numerically zero (lines 22-25)
		% V(:,iter+1) and W(:,iter+1) are already zero in the preallocated arrays.
		psi = zeros(nSubvectors,1);
		theta = zeros(nSubvectors,1);

		%% Apply previous Givens rotations (lines 26-30)
		for j = 1:iter-1
			xi1 = RH(j,iter);
			xi2 = RH(j+1,iter);
			RH(j,iter) = c(j) * xi1 + s(j) * xi2;
			RH(j+1,iter) = - conj(s(j)) * xi1 + conj(c(j)) * xi2;
		end

		%% Calculate new Givens rotations (lines 31-34)
		c(iter) = 1;
		s(iter) = 0;
		RH(iter+1,iter) = 0;

	else

		%% Normalize the new Arnoldi vector (lines 22-25)
		psi = psi / sum(psi);
		V(:,iter+1) = vnew / H(iter+1,iter);
		W(:,iter+1) = wnew / H(iter+1,iter);
		for i = 1:nSubvectors
			theta(i) = m(ix{i}).' * conj(W(ix{i},iter+1));
		end

		%% Apply previous Givens rotations (lines 26-30)
		for j = 1:iter-1
			xi1 = RH(j,iter);
			xi2 = RH(j+1,iter);
			RH(j,iter) = c(j) * xi1 + s(j) * xi2;
			RH(j+1,iter) = - conj(s(j)) * xi1 + conj(c(j)) * xi2;
		end

		%% Calculate new Givens rotations (lines 31-34)
		alpha = sqrt(abs(RH(iter,iter))^2 + abs(RH(iter+1,iter))^2);
		sign_x1 = sign(RH(iter,iter));
		c(iter) = abs(RH(iter,iter)) / alpha;
		s(iter) = sign_x1 * conj(RH(iter+1,iter)) / alpha;
		RH(iter,iter) = sign_x1 * alpha;
		RH(iter+1,iter) = 0;

	end

	%% Update the (partial) residual norms (lines 35-41)
	% gIter holds g(iter) as it stood before this iteration's update; both g(iter)
	% and g(iter+1) below are derived from it, so it must be read before either write.
	gIter = g(iter);
	g(iter+1) = -conj(s(iter)) * gIter;
	g(iter) = c(iter) * gIter;
	m = - s(iter) * m + c(iter) * V(:,iter+1);
	for i = 1:nSubvectors
		mu(i) = abs(s(iter))^2 * mu(i) + abs(c(iter))^2 * psi(i) - 2 * real(s(iter) * conj(c(iter)) * theta(i));
		eta(i) = abs(g(iter+1)) * sqrt(max(mu(i), 0));
	end
	eta(end) = abs(g(iter+1));

	%% Store the iteration history
	if (need_RsubnormsHIST)
		RsubnormsHIST(:,iter+1) = eta;
	end
	% Reconstruct the current iterate (lines 42-43). The residual history needs
	% x as well, to evaluate b - A*x below, so the reconstruction is driven by
	% either request and only the storing is gated on XHIST. The distinction
	% matters now that isargout decides the two independently: were the
	% reconstruction still gated on need_XHIST alone, a caller asking for RHIST
	% but not XHIST would silently receive the initial residual in every column.
	if (need_XHIST || need_RHIST)
		y = solveTriangular(RH(1:iter,1:iter), g(1:iter));
		if have_preconditioner
			x = x0 + applyP(V(:,1:iter) * y, varargin{:});
		else
			x = x0 + V(:,1:iter) * y;
		end
		if (need_XHIST)
			XHIST(:,iter+1) = x;
		end
	end
	% Evaluate and store the current residual. This is a genuine evaluation of
	% b - A*x, not the residual propagated alongside the Givens rotations, which
	% would be free: the Arnoldi relation gives r = g(iter+1) * m exactly, m
	% being the vector already updated above for the subvector norms, and its
	% norm is what eta(end) already reports. Taking RHIST from there would make
	% it the recursion's own account of itself, where its whole use is to be the
	% independent check. The two also part company in practice, agreeing only to
	% about eps divided by the relative residual reached -- that is, worst in the
	% very regime a convergence study examines.
	if (need_RHIST)
		r  = b - A(x,varargin{:});
		RHIST(:,iter+1) = r;
	end

end % while (~done)

%% Truncate the preallocated arrays to the number of iterations actually performed.
V = V(:,1:iter+1);
W = W(:,1:iter+1);
H = H(1:iter+1,1:iter);
RH = RH(1:iter+1,1:iter);
g = g(1:iter+1);
if (need_ZHIST), Z = Z(:,1:iter); end
if (need_XHIST), XHIST = XHIST(:,1:iter+1); end
if (need_RHIST), RHIST = RHIST(:,1:iter+1); end
if (need_RsubnormsHIST), RsubnormsHIST = RsubnormsHIST(:,1:iter+1); end

%% Reconstruct the final iterate and residual (lines 42-43)
if (iter > 0) && ~(need_XHIST || need_RHIST)
	y = solveTriangular(RH(1:iter,1:iter), g(1:iter));
	if have_preconditioner
		x = x0 + applyP(V(:,1:iter) * y, varargin{:});
	else
		x = x0 + V(:,1:iter) * y;
	end
end

%% Evaluate the final residual
if (iter > 0) && ~need_RHIST
	r  = b - A(x,varargin{:});
end

%% Finalize some output arguments
Rsubnorms = eta;
if need_VHIST
	VHIST = V;
end
if need_WHIST
	WHIST = W;
end
if need_ZHIST
	ZHIST = Z;
end


%% Local functions.
function zr = assertReal(z, description)
% Return the real part of z after verifying that its imaginary part is
% negligible relative to its magnitude. A quantity that must be real (a squared
% M-inverse-norm) with a non-negligible imaginary part indicates that the inner
% product M is not Hermitian, which violates the assumptions of the method.
% The relative tolerance sqrt(eps) sits between roundoff-level imaginary parts
% (O(n*eps), no cancellation occurs in a squared norm) and a genuine violation
% (O(1)).
reltol = sqrt(eps);
if (any(abs(imag(z(:))) > reltol * abs(z(:))))
	error('subgmres:notHermitian', ...
		'%s has a non-negligible imaginary part; the inner product M does not appear to be Hermitian.', description);
end
zr = real(z);


function zr = assertNonnegative(z, scale, description)
% Return max(z, 0), verifying that z (assumed real) is not negative beyond a
% roundoff-level tolerance relative to the given scale. The scale should bound
% the magnitude of the terms that z was summed from, e.g., abs(v).' * abs(w) for
% an inner product v.' * w; unlike a squared norm, this quadratic form can lose
% accuracy to cancellation, so the tolerance is measured against the terms, not
% against z itself. A genuinely negative value indicates that the inner product
% M is not positive definite, which violates the assumptions of the method.
reltol = sqrt(eps);
if (any(z(:) < -reltol * scale(:)))
	error('subgmres:notPositiveDefinite', ...
		'%s is negative; the inner product M does not appear to be positive definite.', description);
end
zr = max(z, 0);


function y = solveTriangular(R, g)
% Solve the upper triangular system R*y = g for the coefficients of the current
% iterate, without emitting a near-singularity warning.
%
% R is the triangular factor of the Hessenberg matrix, and it becomes singular
% precisely when the method has converged: at a happy breakdown, and equally
% once the residual has reached roundoff and the next Krylov direction carries
% no new information. Continuing to iterate past that point -- which a caller
% does whenever the tolerance cannot be met, so that the run stops at maxiter
% instead -- therefore reaches this solve with a numerically singular R, and
% both MATLAB and Octave warn about it.
%
% The warning is not informative here. The solve still returns an accurate
% iterate, the leading columns of R being well conditioned and the offending
% trailing ones multiplying directions that no longer move the residual; and
% the outcome of the run is already reported through the flag output. Silencing
% it keeps the diagnostic honest, rather than having every over-long run print
% a message that looks like a defect.
%
% The state is saved and restored around the single statement, so that warnings
% raised by caller-supplied function handles for A, P or M are unaffected.
ws = warning('off', 'all');
y = R \ g;
warning(ws);
