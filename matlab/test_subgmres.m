% This script provides unit tests for subgmres.m.
% Its synopsis is 
%   [x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,WHIST,ZHIST,H,RH,toltable,ix]
%     = subgmres(A,b,ix,rtol,atol,maxiter,P,M,x0,varargin)
%
% This script is to be run as
%   runtests('test_subgmres')

% Create some common input for many test problems.
rng(42);
n = 5;
A = randn(n,n);
b = randn(n,1);
Mref = spd(n);
Pref = randn(n,n);
atolSame = 1e-12;
rtolSame = 1e-6;
% Tolerance for the paper-identity checks. It must live here, in the shared
% setup, not inside a %% section: MATLAB runs each section as an independent
% test that sees only this preamble.
tolId = 1e-8;

% Create extra input for test problems using subvector partitioning.
n1 = max(round(n/3),1);
n2 = n - n1;
ixsubref = {[1:n1], [n1+1:n1+n2]};
Msubref = blkdiag(spd(n1),spd(n2));


%% Perform convergence tests without subvector partitioning.

% Case: M = identity, P = identity
x0 = zeros(n,1);
r0 = b - A*x0;
M = eye(n);
P = eye(n);
ix = [];
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,M,x0,ix,A,b,atolSame,rtolSame);

% Case: M = assigned above, P = identity
x0 = zeros(n,1);
r0 = b - A*x0;
M = Mref;
P = eye(n);
ix = [];
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b,[],[],[],[],[],M);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,M,x0,ix,A,b,atolSame,rtolSame);

% Case: M = identity, P = assigned above
x0 = zeros(n,1);
r0 = b - A*x0;
M = eye(n);
P = Pref;
ix = [];
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b,[],[],[],[],P,[]);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,M,x0,ix,A,b,atolSame,rtolSame);

% Case: M = assigned above, P = assigned above
x0 = zeros(n,1);
r0 = b - A*x0;
M = Mref;
P = Pref;
ix = [];
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b,[],[],[],[],P,M);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,M,x0,ix,A,b,atolSame,rtolSame);


%% Perform convergence tests with subvector partitioning.

% Case: M = identity, P = identity
x0 = zeros(n,1);
r0 = b - A*x0;
Msub = eye(n);
P = eye(n);
ix = ixsubref;
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b,ix);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,Msub,x0,ix,A,b,atolSame,rtolSame);
% [x_,flag_,Rsubnorms_,iter_,RsubnormsHIST_,XHIST_,RHIST_,VHIST_,WHIST_,ZHIST_,H_,toltable_] = subgmres(A,b);
% agreementTests(x,x_);

% Case: M = assigned above, P = identity
x0 = zeros(n,1);
r0 = b - A*x0;
Msub = Msubref;
P = eye(n);
ix = ixsubref;
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b,ix,[],[],[],[],Msub);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,Msub,x0,ix,A,b,atolSame,rtolSame);

% Case: M = identity, P = assigned above
x0 = zeros(n,1);
r0 = b - A*x0;
Msub = eye(n);
P = Pref;
ix = ixsubref;
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b,ix,[],[],[],P,[]);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,Msub,x0,ix,A,b,atolSame,rtolSame);

% Case: M = assigned above, P = assigned above
x0 = zeros(n,1);
r0 = b - A*x0;
Msub = Msubref;
P = Pref;
ix = ixsubref;
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(A,b,ix,[],[],[],P,Msub);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,Msub,x0,ix,A,b,atolSame,rtolSame);


%% Verify that A, applyP and applyM receive extra input arguments (varargin passthrough).
% Wrap A, P and M as handles that REQUIRE a second argument; if subgmres did not
% pass varargin through, these calls would error on a missing argument, so a
% correct solve proves the passthrough.
Afun = @(y, extra) A*y;
Pfun = @(y, extra) Pref\y;
Mfun = @(y, extra) Mref\y;
xh = subgmres(Afun, b, [], rtolSame, [], [], Pfun, Mfun, [], 12345);
xm = subgmres(A, b, [], rtolSame, [], [], Pref, Mref);
assert(norm(xh - xm) <= 1e-8);


%% Verify paper identities: Arnoldi relation, Hessenberg entries, M-inverse-orthonormality of V, the primal companions WHIST (W = M\V, M-orthonormality), and the primal basis ZHIST (Z = P\V, P'*inv(M)*P-orthonormality).
[~,~,~,iter,~,~,~,VHIST,WHIST,ZHIST,H] = subgmres(A, b, [], rtolSame, [], [], Pref, Mref);
paperIdentityTests(VHIST, WHIST, ZHIST, H, iter, A, Pref, Mref, tolId);


%% Convergence and identities with complex data.
rng(11);
Ac = randn(n) + 1i*randn(n);
bc = randn(n,1) + 1i*randn(n,1);
Rc = randn(n) + 1i*randn(n);
Mc = Rc'*Rc + n*eye(n);                 % Hermitian positive definite
P = randn(n) + 1i*randn(n);
x0 = zeros(n,1);
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,WHIST,ZHIST,H,RH,toltable] = subgmres(Ac, bc, [], [], [], [], P, Mc);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,Mc,x0,[],Ac,bc,atolSame,rtolSame);
paperIdentityTests(VHIST, WHIST, ZHIST, H, iter, Ac, P, Mc, tolId);
% Complex data with subvector partitioning and a block-diagonal HPD M.
Rc1 = randn(n1)+1i*randn(n1);
Rc2 = randn(n2)+1i*randn(n2);
Mcsub = blkdiag(Rc1'*Rc1 + n1*eye(n1), Rc2'*Rc2 + n2*eye(n2));
[x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,VHIST,~,~,H,RH,toltable] = subgmres(Ac, bc, ixsubref, [], [], [], P, Mcsub);
standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,Mcsub,x0,ixsubref,Ac,bc,atolSame,rtolSame);


%% The iterate minimizes the residual over the affine Krylov space.
% The property the method exists for. Everything else in this file checks the
% Arnoldi machinery -- the relation, the orthonormality of the bases, the
% bookkeeping of the norms -- and none of it checks that the machinery solves
% the minimization it was built for. An implementation that assembled every
% basis correctly and then solved the wrong least-squares problem would pass
% all of those and fail this.
%
% Complex, preconditioned, non-standard inner product and a nonzero initial
% guess, so no term of the statement is trivially absent. Pmin is a decent
% preconditioner, which makes the run stop well short of the Krylov dimension;
% at exhaustion the residual is machine zero and there is nothing to measure
% orthogonality against.
nmin = 40;
randc = @() randn(nmin,nmin) + 1i*randn(nmin,nmin);
Amin = randc() + 8*eye(nmin);
Pmin = Amin + 0.6*randc();
Mroot = randc();
Mmin = Mroot*Mroot' + nmin*eye(nmin);
bmin = randn(nmin,1) + 1i*randn(nmin,1);
x0min = randn(nmin,1) + 1i*randn(nmin,1);
[xmin, flagmin, ~, itermin, ~, XHISTmin, RHISTmin, ~, ~, ZHISTmin] = ...
	subgmres(Amin, bmin, [], 1e-8, [], nmin, Pmin, Mmin, x0min);
assert(flagmin == 0);
assert(itermin < nmin);
mInverseNorm = @(v) sqrt(real(v' * (Mmin \ v)));
initialNorm = mInverseNorm(RHISTmin(:,1));
for k = 1:itermin
	residual = RHISTmin(:,k+1);
	AZ = Amin * ZHISTmin(:,1:k);
	% First-order optimality: the residual is M-inverse-orthogonal to
	% A P^{-1} K_k, the space the correction was drawn from. This is the
	% minimization written as a Galerkin condition, an identity rather than an
	% approximation.
	assert(norm(AZ' * (Mmin \ residual)) / initialNorm <= 1e-9);
	% The same statement in its literal form: no other point of the affine
	% space does better. Two magnitudes, since a first-order condition alone
	% would also hold at a maximum or a saddle.
	optimal = mInverseNorm(residual);
	for trial = 1:3
		direction = randn(k,1) + 1i*randn(k,1);
		for step = [1e-1 1e-3]
			moved = XHISTmin(:,k+1) + ZHISTmin(:,1:k) * (step*direction);
			assert(mInverseNorm(bmin - Amin*moved) >= optimal);
		end
	end
end

%% The documented defaults are the actual defaults.
% Each asserted against its documented value rather than inferred from a run
% that happens to succeed. ndef exceeds 100 so that the maxiter default is a
% real cap and not simply n.
ndef = 120;
Adef = randn(ndef,ndef) + 3*sqrt(ndef)*eye(ndef);
bdef = randn(ndef,1);
% rtol defaults to 1e-6 on the total residual only, atol to Inf.
[xdef, flagdef, ~, iterdef, ~, ~, ~, ~, ~, ~, ~, ~, toltabledef] = subgmres(Adef, bdef);
assert(toltabledef(end,1) == 1e-6);
assert(all(isinf(toltabledef(1:end-1,1))));
assert(all(isinf(toltabledef(:,2))));
% x0 defaults to the zero vector: passing it explicitly changes nothing.
[xexplicit, ~, ~, iterexplicit] = subgmres(Adef, bdef, [], [], [], [], [], [], zeros(ndef,1));
assert(iterexplicit == iterdef);
assert(isequal(xexplicit, xdef));
% maxiter defaults to round(min(n,100)). Given a tolerance it cannot meet, the
% run must stop at exactly that many iterations.
[~, flagcapped, ~, itercapped] = subgmres(Adef, bdef, [], 0);
assert(itercapped == 100);
assert(flagcapped == 1);
[~, ~, ~, iterexplicitcap] = subgmres(Adef, bdef, [], 0, [], 17);
assert(iterexplicitcap == 17);

%% Iterating past convergence must stay silent.
% Running to the maxiter cap drives the triangular factor of the Hessenberg
% matrix singular: once the residual has reached roundoff, the trailing columns
% carry no information and both MATLAB and Octave warn when the coefficients are
% solved for. The warning says nothing the flag output does not, and a released
% test suite that prints "matrix singular" reads as a defect, so subgmres
% silences it around that one solve. Guard that, and guard that the iterate is
% still accurate, since silencing a warning must not mean ignoring it.
% Each section gets the shared setup and nothing else, so build the system here.
% n exceeds 100 so that the default maxiter is a real cap, and rtol = 0 is a
% tolerance the run cannot meet, forcing it well past convergence.
nq = 120;
Aq = randn(nq,nq) + 3*sqrt(nq)*eye(nq);
bq = randn(nq,1);
lastwarn('');
[xquiet, flagquiet, ~, iterquiet] = subgmres(Aq, bq, [], 0);
assert(isempty(lastwarn()));
assert(iterquiet == 100 && flagquiet == 1);
assert(norm(bq - Aq * xquiet) <= 1e-10 * norm(bq));
% The history outputs take a second solve through the same path, once per
% iteration rather than once at the end.
lastwarn('');
[~, ~, ~, ~, ~, XHquiet] = subgmres(Aq, bq, [], 0);
assert(isempty(lastwarn()));
assert(norm(bq - Aq * XHquiet(:,end)) <= 1e-10 * norm(bq));
% Suppression must be local: the global warning state has to come back exactly
% as it was, so that warnings from caller-supplied handles for A, P or M, and
% from anything the caller does afterwards, are unaffected. Comparing the state
% across the call tests this without emitting a warning of our own, which would
% put back the console noise this section exists to remove.
stateBefore = warning();
[~] = subgmres(Aq, bq, [], 0);
assert(isequal(warning(), stateBefore));

%% Error paths: a non-Hermitian or indefinite inner product must raise an error.
threw = '';
try
	subgmres(A, b, [], rtolSame, [], [], [], eye(n) + 1i*triu(ones(n),1));
catch err
	threw = err.identifier;
end
assert(strcmp(threw, 'subgmres:notHermitian'));
Q = orth(randn(n));
Mindef = Q*diag([-1; (2:n)'])*Q';
Mindef = (Mindef + Mindef')/2;          % Hermitian but indefinite
threw = '';
try
	subgmres(A, b, [], rtolSame, [], [], [], Mindef);
catch err
	threw = err.identifier;
end
assert(strcmp(threw, 'subgmres:notPositiveDefinite'));


%% Edge cases: exact-zero initial residual and reaching maxiter.
% x0 already solves the system: the initial residual is exactly zero.
x0 = ones(n,1);
bexact = A*x0;
[x,flag,Rsubnorms,iter] = subgmres(A, bexact, [], rtolSame, [], [], [], [], x0);
assert(flag == 0);
assert(iter == 0);
assert(~any(isnan(x)));
% A tight tolerance with a single permitted iteration must report non-convergence.
[x,flag] = subgmres(A, b, [], 1e-14, [], 1, [], []);
assert(flag == 1);


%% Subvector partitions given as explicit, non-contiguous index sets.
% The starting-index form of ix implies contiguity; the cell-array form does
% not, and an assembly numbering unknowns by node rather than by field leaves
% each field scattered through the vector. ixsubref in the preamble is a cell
% array of contiguous ranges, which exercises the form but not the property.
%
% Built locally, well conditioned and comfortably larger than the iteration
% count it needs: on a system small enough to be solved by exhausting the
% Krylov space, every run ends where the monitored norms have parted company
% with the true ones, and no comparison against directly computed norms would
% mean anything there.
nc = 40;
ncBlock = 20;
Anc = randn(nc,nc) + 3*sqrt(nc)*eye(nc);
bnc = randn(nc,1);
Pnc = randn(nc,nc) + 3*sqrt(nc)*eye(nc);
% M must not couple the two blocks, or the subvector norms would not split.
Mnc = zeros(nc,nc);
for blockStart = [1 ncBlock+1]
	root = randn(ncBlock,ncBlock);
	Mnc(blockStart:blockStart+ncBlock-1, blockStart:blockStart+ncBlock-1) = ...
		root'*root + ncBlock*eye(ncBlock);
end
% Even positions take the first block, odd positions the second.
order = zeros(1,nc);
order(1:2:end) = 1:ncBlock;
order(2:2:end) = ncBlock+1:nc;
scattered = {find(order <= ncBlock), find(order > ncBlock)};
% Neither set is an interval, which is the whole point.
assert(any(diff(scattered{1}) ~= 1));
assert(any(diff(scattered{2}) ~= 1));
Ap = Anc(order,order);
bp = bnc(order);
Pp = Pnc(order,order);
Mp = Mnc(order,order);

% Relabelling the unknowns must change nothing but the labels.
[xContig, flagContig, RsubContig, iterContig] = ...
	subgmres(Anc, bnc, [1 ncBlock+1], 1e-8, [], nc, Pnc, Mnc);
[xScattered, flagScattered, RsubScattered, iterScattered] = ...
	subgmres(Ap, bp, scattered, 1e-8, [], nc, Pp, Mp);
assert(iterScattered == iterContig);
assert(flagScattered == flagContig);
assert(iterScattered < nc);                     % converged, not exhausted
assert(norm(RsubScattered - RsubContig) <= 1e-8 * max(1, norm(RsubContig)));
assert(norm(xScattered - xContig(order)) <= 1e-8 * norm(xContig));

% The reported norms must be the M-inverse norms of the residual restricted to
% each scattered set -- an indexing claim, and the one a partition that is not
% an interval could most easily get wrong.
residualScattered = bp - Ap*xScattered;
for i = 1:2
	idx = scattered{i};
	direct = sqrt(real(residualScattered(idx)' * (Mp(idx,idx) \ residualScattered(idx))));
	assert(abs(RsubScattered(i)/direct - 1) <= 1e-6);
end

% The stopping test must apply each tolerance to its own scattered set.
rtolScattered = [1e-10; 1e-4; inf];
[~, flagTolerances, RsubTolerances] = ...
	subgmres(Ap, bp, scattered, rtolScattered, [], nc, Pp, Mp);
assert(flagTolerances == 0);
for i = 1:2
	idx = scattered{i};
	initial = sqrt(real(bp(idx)' * (Mp(idx,idx) \ bp(idx))));
	assert(RsubTolerances(i) <= rtolScattered(i) * initial);
end

%% A subvector starting at exactly zero must not block the stopping test.
% With a scalar rtol the per-subvector entries are Inf, so a subvector whose
% initial residual is exactly zero gets Inf*0 = NaN for its relative tolerance.
% MATLAB's min drops NaN and numpy's propagates it, so the two implementations
% disagreed here until the NaN was mapped to Inf explicitly: this converged
% while the Python port iterated until the Krylov space was exhausted and its
% triangular solve failed. The case is an ordinary one -- a Stokes system with
% no source in the continuity equation starts with a zero pressure residual.
% Built locally: the shared preamble's n = 5 is too small to say anything.
nz = 12; nz1 = 8;
Az = randn(nz,nz) + 4*eye(nz);
bz = [randn(nz1,1); zeros(nz-nz1,1)];
[xz,flagz,~,iterz] = subgmres(Az,bz,[1 nz1+1],1e-8,[],200);
assert(flagz == 0);
assert(iterz <= nz);
assert(norm(bz - Az*xz) <= 1e-8 * norm(bz));

% Only the vacuous relative requirement is dropped, never the whole test. The
% zero subvector's residual does not stay zero, so an unreachable absolute
% tolerance on that subvector alone must still prevent convergence. A larger,
% well-conditioned system, so that convergence happens well before the Krylov
% space is exhausted and the two runs can actually differ.
na = 40; na1 = 25;
Aa = randn(na,na) + 3*sqrt(na)*eye(na);
ba = [randn(na1,1); zeros(na-na1,1)];
[~,flagLoose,~,~,SNloose] = subgmres(Aa,ba,[1 na1+1],1e-8,[],30);
assert(flagLoose == 0);
assert(SNloose(2,1) == 0);
assert(SNloose(2,2) > 0);
[~,flagBound] = subgmres(Aa,ba,[1 na1+1],1e-8,[inf; 1e-30; inf],30);
assert(flagBound == 1);

%% Optional outputs: RHIST must be right when XHIST is discarded (regression).
% Evaluating the residual needs the reconstructed iterate. While the optional
% outputs were gated on cumulative nargout thresholds, asking for RHIST always
% implied XHIST and the dependency stayed invisible; under isargout the two are
% requested independently, so subgmres has to reconstruct the iterate for either
% and store it only for XHIST. Gated on XHIST alone, the call below would return
% the initial residual in every column.
[~,~,~,~,~,XHISTref,RHISTref] = subgmres(A,b,[],rtolSame);
[~,~,~,~,~,~,RHIST] = subgmres(A,b,[],rtolSame);
assert(isequal(size(RHIST), size(RHISTref)));
assert(norm(RHIST - RHISTref,'fro') <= atolSame * max(1,norm(RHISTref,'fro')));
% The columns are the true residuals of the corresponding iterates, and they do
% decrease -- the symptom of the reconstruction being skipped is a history in
% which every column still equals the initial residual.
assert(norm(RHIST - (b - A*XHISTref),'fro') <= atolSame * max(1,norm(RHISTref,'fro')));
assert(norm(RHIST(:,end)) < norm(RHIST(:,1)));

%% Optional outputs discarded with ~ must cost nothing (isargout gating).
% Counting applications of A is what makes the saving observable. Without the
% histories, A is applied once per iteration for the Arnoldi step, once at the
% start for the initial residual and once at the end for the returned one.
% Requesting RHIST adds one application per iteration and drops the final one.
% Under the former nargout thresholds, VHIST implied both histories and the
% first call below would have paid for them.
global nApply
Acount = @(v) countingApply(A,v);
nApply = 0;
[~,~,~,iterMinimal] = subgmres(Acount,b,[],rtolSame);
applyMinimal = nApply;
nApply = 0;
[~,~,~,iterDiscarded,~,~,~,VHIST] = subgmres(Acount,b,[],rtolSame);
applyDiscarded = nApply;
nApply = 0;
[~,~,~,iterRHIST,~,~,RHIST,VHIST] = subgmres(Acount,b,[],rtolSame);
applyWithRHIST = nApply;
% Same iteration path throughout: which outputs are requested must not alter
% the run itself.
assert(iterMinimal == iterDiscarded);
assert(iterDiscarded == iterRHIST);
% Reaching past the discarded histories to VHIST costs exactly what asking for
% nothing beyond the fourth output costs -- under Octave, where isargout sees
% the ~ placeholders. MATLAB has no isargout, so subgmres falls back to a
% threshold on nargout and computes the discarded histories anyway; there the
% call costs what requesting RHIST costs. Assert whichever the platform
% promises, so that this test states the real contract on both rather than
% passing only where it was written.
if (exist('isargout') ~= 0)
	assert(applyDiscarded == applyMinimal);
	assert(applyDiscarded == iterDiscarded + 2);
else
	assert(applyDiscarded == 2*iterDiscarded + 1);
end
% Actually requesting RHIST is what costs the extra application per iteration.
assert(applyWithRHIST == 2*iterRHIST + 1);


function y = countingApply(A,v)
% Apply A, tallying the applications in a global so that a test can assert how
% much work an optional output costs.
global nApply
nApply = nApply + 1;
y = A*v;
end

function standardTests(x,flag,Rsubnorms,iter,RsubnormsHIST,XHIST,RHIST,H,RH,toltable,M,x0,ix,A,b,atolSame,rtolSame)
% This function performs a number of tests to verify certain aspects
% of the subgmres implementation.
n = length(x);
% subgmres treats an empty ix as a single (trivial) subvector spanning the whole
% vector, returning [subvector; total] = 2 rows. Mirror that convention here.
if isempty(ix)
	ix = {(1:n)'};
end
nSubvectors = length(ix);

% Make sure subgmres has converged.
assert(flag == 0);

% Get the tolerances used by subgmres.
rtol = toltable(end,1);
atol = toltable(end,2);

% Evaluate the true residual, its norm, and the norm of the initial residual.
% The M-inverse-norm follows the paper's antidual inner product,
% <r,r>_{M^{-1}} = r' * (M \ r), which is antilinear in the FIRST argument: the
% conjugate sits outside the solve with M, and conj(M\r) differs from M\conj(r)
% for complex M. It is real for Hermitian positive-definite M and reduces to the
% ordinary weighted norm for real data.
r = b - A*x;
r0 = b - A*x0;
s = M \ r;
s0 = M \ r0;
eta = zeros(nSubvectors+1,1);
eta0 = zeros(nSubvectors+1,1);
for i = 1:nSubvectors
	eta(i) = sqrt(real(r(ix{i})' * s(ix{i})));
	eta0(i) = sqrt(real(r0(ix{i})' * s0(ix{i})));
end
eta(end) = sqrt(real(r' * s));
eta0(end) = sqrt(real(r0' * s0));

% Ensure that the convergence tolerances are met.
assert(all(eta <= rtol .* eta0));
assert(all(eta <= atol));

% Ensure that the true residual norm agrees with the norm subgmres has.
assert(all(abs(eta - Rsubnorms) < atolSame));

% Ensure that the residual subvector norms squared sum up to the square of the total residual norm.
if (nSubvectors > 0)
	assert(abs(sum(Rsubnorms(1:end-1).^2) - Rsubnorms(end).^2) <= getTolerance(rtolSame, atolSame, Rsubnorms(end)).^2);
end

% Ensure that the residual subvector norms squared sum up to the square of the total residual norm throughout the history.
if (nSubvectors > 0)
	assert(all(abs(sum(RsubnormsHIST(1:end-1,:).^2, 1) - RsubnormsHIST(end,:).^2) <= getTolerance(rtolSame, atolSame, RsubnormsHIST(end,:)).^2));
end

% Ensure that the residual subvector norms squared match the subvector norms squared obtained from using XHIST
if (nSubvectors > 0)
	RES = b - A * XHIST;
	tmp = real(conj(RES) .* (M \ RES));
	for i = 1:nSubvectors
		assert(all(abs(sum(tmp(ix{i},:), 1) - RsubnormsHIST(i,:).^2) <= getTolerance(rtolSame, atolSame, sqrt(sum(tmp(ix{i},:), 1)))));
	end
end

% Ensure that the total residual norm history has the expected size.
assert(size(RsubnormsHIST,1) == nSubvectors+1);
assert(size(RsubnormsHIST,2) == iter+1);

% Ensure that the total residual norm history is non-increasing.
assert(all(diff(RsubnormsHIST(end,:)) <= 0));

% Ensure that the most recent entry in the residual subnorm vector agrees with Rsubnorms
assert(all(Rsubnorms == RsubnormsHIST(:,end)));

% Ensure that the iterates' history has the expected size.
assert(size(XHIST,1) == n);
assert(size(XHIST,2) == iter+1);

% Ensure that the residuals' history has the expected size.
assert(size(RHIST,1) == n);
assert(size(RHIST,2) == iter+1);

% Ensure that H is an upper Hessenberg matrix.
assert(all(all(tril(H,-2) == 0)));

% Ensure that RH is an upper triangular matrix.
assert(all(all(tril(RH,-1) == 0)));

end

function tol = getTolerance(rtol, atol, x)
% This function returns the componentwise maximum of rtol * x and atol,
% where rtol and atol are numbers and x is a row vector.
tol = max([atol * ones(size(x)); rtol * x]);
end

function paperIdentityTests(V, W, Z, H, iter, A, P, M, tol)
% Verify, against the quantities returned by subgmres, the identities from the
% paper: the Arnoldi relation A P^{-1} V_k = V_{k+1} H_{k+1,k}
% (eq:Arnoldi-relation), the Hessenberg entries h_{j,l} = <v_j, A P^{-1} v_l>
% (eq:upper-Hessenberg-matrix-entries), the M-inverse-orthonormality of the dual
% Krylov basis V, the primal companions W = M^{-1} V and their
% M-orthonormality (eq:primal-companions-of-the-dual-Krylov-basis and the
% second statement of the lemma below), and, for the primal Krylov basis Z: the relation
% z_j = P^{-1} v_j (eq:relation-primal-and-dual-Krylov-bases) and its
% P'*inv(M)*P-orthonormality (Lemma lemma:orthonormal-basis-of-primal-Krylov-subspace).
% All inner products follow the paper's convention <r,s>_{M^{-1}} = r' * M^{-1} * s,
% which is antilinear in the *first* argument; the conjugate therefore sits
% outside the solve with M, and conj(M\v) differs from M\conj(v) for complex M.
applyPinv = @(y) P\y;
applyMinv = @(y) M\y;
k = iter;
if (k < 1)
	return
end

% A P^{-1} [v_1, ..., v_k] and the auxiliary basis w_j = M^{-1} v_j.
APV   = A * applyPinv(V(:,1:k));
MinvV = applyMinv(V(:,1:k+1));

% Arnoldi relation.
assert(norm(APV - V(:,1:k+1) * H(1:k+1,1:k), 'fro') <= tol * max(1, norm(APV, 'fro')));

% Rows of V that carry information. On a happy breakdown subgmres leaves
% V(:,k+1) at zero; on a near-breakdown it divides by an h_{k+1,k} of the order
% of eps*norm(H), so V(:,k+1) is normalized but consists of amplified roundoff.
% Neither case says anything about the identities, so drop that last vector.
last = k + 1;
if (abs(H(k+1,k)) <= 1e-10 * norm(H(1:k+1,1:k), 'fro'))
	last = k;
end

% Hessenberg entries h_{j,l} = <v_j, A P^{-1} v_l>_{M^{-1}}. The argument order
% matters: the opposite one would produce the entrywise conjugate of H.
MinvAPV = applyMinv(APV);
for l = 1:k
	for j = 1:min(l+1,last)
		assert(abs(H(j,l) - V(:,j)' * MinvAPV(:,l)) <= tol * max(1, abs(H(j,l))));
	end
end

% M-inverse-orthonormality of V.
G = V(:,1:last)' * MinvV(:,1:last);
assert(norm(G - eye(last), 'fro') <= tol);

% Primal companions W = M^{-1} V (eq:primal-companions-of-the-dual-Krylov-basis).
% By the second statement of the lemma they are M-orthonormal. They are a
% different basis from Z, the two coinciding only when P == M.
assert(norm(W(:,1:last) - MinvV(:,1:last), 'fro') <= tol * max(1, norm(MinvV(:,1:last), 'fro')));
assert(norm(W(:,1:last)' * M * W(:,1:last) - eye(last), 'fro') <= tol);

% Primal basis relation z_j = P^{-1} v_j.
PinvV = applyPinv(V(:,1:k));
assert(norm(Z(:,1:k) - PinvV, 'fro') <= tol * max(1, norm(PinvV, 'fro')));

% P'*inv(M)*P-orthonormality of the primal basis Z, inherited at no extra cost
% from the M-inverse-orthonormality of V (same lemma as above).
N = P' * (M \ P);
GZ = Z(:,1:k)' * N * Z(:,1:k);
assert(norm(GZ - eye(k), 'fro') <= tol);
end
