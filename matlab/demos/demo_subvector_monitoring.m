% demo_subvector_monitoring
% ------------------------------------------------------------------------
% Subvector residual monitoring -- the paper's namesake feature -- on a Stokes
% saddle point discretized by finite differences.
%
% The system is assembled here, in a dozen lines, on a staggered (MAC) grid:
% velocities on cell faces, pressures at cell centres, giving [A B'; B 0] with
% A the vector Laplacian and B the discrete divergence. Nothing beyond the base
% language is needed -- the drivers that reproduce the paper's actual
% experiments use Firedrake, but a demo should not.
%
% The demo shows three things:
%   1. the residual subvectors converging under their own relative tolerances,
%      in the inner product induced by M = blkdiag(A, Mp) -- the H1 seminorm on
%      velocity, the L2 mass on pressure, which is the paper's Stokes choice;
%   2. why one would bother: in the standard inner product the total residual
%      norm meets the same tolerance while the pressure block is left far less
%      converged, and monitoring the total alone cannot see it;
%   3. that the subvectors need not be contiguous -- the same solve, with the
%      unknowns interleaved and the partition passed as explicit index sets.
%
% The pressure is left unpinned, so the system is singular: a constant pressure
% is a nullvector. That is deliberate. The right hand side is consistent by
% construction, and the iterates never leave the range of the operator, so the
% computed pressure comes out with zero mean by itself -- the space the paper
% works in, obtained for free rather than imposed.
%
% Parallel to the Python demos/demo_subvector_monitoring.py.

addpath('..');

n = 16;
h = 1/n;
% Dirichlet second difference, and its wall variant: at a wall the boundary
% sits half a cell from the first unknown rather than a whole one, which raises
% the diagonal of the two end rows from 2 to 3.
secondDifference = @(m) spdiags([-ones(m,1) 2*ones(m,1) -ones(m,1)], -1:1, m, m);
atWall = secondDifference(n);
atWall(1,1) = 3;
atWall(n,n) = 3;
% u lives on the (n-1) x n vertical faces, v on the n x (n-1) horizontal ones;
% each is walled in one direction and Dirichlet in the other.
Lu = (kron(speye(n), secondDifference(n-1)) + kron(atWall, speye(n-1))) / h^2;
Lv = (kron(speye(n-1), atWall) + kron(secondDifference(n-1), speye(n))) / h^2;
% Divergence: each cell differences the two faces bracketing it.
faces = spdiags([-ones(n,1) ones(n,1)], [-1 0], n, n-1);
B = [kron(speye(n), faces), kron(faces, speye(n))] / h;
A = blkdiag(Lu, Lv);
K = [A, B'; B, sparse(size(B,1), size(B,1))];
% These are pointwise difference operators, so the Schur complement is O(1) and
% the pressure mass matrix is the identity; the h^2 of a finite element mass
% matrix would be wrong here by exactly h^-2.
M = blkdiag(A, speye(size(B,1)));
nVelocity = size(A,1);
nPressure = size(B,1);
nTotal = nVelocity + nPressure;

% A manufactured right hand side. Taking b = K*xExact makes it consistent by
% construction -- the alternative, a physical b with no source in the continuity
% equation, has an exactly zero pressure residual to begin with, against which a
% *relative* tolerance would be vacuous.
rng(0);
xExact = randn(nTotal, 1);
xExact(nVelocity+1:end) = xExact(nVelocity+1:end) - mean(xExact(nVelocity+1:end));
b = K * xExact;

constantPressure = [zeros(nVelocity,1); ones(nPressure,1)];
fprintf('Stokes, %d x %d cells: %d unknowns (%d velocity, %d pressure)\n', ...
	n, n, nTotal, nVelocity, nPressure);
fprintf('  a constant pressure is a nullvector: ||K z|| / ||z|| = %.1e\n', ...
	norm(K * constantPressure) / norm(constantPressure));
fprintf('  the right hand side is consistent:   |z.b| / ||b||   = %.1e\n', ...
	abs(constantPressure' * b) / norm(b));

%% 1. Monitoring the two blocks, each under its own relative tolerance.
% ix holds the starting index of each subvector; the tolerance vector carries
% one entry per subvector plus a final one for the total residual.
ix = [1 nVelocity+1];
rtol = [1e-9; 1e-7; 1e-8];
[x, flag, Rsubnorms, iter, RsubnormsHIST, ~, ~, ~, ~, ~, ~, ~, toltable] = ...
	subgmres(K, b, ix, rtol, [], [], M, M);
fprintf('\n1. proper inner product M = blkdiag(A, Mp)   ->  %d iterations, flag %d\n', ...
	iter, flag);
fprintf('   requested [rtol, atol] per subvector, then the total:\n');
names = {'velocity', 'pressure', 'total   '};
for i = 1:3
	fprintf('      %s  rtol %.1e   atol %.1e\n', names{i}, toltable(i,1), toltable(i,2));
end
fprintf('   %10s %12s %12s %12s\n', 'iteration', 'velocity', 'pressure', 'total');
for k = [0:max(1,floor(iter/5)):iter-1, iter]
	fprintf('   %10d %12.3e %12.3e %12.3e\n', k, RsubnormsHIST(1,k+1), ...
		RsubnormsHIST(2,k+1), RsubnormsHIST(3,k+1));
end
fprintf('   pressure mean, never imposed: %.1e\n', mean(x(nVelocity+1:end)));
fprintf('   error against the exact solution: %.2e\n', norm(x - xExact) / norm(xExact));

%% 2. Why monitor the blocks at all.
% Two solves stopped on the total residual alone, differing only in the inner
% product that measures it. Both meet the tolerance they were given. In the
% standard inner product the two blocks are nowhere near equally converged, and
% a total norm cannot reveal that; in the M-inverse norm, which weighs the
% blocks commensurately, they come out together. This is the same knob as
% demo_custom_inner_product, seen per block rather than in aggregate.
fprintf('\n2. what a total norm hides. Same solve, same tolerance 1e-8 on the\n');
fprintf('   total residual and nothing else, only the inner product differs:\n');
innerProducts = {M, []};
labels = {'M-inverse norm', 'standard norm '};
for i = 1:2
	[xi, flagi, ~, iteri] = subgmres(K, b, [], 1e-8, [], [], M, innerProducts{i});
	residual = b - K * xi;
	fprintf('      %s %3d iterations   velocity %.2e   pressure %.2e\n', ...
		labels{i}, iteri, ...
		norm(residual(1:nVelocity)) / norm(b(1:nVelocity)), ...
		norm(residual(nVelocity+1:end)) / norm(b(nVelocity+1:end)));
end

%% 3. Subvectors need not be contiguous.
% Real assemblies often number unknowns by cell rather than by field, leaving
% each field scattered through the vector. Interleave the two blocks and pass
% the partition as explicit index sets instead of starting indices; nothing else
% changes, and neither does the answer.
[~, order] = sort([linspace(0,1,nVelocity), linspace(0,1,nPressure)]);
scattered = {find(order <= nVelocity), find(order > nVelocity)};
[xp, flagp, RsubnormsP, iterp] = subgmres(K(order,order), b(order), scattered, ...
	rtol, [], [], M(order,order), M(order,order));
fprintf('\n3. same system, unknowns interleaved, partition as index sets   ->  %d iterations, flag %d\n', ...
	iterp, flagp);
fprintf('   velocity indices begin %s ... (not contiguous)\n', ...
	mat2str(scattered{1}(1:5)));
fprintf('   subvector norms agree with part 1 to %.1e\n', ...
	max(abs(RsubnormsP - Rsubnorms)));
