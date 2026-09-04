% demo_custom_inner_product
% ------------------------------------------------------------------------
% Canonical GMRES in a non-standard inner product -- the paper's central point:
% the norm in which the residual is minimized need not be the standard one, and
% choosing it is separate from choosing a preconditioner.
%
% Passing M changes which residual is made smallest at every step, and so the
% whole path taken to the solution, without changing the system A x = b at all.
% The two runs below differ in exactly one argument. What they produce is not
% "better" and "worse" convergence but convergence measured in two different
% currencies: each run is optimal in its own norm at every iteration and is
% beaten by the other in the other norm. The table shows both histories in both
% norms, which is the only fair way to look at them.
%
% Parallel to the Python demos/demo_custom_inner_product.py.

addpath('..');

rng(1);
n = 100;
A = randn(n,n) + 3*sqrt(n)*eye(n);
b = randn(n,1);

% A diagonal weighting spanning six orders of magnitude: an extreme choice, so
% that the divergence between the two histories is unmistakable.
weights = logspace(-3, 3, n)';
M = spdiags(weights, 0, n, n);

% Both runs ask for the residual history. The norms each run monitors for
% itself are in its own norm alone, and the comparison needs both residuals in
% both norms, so the residual vectors are what is wanted here.
[xStandard, flagStandard, ~, iterStandard, ~, ~, RstandardHIST] = ...
	subgmres(A, b, [], 1e-10, [], n);
[xWeighted, flagWeighted, ~, iterWeighted, ~, ~, RweightedHIST] = ...
	subgmres(A, b, [], 1e-10, [], n, [], M);
fprintf('standard inner product: %d iterations, flag %d\n', iterStandard, flagStandard);
fprintf('M-weighted:             %d iterations, flag %d\n', iterWeighted, flagWeighted);

standardNorm = @(R) sqrt(sum(abs(R).^2, 1));
mInverseNorm = @(R) sqrt(sum(abs(R).^2 ./ weights, 1));

% Compare over the iterations both runs reached.
common = min(iterStandard, iterWeighted) + 1;
standardInStandard = standardNorm(RstandardHIST(:,1:common));
weightedInStandard = standardNorm(RweightedHIST(:,1:common));
standardInMinverse = mInverseNorm(RstandardHIST(:,1:common));
weightedInMinverse = mInverseNorm(RweightedHIST(:,1:common));

fprintf('\n%5s | %29s | %30s\n', '', 'measured in the standard norm', ...
	'measured in the M-inverse norm');
fprintf('%5s | %14s %14s | %14s %15s\n', 'k', 'standard run', 'M-weighted run', ...
	'standard run', 'M-weighted run');
for k = [0:max(1,floor(common/7)):common-2, common-1]
	fprintf('%5d | %14.3e %14.3e | %14.3e %15.3e\n', k, ...
		standardInStandard(k+1), weightedInStandard(k+1), ...
		standardInMinverse(k+1), weightedInMinverse(k+1));
end

% Neither run is uniformly better. Each is the smaller one in the norm it was
% asked to minimize, at every single iteration -- which is what "minimizing the
% residual" means once one has said in which norm.
fprintf('\nstandard run never beaten in the standard norm:   %d\n', ...
	all(standardInStandard <= weightedInStandard));
fprintf('M-weighted run never beaten in the M-inverse norm: %d\n', ...
	all(weightedInMinverse <= standardInMinverse));

% Both solve the same system, so both end with a small true residual; the
% choice of norm decides the route, not the destination.
fprintf('  [standard  ] ||b - A x|| / ||b|| = %.3e\n', norm(b - A*xStandard) / norm(b));
fprintf('  [M-weighted] ||b - A x|| / ||b|| = %.3e\n', norm(b - A*xWeighted) / norm(b));
