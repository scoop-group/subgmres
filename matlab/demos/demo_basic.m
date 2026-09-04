% demo_basic
% ------------------------------------------------------------------------
% The simplest call, and how to read what comes back.
%
% subgmres(A, b) with nothing else solves in the standard norm, without a
% preconditioner and without partitioning the residual -- the GMRES most users
% reach for first. This demo does that, then looks at the three things worth
% understanding before using anything fancier: what the defaults actually were,
% how to tell convergence from exhaustion, and what the returned iterate is
% worth when the iteration stopped short.
%
% Parallel to the Python demos/demo_basic.py.

addpath('..');

rng(0);
n = 200;
A = randn(n,n) + 3*sqrt(n)*eye(n);      % well conditioned
b = randn(n,1);
relativeResidual = @(x) norm(b - A*x) / norm(b);

%% 1. The simplest call there is.
[x, flag, ~, iter, ~, ~, ~, ~, ~, ~, ~, ~, toltable] = subgmres(A, b);
fprintf('1. subgmres(A, b), everything left at its default\n');
fprintf('   flag %d, %d iterations, ||b - A x|| / ||b|| = %.2e\n', ...
	flag, iter, relativeResidual(x));

%% 2. What the defaults were.
% toltable reports the tolerances that were actually in force, which is more
% useful than remembering them: one row per subvector, then one for the total
% residual, with the relative tolerance first and the absolute second. With no
% partition requested there is a single subvector -- the whole vector -- so
% there are two rows, and only the total carries the default rtol of 1e-6.
% maxiter defaults to round(min(n,100)), which is 100 here, not 200.
fprintf('\n2. the tolerances that were in force\n');
names = {'whole vector', 'total       '};
for i = 1:2
	fprintf('   %s  rtol %.1e   atol %.1e\n', names{i}, toltable(i,1), toltable(i,2));
end

%% 3. Convergence, and stopping short.
% flag is the one output to check before using x. 0 means the tolerances were
% met; 1 means the iteration budget ran out first, and then x is simply the
% best iterate found so far -- a perfectly good vector, just not yet accurate.
% It is not a failure code and x is not garbage: the residual below is the
% honest measure of what was achieved, and picking up where it left off is a
% matter of passing that x back as x0 (see demo_restarts).
fprintf('\n3. what happens when the budget runs out\n');
[xShort, flagShort, ~, iterShort] = subgmres(A, b, [], 1e-14, [], 8);
fprintf('   rtol 1e-14 in at most 8 iterations: flag %d, %d iterations, ||r||/||b|| = %.2e\n', ...
	flagShort, iterShort, relativeResidual(xShort));
[xResumed, flagResumed, ~, iterResumed] = subgmres(A, b, [], 1e-14, [], 8, [], [], xShort);
fprintf('   eight more from there:              flag %d, %d iterations, ||r||/||b|| = %.2e\n', ...
	flagResumed, iterResumed, relativeResidual(xResumed));
[xFinished, flagFinished, ~, iterFinished] = subgmres(A, b, [], 1e-14, [], n);
fprintf('   or simply a larger budget:          flag %d, %d iterations, ||r||/||b|| = %.2e\n', ...
	flagFinished, iterFinished, relativeResidual(xFinished));
