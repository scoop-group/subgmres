% demo_Helmholtz
% ------------------------------------------------------------------------
% A complex-valued Helmholtz problem, solved with a preconditioner.
%
% Two things at once, because they belong together. A preconditioner P is passed
% as one more argument, and subgmres uses its inverse action -- given a matrix it
% solves against it for you (demo_operator_forms covers the other form, where
% you supply the solve yourself). And everything works over the complex field:
% A, b, the Krylov basis and the iterates are complex here, while the residual
% norms the solver reports remain real, as norms must.
%
% The system is a one-dimensional Helmholtz problem on (0, 1) -- see
% helmholtzOperator.m. This is the standard hard case for Krylov methods: the
% operator is indefinite, and unpreconditioned GMRES gets nowhere within any
% sensible budget. The preconditioner is the complex-shifted Laplacian, the same
% discretization with k^2 replaced by (1 + i/2) k^2, the textbook remedy.
%
% Parallel to the Python demos/demo_Helmholtz.py.

addpath('..');

n = 400;
k = 40.0;
A = helmholtzOperator(n, k, 0);
b = zeros(n,1);
b(round(n/4)) = n;                      % a point source a quarter of the way in
relativeResidual = @(x) norm(b - A*x) / norm(b);
fprintf('1-D Helmholtz: %d unknowns, k = %.0f, %.0f points per wavelength\n', ...
	n, k, 2*pi*n/k);
fprintf('A is complex: %d\n', ~isreal(A));

% Unpreconditioned, with a budget a quarter the size of the problem.
[xPlain, flagPlain, ~, iterPlain] = subgmres(A, b, [], 1e-8, [], 100);
fprintf('\n   no preconditioner:   flag %d, %d iterations, ||r||/||b|| = %.2e\n', ...
	flagPlain, iterPlain, relativeResidual(xPlain));

% The same solve with the shifted Laplacian passed as P. Nothing else changes.
P = helmholtzOperator(n, k, 0.5);
[xPrecond, flagPrecond, Rsubnorms, iterPrecond] = subgmres(A, b, [], 1e-8, [], 100, P);
fprintf('   shifted Laplacian:   flag %d, %d iterations, ||r||/||b|| = %.2e\n', ...
	flagPrecond, iterPrecond, relativeResidual(xPrecond));

% The solution is genuinely complex -- a travelling wave, not a standing one --
% and the reported residual norms are real regardless.
fprintf('\n   solution is complex:   max |Re| = %.3e,  max |Im| = %.3e\n', ...
	max(abs(real(xPrecond))), max(abs(imag(xPrecond))));
fprintf('   reported norms are real: %d, final total %.3e\n', ...
	isreal(Rsubnorms), Rsubnorms(end));
