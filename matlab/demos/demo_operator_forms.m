% demo_operator_forms
% ------------------------------------------------------------------------
% The forms A, P and M may take, and what subgmres assumes about each.
%
% Any of the three may be handed over as a matrix, sparse or dense, or as a
% function handle. The one thing worth knowing is that a matrix and a function
% do not mean the same thing for P and M:
%
%   * as a matrix, P (or M) is the operator itself, and subgmres solves against
%     it -- what the iteration needs is the inverse action, and it obtains that
%     for you;
%   * as a function, you supply that inverse action yourself. The handle must
%     return P \ x, not P * x.
%
% Getting this backwards is a quiet mistake: the iteration still runs, and
% merely fails to converge, so the demo checks the two forms against each other
% rather than asserting that either looks plausible on its own.
%
% The system is a one-dimensional convection-diffusion operator with a shift,
% which gives the last part something to pass around: a parameter the operators
% depend on. Trailing arguments to subgmres are forwarded to AFUN, PFUN and
% MFUN, so a parameterized operator needs no wrapper -- and every one of them
% must then accept those arguments, whether or not it uses them. The Python
% port has no such mechanism and needs none, a closure or functools.partial
% carrying the parameter instead; it also accepts a scipy LinearOperator as a
% third form of A, which has no MATLAB counterpart.
%
% Parallel to the Python demos/demo_operator_forms.py.

addpath('..');

n = 400;
h = 1/(n+1);
wind = 4.0;
shift = 0.5;
rtol = 1e-10;

% The operator, as a sparse matrix: -u'' + wind*u' + shift*u.
diffusion = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)], -1:1, n, n) / h^2;
convection = spdiags([-ones(n,1) ones(n,1)], [-1 1], n, n) * (wind/(2*h));
Amatrix = diffusion + convection + shift*speye(n);
b = ones(n,1);
relativeResidual = @(x) norm(b - Amatrix*x) / norm(b);

%% 1. A as a sparse matrix and as a function handle.
% The handle never assembles anything: it applies the stencil in place. Both
% runs are preconditioned identically, the subject here being the form of A.
applyOperator = @(x, shift) (2/h^2 + shift)*x ...
	+ [(-1/h^2 + wind/(2*h))*x(2:end); 0] ...
	+ [0; (-1/h^2 - wind/(2*h))*x(1:end-1)];
fprintf('1. the same A in two forms, preconditioned identically\n');
[xMatrix, flagMatrix, ~, iterMatrix] = subgmres(Amatrix, b, [], rtol, [], n, diffusion);
[xHandle, flagHandle, ~, iterHandle] = subgmres(@(x) applyOperator(x, shift), b, ...
	[], rtol, [], n, diffusion);
fprintf('   %16s: %3d iterations, flag %d, ||r||/||b|| = %.2e\n', ...
	'sparse matrix', iterMatrix, flagMatrix, relativeResidual(xMatrix));
fprintf('   %16s: %3d iterations, flag %d, ||r||/||b|| = %.2e\n', ...
	'function handle', iterHandle, flagHandle, relativeResidual(xHandle));
fprintf('   the two agree to %.1e\n', norm(xMatrix - xHandle));

%% 2. P and M as matrices, and as functions supplying the inverse action.
% The diffusion part preconditions this operator well. Handed over as a matrix
% it is factorized internally; handed over as a handle it must arrive already
% inverted, which is what the Cholesky solve below does. The weighting M is
% diagonal, so its inverse action is a division.
weights = logspace(-2, 2, n)';
Mmatrix = spdiags(weights, 0, n, n);
cholFactor = chol(diffusion);
diffusionSolve = @(x) cholFactor \ (cholFactor' \ x);
fprintf('\n2. P and M as matrices, then as functions returning the inverse action\n');
[xAsMatrices, flagAsMatrices, ~, iterAsMatrices] = ...
	subgmres(Amatrix, b, [], rtol, [], n, diffusion, Mmatrix);
[xAsFunctions, flagAsFunctions, ~, iterAsFunctions] = ...
	subgmres(Amatrix, b, [], rtol, [], n, diffusionSolve, @(x) x ./ weights);
fprintf('   %12s: %3d iterations, flag %d, ||r||/||b|| = %.2e\n', ...
	'as matrices', iterAsMatrices, flagAsMatrices, relativeResidual(xAsMatrices));
fprintf('   %12s: %3d iterations, flag %d, ||r||/||b|| = %.2e\n', ...
	'as functions', iterAsFunctions, flagAsFunctions, relativeResidual(xAsFunctions));
fprintf('   the two agree to %.1e\n', norm(xAsMatrices - xAsFunctions));

%% 3. Operators that depend on a parameter.
% Everything after x0 is forwarded to AFUN, PFUN and MFUN alike -- all three
% arguments here, not just the one the operator wants. PFUN below ignores them
% entirely and still has to accept them, which `varargin` in the handle is the
% tidy way to do; a handle written @(x, ~) would take exactly one extra and
% fail with "called with too many inputs" on the other two.
fprintf('\n3. a parameterized operator, the shift forwarded by subgmres\n');
[xForwarded, flagForwarded, ~, iterForwarded] = subgmres(@applyOperatorShifted, b, ...
	[], rtol, [], n, @(x, varargin) diffusionSolve(x), [], [], shift, h, wind);
[xPlain, flagPlain, ~, iterPlain] = subgmres(Amatrix, b, [], rtol, [], n, diffusion);
fprintf('   forwarded shift = %.1f:  %d iterations, flag %d\n', ...
	shift, iterForwarded, flagForwarded);
fprintf('   the assembled matrix:  %d iterations, flag %d\n', iterPlain, flagPlain);
fprintf('   the two agree to %.1e\n', norm(xForwarded - xPlain));
