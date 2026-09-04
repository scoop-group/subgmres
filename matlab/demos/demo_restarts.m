% demo_restarts
% ------------------------------------------------------------------------
% Restarted GMRES.
%
% subgmres does not implement restarting, and deliberately so: because
% canonical GMRES is formulated for an arbitrary initial guess, a restarted
% method needs no machinery of its own. One re-invokes the solver every m steps
% with x0 set to the iterate the previous cycle produced. That loop is the whole
% feature, and it is written out below in eight lines.
%
% Restarting caps what the method has to store and orthogonalize against. Full
% GMRES holds the entire Krylov basis and orthogonalizes each new vector against
% all of it, so both grow without bound; GMRES(m) never holds more than m
% vectors. The price is paid in iterations, and if m is too small it is paid in
% convergence itself -- the last part below shows a restart length that never
% gets there at all.
%
% The system is a convection-diffusion operator on (-1,1)^2, discretized by
% centred finite differences, with the recirculating wind of the paper's Oseen
% problems. Nothing beyond the base language is needed.
%
% Parallel to the Python demos/demo_restarts.py.

addpath('..');

n = 24;
[A, P, meshPeclet] = convectionDiffusion(n, 1/20);
b = ones(size(A,1), 1);
target = 1e-8 * norm(b);
relativeResidual = @(x) norm(b - A*x) / norm(b);
fprintf('convection-diffusion on %d x %d interior points: %d unknowns\n', ...
	n, n, size(A,1));
fprintf('mesh Peclet number %.1f\n', meshPeclet);

%% 1. What restarting costs, and what it buys.
% The Krylov dimension held at once is the memory: full GMRES ends up holding
% one vector per iteration, GMRES(m) never more than m. (subgmres stores both
% the dual basis V and its primal companions W, so the vectors in flight are
% twice the figure below in each case.)
[xFull, flagFull, ~, iterFull] = subgmres(A, b, [], inf, target, size(A,1));
fprintf('\n1. no preconditioner\n');
fprintf('   %12s %12s %8s %12s %13s\n', 'method', 'iterations', 'cycles', ...
	'basis held', '||r||/||b||');
fprintf('   %12s %12d %8s %12d %13.1e\n', 'full GMRES', iterFull, '-', ...
	iterFull + 1, relativeResidual(xFull));
for m = [40 20 10]
	[x, iterations, cycles, converged] = restartedGMRES(A, b, m, target, 200, []);
	fprintf('   %12s %12d %8d %12d %13.1e\n', sprintf('GMRES(%d)', m), ...
		iterations, cycles, m + 1, relativeResidual(x));
end

%% 2. The preconditioner and inner product carry over unchanged.
% Nothing in the loop knows about P: it is passed straight through to each
% cycle, exactly as an unrestarted call would pass it. An M would go the same
% way. With a preconditioner that actually works, the iteration counts collapse
% and restarting stops costing anything much -- which is the usual reason one
% can afford a short restart length in practice.
[xFullP, flagFullP, ~, iterFullP] = subgmres(A, b, [], inf, target, size(A,1), P);
fprintf('\n2. same runs, preconditioned by the diffusion part alone\n');
fprintf('   %12s %12d %8s %12d %13.1e\n', 'full GMRES', iterFullP, '-', ...
	iterFullP + 1, relativeResidual(xFullP));
for m = [20 10]
	[x, iterations, cycles, converged] = restartedGMRES(A, b, m, target, 200, P);
	fprintf('   %12s %12d %8d %12d %13.1e\n', sprintf('GMRES(%d)', m), ...
		iterations, cycles, m + 1, relativeResidual(x));
end

%% 3. Restarting can also fail.
% Halve the diffusion and the same GMRES(10) no longer converges at all. The
% information a cycle would have needed is exactly what the restart discarded,
% and no number of cycles recovers it -- full GMRES on this operator still
% reaches the target.
[Ahard, ~, pecletHard] = convectionDiffusion(n, 1/40);
bHard = ones(size(Ahard,1), 1);
targetHard = 1e-8 * norm(bHard);
[xHardFull, ~, ~, iterHardFull] = subgmres(Ahard, bHard, [], inf, targetHard, size(Ahard,1));
[xHard, iterations, cycles, converged] = restartedGMRES(Ahard, bHard, 10, targetHard, 50, []);
fprintf('\n3. the same operator at mesh Peclet %.1f, where a short restart is not enough\n', ...
	pecletHard);
fprintf('   full GMRES  converged in %d iterations\n', iterHardFull);
if converged
	verdict = 'converged';
else
	verdict = 'did NOT converge';
end
fprintf('   GMRES(10)   %s after %d cycles, %d iterations, stuck at ||r||/||b|| = %.1e\n', ...
	verdict, cycles, iterations, norm(bHard - Ahard*xHard) / norm(bHard));
