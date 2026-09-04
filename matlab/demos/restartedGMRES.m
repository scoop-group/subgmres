function [x, iterations, cycles, converged] = restartedGMRES(A, b, restartLength, target, maxCycles, P)
% GMRES(restartLength): re-invoke the solver from where it left off.
%
% The stopping criterion is absolute. It has to be: each cycle measures its own
% relative tolerance against its own starting residual, so a relative criterion
% would ask for a further reduction by that factor every cycle rather than the
% one reduction that was wanted overall. Passing rtol = Inf and atol = target
% states the global goal once.
x = zeros(size(A,1), 1);
iterations = 0;
for cycles = 1:maxCycles
	% P is passed straight through, and x0 is the previous cycle's iterate --
	% which is the entire mechanism. Pass P explicitly, empty for none: leaving
	% it to varargin would slide x0 into the inner-product slot when absent.
	[x, flag, ~, iter] = subgmres(A, b, [], inf, target, restartLength, P, [], x);
	iterations = iterations + iter;
	if flag == 0
		converged = true;
		return
	end
end
converged = false;
end

% Helper for demo_restarts; see the note in convectionDiffusion.m on why this
% is a file of its own.
