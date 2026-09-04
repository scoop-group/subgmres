function [A, P, meshPeclet] = convectionDiffusion(n, epsilon)
% -epsilon*Laplace(u) + w.grad(u) on (-1,1)^2, n x n interior points. Returns
% also the diffusion part alone, a textbook preconditioner for this operator.
h = 2 / (n + 1);
axis = -1 + h * (1:n);
[X, Y] = ndgrid(axis, axis);
% The recirculating wind of the paper's Oseen cavity, divergence free and
% tangential to the boundary.
windX = 2 * Y(:) .* (1 - X(:).^2);
windY = -2 * X(:) .* (1 - Y(:).^2);
I = speye(n);
second = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)], -1:1, n, n) / h^2;
first = spdiags([-ones(n,1) ones(n,1)], [-1 1], n, n) / (2*h);
laplacian = kron(I, second) + kron(second, I);
A = epsilon * laplacian ...
	+ spdiags(windX, 0, n*n, n*n) * kron(I, first) ...
	+ spdiags(windY, 0, n*n, n*n) * kron(first, I);
P = epsilon * laplacian;
meshPeclet = max(hypot(windX, windY)) * h / (2 * epsilon);
end

% Helper for demo_restarts. Kept in its own file rather than as a local
% function of the script: MATLAB wants local functions at the end of a script,
% Octave only sees them once execution has passed the definition, and there is
% no placement that satisfies both.
