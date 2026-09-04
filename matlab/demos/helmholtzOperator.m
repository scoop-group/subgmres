function A = helmholtzOperator(n, k, shift)
% The one-dimensional Helmholtz operator of demo_Helmholtz, or its
% complex-shifted variant: -u'' - (1 + i*shift) k^2 u on (0,1), Dirichlet at the
% left end and a first-order Sommerfeld radiation condition at the right,
% imposed through a ghost point so that the last row stays second order.
%
% A file of its own rather than a local function of the script: MATLAB wants
% those at the end, Octave only sees them once execution has passed the
% definition, and no placement satisfies both.
h = 1/n;
wavenumber = (1 + 1i*shift) * k^2;
main = (2/h^2 - wavenumber) * ones(n,1);
off = -ones(n,1) / h^2;
A = spdiags([off, main, off], -1:1, n, n);
% Sommerfeld at x = 1, via the ghost value u(n+1) = u(n-1) + 2*i*k*h*u(n).
A(n, n-1) = -2/h^2;
A(n, n) = 2/h^2 - wavenumber - 2i*k/h;
end
