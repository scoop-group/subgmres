function A = spd(n)

% This function returns a random spd matrix A of size nxn.

A = randn(n,n); 
A = A + A';
A = A + 3*n*eye(n);

assert(all(eig(A) > 0));

