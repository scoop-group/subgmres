% demo_residual_accuracy
% ------------------------------------------------------------------------
% How far the monitored residual norm can be trusted.
%
% subgmres reports residual norms it never computes a residual for. They come
% out of the Givens recursion, at no cost -- which is what makes subvector
% monitoring cheap enough to be worth having -- and the last row of
% RSUBNORMSHIST is the norm of the residual the recursion carries along, not of
% b - A x.
%
% In exact arithmetic the two agree. In floating point they drift apart, and the
% drift is not random: the relative gap between them grows like the reciprocal
% of the relative residual reached. The last column below is that statement as
% an experiment -- the product of the two stays at roughly machine epsilon over
% eleven orders of magnitude, while the gap itself climbs from 1e-16 to a factor
% of two.
%
% The practical reading: a stopping tolerance of 1e-6 or 1e-8 is measured
% faithfully, which is the range convergence criteria live in. Ask for 1e-14 and
% the number reported is no longer the residual you have. Nothing here is a
% defect -- it is the reason the solver also offers RHIST, which pays one
% application of A per iteration to ask the system rather than the recursion.
%
% Parallel to the Python demos/demo_residual_accuracy.py.

addpath('..');

rng(5);
n = 120;
A = randn(n,n) + 3*sqrt(n)*eye(n);
b = randn(n,1);

% Driven past any sensible stopping tolerance, to where the two disagree: rtol
% and atol are both zero, so only the iteration budget stops it.
[x, flag, ~, iter, RsubnormsHIST, ~, RHIST] = subgmres(A, b, [], 0, 0, 32);
monitored = RsubnormsHIST(end,:);
trueNorms = sqrt(sum(abs(RHIST).^2, 1));
relative = trueNorms / trueNorms(1);
gap = abs(monitored - trueNorms) ./ trueNorms;

fprintf('condition number of A: %.2f, machine epsilon: %.1e\n', cond(A), eps);
fprintf('\n%4s %13s %13s %13s %10s %11s\n', 'k', 'monitored', 'true', ...
	'rel.residual', 'gap', 'gap x rel');
for k = 0:2:iter
	fprintf('%4d %13.4e %13.4e %13.2e %10.1e %11.1e\n', k, monitored(k+1), ...
		trueNorms(k+1), relative(k+1), gap(k+1), gap(k+1)*relative(k+1));
end

% Where a stopping criterion would actually have fired, the monitored norm is
% exact to within a few ulp; it is only past that point that it stops meaning
% anything.
for tolerance = [1e-6 1e-8 1e-12]
	k = find(relative <= tolerance, 1) - 1;
	fprintf('\nstopping at rtol %.0e would fire at iteration %d, where the monitored norm is off by %.1e\n', ...
		tolerance, k, gap(k+1));
end
