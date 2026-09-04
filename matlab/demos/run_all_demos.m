% run_all_demos
% ------------------------------------------------------------------------
% Smoke check: every demo still runs.
%
% Not a test, and deliberately not part of test_subgmres. The suite asserts
% that the solver is correct; this asserts only that the demonstrations still
% execute after an API change, which is a different question and one whose
% failures should not be reported as correctness failures. It checks no numbers
% and makes no claim about the output.
%
% Exits with a non-zero status under `octave run_all_demos.m` if any demo
% fails, so it can serve as a release-checklist step.
%
% Parallel to the Python demos/run_all_demos.py.

here = fileparts(mfilename('fullpath'));
if isempty(here)
	here = pwd;
end
originalDirectory = pwd;
% The demos addpath('..') relative to the working directory, so run from theirs.
cd(here);
addpath('..');

files = dir(fullfile(here, 'demo_*.m'));
failures = {};
for i = 1:numel(files)
	if runDemo(fullfile(here, files(i).name))
		fprintf('  ok    %s\n', files(i).name);
	else
		fprintf('  FAIL  %s\n', files(i).name);
		failures{end+1} = files(i).name;
	end
end

fprintf('\n  %d of %d demos ran', numel(files) - numel(failures), numel(files));
if isempty(failures)
	fprintf('\n');
else
	fprintf('; failed: %s\n', strjoin(failures, ', '));
end
cd(originalDirectory);
if ~isempty(failures) && ~exist('OCTAVE_VERSION', 'builtin')
	error('run_all_demos:failures', '%d demo(s) failed.', numel(failures));
elseif ~isempty(failures)
	exit(1);
end
