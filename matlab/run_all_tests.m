% run_all_tests
% ------------------------------------------------------------------------
% Run the MATLAB/Octave test suite.
%
% An entry point whose name says what it does. The tests themselves are in
% test_subgmres.m, a script-based test file in MATLAB's sense: each %% section
% is one independent test. MATLAB runs such a file with its built-in
% runtests(); Octave has no runtests, so runtests.m in this directory is a shim
% that emulates enough of it. Either way this script is the thing to run, and
% neither of those two files needs to be understood to do so.
%
% Exits with a non-zero status under `octave run_all_tests.m` if any test
% fails, and raises under MATLAB, so it can serve as a release-checklist step.

here = fileparts(mfilename('fullpath'));
if isempty(here)
	here = pwd;
end
originalDirectory = pwd;
cd(here);
results = runtests('test_subgmres');
cd(originalDirectory);

failed = sum(~[results.Passed]);
if failed > 0
	if exist('OCTAVE_VERSION', 'builtin')
		exit(1);
	else
		error('run_all_tests:failures', '%d test(s) failed.', failed);
	end
end
