function results = runtests(name)
% RUNTESTS  Minimal cross-engine shim for MATLAB's script-based runtests.
%   RUNTESTS(NAME) runs the script-based test file NAME (e.g. 'test_subgmres').
%
%   Under MATLAB this simply defers to the built-in test runner (this file only
%   shadows runtests on the path). Under Octave -- which has no runtests, and which
%   cannot see a script's local functions -- it emulates the behaviour to a
%   limited degree: it registers the test file's local functions, executes the
%   shared setup (all code before the first %% section) once per test, and then
%   runs each %%-delimited section as an independent test, reporting a pass/fail
%   count. It does not reproduce the full MATLAB unit-testing framework.

	if ~exist('OCTAVE_VERSION', 'builtin')
		% Running under MATLAB: defer to the genuine test runner.
		%
		% Do NOT call runtests() here. This file shadows it, and the shadowing
		% cannot be shaken off by taking this folder off the path: MATLAB
		% searches the CURRENT FOLDER before the path, so whenever MATLAB runs
		% from this folder -- as run_tests_Heidelberg.sh does -- the call would
		% resolve straight back into this shim and recurse. (The symptom is a
		% flood of "In runtests (line ...)" warning traces, one per level, from
		% the repeated rmpath of an already-removed folder.) Taking this folder
		% off the path would also hide subgmres.m and spd.m from the tests.
		%
		% Instead go directly to the matlab.unittest API that runtests() itself
		% is a thin wrapper around. It resolves by class name, not by file
		% lookup, so no shadowing is involved and the path is left untouched.
		testfile = which(name);
		if isempty(testfile)
			testfile = name;
		end
		suite = matlab.unittest.TestSuite.fromFile(testfile);
		runner = matlab.unittest.TestRunner.withTextOutput();
		results = runner.run(suite);
		return
	end

	% ----- Octave emulation -----

	% Locate and read the test file.
	testfile = name;
	if numel(testfile) < 2 || ~strcmp(testfile(end-1:end), '.m')
		testfile = [testfile '.m'];
	end
	if exist(testfile, 'file') ~= 2
		testfile = which(name);
	end
	if isempty(testfile) || exist(testfile, 'file') ~= 2
		error('runtests:fileNotFound', 'Could not find test file "%s".', name);
	end
	lines = strsplit(fileread(testfile), sprintf('\n'), 'collapsedelimiters', false);

	% Split off the local function definitions (assumed to sit at the end, as in
	% a MATLAB script-based test file).
	funcStart = numel(lines) + 1;
	for i = 1:numel(lines)
		if ~isempty(regexp(lines{i}, '^\s*function\s', 'once'))
			funcStart = i;
			break
		end
	end
	scriptLines = lines(1:funcStart-1);
	funcSource  = strjoin(lines(funcStart:end), sprintf('\n'));

	% Register the local functions in this workspace.
	if ~isempty(strtrim(funcSource))
		eval(funcSource);
	end

	% Find the %% section boundaries within the script part.
	isSection = ~cellfun(@isempty, regexp(scriptLines, '^\s*%%', 'once'));
	secIdx = find(isSection);
	if isempty(secIdx)
		sharedSource = '';
		blocks = {scriptLines};
		names  = {name};
	else
		sharedSource = strjoin(scriptLines(1:secIdx(1)-1), sprintf('\n'));
		bounds = [secIdx(:); numel(scriptLines)+1];
		blocks = cell(numel(secIdx), 1);
		names  = cell(numel(secIdx), 1);
		for k = 1:numel(secIdx)
			blk = scriptLines(bounds(k):bounds(k+1)-1);
			blocks{k} = strjoin(blk, sprintf('\n'));
			names{k}  = strtrim(regexprep(blk{1}, '^\s*%%\s*', ''));
			if isempty(names{k}), names{k} = sprintf('Section %d', k); end
		end
	end

	% Run each section as an independent test, re-running the shared setup first
	% so that shared variables start fresh (as MATLAB does).
	npass = 0; nfail = 0;
	results = struct('Name', {}, 'Passed', {}, 'Message', {});
	for k = 1:numel(blocks)
		[ok, msg] = evalSection(sharedSource, blocks{k});
		if ok
			npass = npass + 1;
			printf('  PASS  %s\n', names{k});
		else
			nfail = nfail + 1;
			printf('  FAIL  %s\n          %s\n', names{k}, msg);
		end
		results(k) = struct('Name', names{k}, 'Passed', ok, 'Message', msg);
	end
	printf('\n  %d passed, %d failed (%d total)\n', npass, nfail, numel(blocks));
end

function [ok, msg] = evalSection(sharedSource, blockSource)
% Evaluate the shared setup and ONE %% section in a fresh function workspace.
%
% The fresh workspace is the point: MATLAB's script-based test runner gives each
% test the shared setup and nothing else, so a variable defined in one section
% is NOT visible in the next. Evaluating every section in one common workspace
% (as this shim first did) is more permissive than MATLAB and silently hides
% exactly that class of bug -- it did hide one, a tolerance defined in one
% section and used in another, which only surfaced on the MATLAB run.
%
% The test file's local functions are registered by the caller via eval(), which
% in Octave defines them as command-line functions; those are globally visible,
% so this workspace still sees them.
	ok = true; msg = '';
	try
		if ~isempty(strtrim(sharedSource)), eval(sharedSource); end
		eval(blockSource);
	catch err
		ok = false; msg = err.message;
	end
end
