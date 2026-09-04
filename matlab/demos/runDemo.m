function ok = runDemo(demoFile)
% Run one demo script in a fresh workspace, quietly, reporting only whether it
% completed. Helper for run_all_demos; a function of its own both because
% MATLAB and Octave disagree on where a script's local functions may live, and
% because a function call is the portable way to give each demo a workspace of
% its own -- run in a shared one, demos would overwrite each other's variables
% and could fail, or pass, for reasons that have nothing to do with them.
try
	evalc('run(demoFile)');
	ok = true;
catch err
	fprintf('          %s\n', err.message);
	ok = false;
end
end
