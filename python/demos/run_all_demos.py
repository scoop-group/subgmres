#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke check: every demo still runs.

Not a test, and deliberately not part of the test suite. The suite asserts
that the solver is correct; this asserts only that the demonstrations still
execute after an API change, which is a different question and one whose
failures should not be reported as correctness failures. It checks no numbers
and makes no claim about the output.

Run it from anywhere:  python3 demos/run_all_demos.py
Exits non-zero if any demo fails, so it can serve as a release-checklist step.
"""

import pathlib
import subprocess
import sys

here = pathlib.Path(__file__).resolve().parent
# The demos import subgmres as a package, so they run from src/python.
package_root = here.parent

demos = sorted(here.glob("demo_*.py"))
failures = []
for demo in demos:
	completed = subprocess.run([sys.executable, str(demo)], cwd=package_root,
	                           capture_output=True, text=True)
	if completed.returncode == 0:
		print(f"  ok    {demo.name}")
	else:
		last_line = (completed.stderr.strip().splitlines() or ["(no output)"])[-1]
		print(f"  FAIL  {demo.name}\n          {last_line}")
		failures.append(demo.name)

print(f"\n  {len(demos) - len(failures)} of {len(demos)} demos ran"
      + (f"; failed: {', '.join(failures)}" if failures else ""))
sys.exit(1 if failures else 0)
