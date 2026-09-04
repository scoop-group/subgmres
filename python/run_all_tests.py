#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the Python test suite.

An entry point whose name says what it does. The tests themselves live in
subgmres/tests/ and are ordinary unittest cases, so
`python3 -m unittest discover -s subgmres/tests -t .` does the same thing;
this exists so that nobody has to know that.

Exits non-zero if any test fails.
"""

import pathlib
import subprocess
import sys

here = pathlib.Path(__file__).resolve().parent
completed = subprocess.run(
	[sys.executable, "-m", "unittest", "discover", "-s", "subgmres/tests", "-t", "."],
	cwd=here)
sys.exit(completed.returncode)
