#!/usr/bin/env python
"""Runs a multirate accuracy experiment for the 1D wave equation

(This file is here to run the code as a CI job. See the file wave-equation.py
for more experiments.)
"""

import subprocess


if __name__ == "__main__":
    subprocess.check_call(["./wave-equation.py", "-x", "accuracy"])
