#!/usr/bin/env python
"""Runs a multirate accuracy experiment for the 1D wave equation."""

import os
import subprocess


if __name__ == "__main__":
    subprocess.check_output(
            ["./wave-equation.py", "-x", "accuracy"],
            cwd=os.path.dirname(__file__))
