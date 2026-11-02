#!/usr/bin/env python
"""Clip the physically impossible CCS temperature cells to the validated range, with a per-cell audit log.

Thin entry point over datasets/_preprocess_lib.py (the stage functions are
unchanged from the production pipeline). Extra arguments pass through.
"""
import sys

import _preprocess_lib

sys.argv = [sys.argv[0], "--region", "ccs", "--stage", "clip-temp"] + sys.argv[1:]

if __name__ == "__main__":
    sys.exit(_preprocess_lib.main())
