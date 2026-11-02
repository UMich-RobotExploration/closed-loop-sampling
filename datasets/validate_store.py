#!/usr/bin/env python
"""Validate a preprocessed store against its region config (shapes, ranges,
wet-mask integrity). Usage: validate_store.py --region {mab,ccs} [...]

Thin entry point over datasets/_preprocess_lib.py (validation functions are
unchanged from the production pipeline). Extra arguments pass through.
"""
import sys

import _preprocess_lib

args = sys.argv[1:]
if "--validate" not in args:
    args = args + ["--validate"]
sys.argv = [sys.argv[0]] + args

if __name__ == "__main__":
    sys.exit(_preprocess_lib.main())
