#!/usr/bin/env python
"""Preprocess the MAB raw NetCDF staging into the chunked store (destagger, crop, land-zero); then validate with validate_store.py.

Thin entry point over datasets/_preprocess_lib.py (the stage functions are
unchanged from the production pipeline). Extra arguments pass through.
"""
import sys

import _preprocess_lib

sys.argv = [sys.argv[0], "--region", "mab"] + sys.argv[1:]

if __name__ == "__main__":
    sys.exit(_preprocess_lib.main())
