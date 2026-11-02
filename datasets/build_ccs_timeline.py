#!/usr/bin/env python
"""Build the canonical de-overlapped CCS timeline manifest (later-file-wins dedup) before transforming.

Thin entry point over datasets/_preprocess_lib.py (the stage functions are
unchanged from the production pipeline). Extra arguments pass through.
"""
import sys

import _preprocess_lib

sys.argv = [sys.argv[0], "--region", "ccs", "--stage", "build-timeline"] + sys.argv[1:]

if __name__ == "__main__":
    sys.exit(_preprocess_lib.main())
