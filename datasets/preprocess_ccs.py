#!/usr/bin/env python
"""Transform the CCS staged files onto the canonical de-overlapped timeline (destagger, level-drop, crop).

Thin entry point over datasets/_preprocess_lib.py (the stage functions are
unchanged from the production pipeline). Extra arguments pass through.
"""
import sys

import _preprocess_lib

sys.argv = [sys.argv[0], "--region", "ccs", "--stage", "transform"] + sys.argv[1:]

if __name__ == "__main__":
    sys.exit(_preprocess_lib.main())
