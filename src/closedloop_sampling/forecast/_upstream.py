"""Make the pinned upstream `ladcast` package importable.

The upstream LaDCast repo is fetched at setup (tools/fetch_upstream.py) and
normally installed with `pip install --no-deps third_party/ladcast_upstream`,
after which `import ladcast` just works. For environments where installing is
not possible, the documented override is the LADCAST_UPSTREAM environment
variable: point it at the fetched clone (the directory CONTAINING the
`ladcast/` package) and :func:`ensure_ladcast` will add it to sys.path.
This env-gated fallback is the only path manipulation in the package.
"""
import os
import sys
from pathlib import Path


def ensure_ladcast():
    """Import-or-explain for the upstream `ladcast` dependency."""
    try:
        import ladcast  # noqa: F401
        return
    except ImportError:
        pass
    root = os.environ.get("LADCAST_UPSTREAM")
    if root and (Path(root) / "ladcast" / "__init__.py").exists():
        if root not in sys.path:
            sys.path.insert(0, root)
        import ladcast  # noqa: F401
        return
    raise ImportError(
        "the upstream `ladcast` package is not importable.\n"
        "Setup: python tools/fetch_upstream.py   (clones the pinned commit)\n"
        "then:  pip install --no-deps third_party/ladcast_upstream\n"
        "or:    export LADCAST_UPSTREAM=/path/to/ladcast_upstream  (the clone dir)"
    )
