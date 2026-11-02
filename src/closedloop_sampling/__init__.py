"""closedloop_sampling — closed-loop adaptive sampling on a generative ocean model.

Lightweight top-level import: torch-heavy modules (`.forecast`, `.assimilation`,
`.closed_loop`) are imported on use, so `import closedloop_sampling` works
without torch installed. The sampling and observation modules are numpy-only.
"""
__version__ = "0.1.0"
