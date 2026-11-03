#!/usr/bin/env python
"""Repository hygiene gate: no tool attributions, no site or personal
identifiers anywhere in tracked content.

Scans every git-tracked file (and, with --log, the full commit history) for a
banned-pattern list covering tool names, attribution trailers, personal e-mail
addresses, cluster filesystem prefixes, allocation and account names, and
hostnames. The patterns are stored base64-encoded so this file does not match
itself. Exit status 0 = clean, 1 = findings.

Usage:
  python tools/check_identifiers.py            # scan tracked files
  python tools/check_identifiers.py --log      # also scan `git log -p --all`
Installed as .git/hooks/pre-commit (scans the tree before every commit).
"""
from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# base64 of the case-insensitive regexes (decoded at runtime; see module docstring)
_ENCODED = [
    "Y2xhdWRl", "YW50aHJvcGlj", "YWkgYXNzaXN0YW50", "YWktYXNzaXN0ZWQ=",
    "XGJncHRcYg==", "Y2hhdGdwdA==", "b3BlbmFp", "XGJsbG1cYg==", "Y29waWxvdA==",
    "bGFuZ3VhZ2UgbW9kZWw=", "Z2VuZXJhdGVkIHdpdGg=", "Y28tYXV0aG9yZWQtYnk=",
    "Y293b3Jr",
    "Z21haWw=", "L2hvbWUvc2hhZmV6aQ==", "L25mcy90dXJibw==", "L3NjcmF0Y2gv",
    "YXBhcGFsaWE=", "bWFhbmlnag==", "ZW5naW4x", "ZWNlNTY3", "cm9iNTMw",
    "YXJjLXRz", "Z3JlYXRsYWtlcw==", "c2hheWVzdGVoNzgxMw==",
]
PATTERNS = [re.compile(base64.b64decode(e).decode(), re.IGNORECASE)
            for e in _ENCODED]
# Binary payloads (images, archives, tensors) are scanned with the long,
# distinctive patterns only: the two three-letter word-boundary tokens match
# random byte runs inside compressed streams and carry no signal there. Text
# files keep the full list.
BINARY_PATTERNS = [p for p in PATTERNS if "\\b" not in p.pattern]
BINARY_SUFFIXES = {".png", ".gif", ".jpg", ".pdf", ".npz", ".gz", ".zst",
                   ".safetensors", ".mp4"}

SELF = Path(__file__).resolve()


def _tracked_files():
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT).decode()
    return [ROOT / line for line in out.splitlines() if line]


def scan_text(text: str, origin: str, findings: list, patterns=None):
    for i, line in enumerate(text.splitlines(), 1):
        for pat in (patterns if patterns is not None else PATTERNS):
            if pat.search(line):
                findings.append(f"{origin}:{i}: {pat.pattern!r}: {line.strip()[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true",
                    help="also scan the full commit history (git log -p --all)")
    args = ap.parse_args()

    findings: list = []
    for path in _tracked_files():
        if path.resolve() == SELF:
            continue                        # patterns live here, encoded
        try:
            text = path.read_text(errors="replace")
        except (IsADirectoryError, FileNotFoundError):
            continue
        pats = BINARY_PATTERNS if path.suffix in BINARY_SUFFIXES else None
        scan_text(text, str(path.relative_to(ROOT)), findings, pats)
        scan_text(path.name, f"{path.relative_to(ROOT)} (filename)", findings)

    if args.log:
        log = subprocess.check_output(
            ["git", "log", "-p", "--all", "--format=%an %ae %s %b"], cwd=ROOT,
        ).decode(errors="replace")
        scan_text(log, "git-log", findings)

    if findings:
        print(f"IDENTIFIER GATE: {len(findings)} finding(s)")
        for f in findings[:50]:
            print(" ", f)
        return 1
    print("IDENTIFIER GATE: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
