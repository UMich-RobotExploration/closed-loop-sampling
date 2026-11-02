"""Fetch the upstream LaDCast code (step 1 of setup).

Clones github.com/tonyzyl/ladcast at the pinned commit into
third_party/ladcast_upstream/ (gitignored), verifies the SHA, and prints the
pin plus the license note. Nothing upstream is committed to this repo — one
uniform fetch-at-setup policy across this repository.

LICENSE NOTE (upstream's LICENSE file): LaDCast is dual-licensed — code
modified from third-party sources (primarily /ladcast/models/, i.e. exactly
the modules this package imports) is Apache-2.0; all other original content
is MIT. See the headers in its individual files for attribution lineage.

After fetching, make it importable with ONE of:
    pip install --no-deps third_party/ladcast_upstream
    export LADCAST_UPSTREAM=$PWD/third_party/ladcast_upstream

Run: python tools/fetch_upstream.py
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLONE_DIR = ROOT / "third_party" / "ladcast_upstream"
LOCK = yaml.safe_load(open(ROOT / "configs" / "upstream.lock"))
UPSTREAM_URL = LOCK["url"]
COMMIT = LOCK["commit"]


def sh(*cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT).decode()


def verify_lock():
    """Verify the clone against configs/upstream.lock (commit + file hashes).

    The pin is an untagged upstream master commit; the file hashes guard
    against a history rewrite of the pin."""
    head = sh("git", "-C", str(CLONE_DIR), "rev-parse", "HEAD").strip()
    if head != COMMIT:
        raise SystemExit(f"upstream HEAD {head} != locked {COMMIT}")
    for rel, expect in LOCK["sha256"].items():
        got = hashlib.sha256((CLONE_DIR / rel).read_bytes()).hexdigest()
        if got != expect:
            raise SystemExit(
                f"upstream {rel} sha256 {got} != locked {expect} — the pinned "
                f"commit's content changed upstream; do not proceed."
            )
    print(f"[lock] commit + {len(LOCK['sha256'])} file hash(es) verified "
          f"against configs/upstream.lock")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if CLONE_DIR.exists():
        try:
            head = sh("git", "-C", str(CLONE_DIR), "rev-parse", "HEAD").strip()
            if head == COMMIT:
                print(f"[clone] already present at {COMMIT[:7]} — skipping")
                verify_lock()
                _finish()
                return 0
        except subprocess.CalledProcessError:
            pass
        raise SystemExit(f"{CLONE_DIR} exists but is not at {COMMIT[:7]}; remove it and rerun")
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] {UPSTREAM_URL} -> {CLONE_DIR}")
    sh("git", "clone", UPSTREAM_URL, str(CLONE_DIR))
    sh("git", "-C", str(CLONE_DIR), "checkout", "--detach", COMMIT)
    head = sh("git", "-C", str(CLONE_DIR), "rev-parse", "HEAD").strip()
    if head != COMMIT:
        raise SystemExit(f"post-checkout HEAD {head} != pinned {COMMIT}")
    verify_lock()
    _finish()
    return 0


def _finish():
    print(f"[pin] {UPSTREAM_URL} @ {COMMIT}")
    lic = CLONE_DIR / "LICENSE"
    if lic.exists():
        print("[license] upstream LICENSE (dual Apache-2.0 / MIT):")
        for line in lic.read_text().splitlines():
            if line.strip():
                print(f"    {line}")
    print("[next] make it importable with ONE of:")
    print("    pip install --no-deps third_party/ladcast_upstream")
    print("    export LADCAST_UPSTREAM=$PWD/third_party/ladcast_upstream")


if __name__ == "__main__":
    sys.exit(main())
