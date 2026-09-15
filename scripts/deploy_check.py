#!/usr/bin/env python3
"""Pre-deploy safety check: what would change, and is anything live-only at risk?

The live plugin (`~/.hermes/plugins/cortex`) is normally an ANCESTOR of this repo,
but that is an assumption, not a guarantee — historically the two diverged and a
wholesale copy would have deleted live-only work. This script classifies every
file that exists in both trees before anyone copies anything:

  at HEAD        the repo tip already equals the live file (nothing to deploy)
  behind         the live file equals an older repo commit, so the live copy is a
                 strict ancestor: copying the current version is safe
  DIVERGENT      the live file matches no recent repo commit. It may hold work the
                 repo lacks; a human must reconcile it. Exit code 1.

Files that exist only in the live plugin are reported and never touched.

Usage:
    python3 scripts/deploy_check.py [--live DIR] [--lookback N]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LIVE = Path.home() / ".hermes" / "plugins" / "cortex"
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Closure:
    """Outcome of comparing one live file against the repo's history."""

    at_head: list[str] = field(default_factory=list)
    behind: list[tuple[str, str, int]] = field(default_factory=list)
    divergent: list[str] = field(default_factory=list)
    live_only: list[str] = field(default_factory=list)

    @property
    def deployable(self) -> list[str]:
        """Files whose current repo version would be copied to the plugin."""

        return [name for name, _commit, _count in self.behind]

    @property
    def ok(self) -> bool:
        return not self.divergent


def classify(
    live_files: dict[str, bytes],
    head_blobs: dict[str, bytes],
    ancestor_lookup,
) -> Closure:
    """Pure classification core, so it can be unit-tested without a repo.

    ``ancestor_lookup(rel)`` returns ``[(commit, blob_bytes), ...]`` newest first.
    """

    def sha(data: bytes | None) -> str:
        return hashlib.sha256(data or b"").hexdigest()

    result = Closure()
    for rel, live_bytes in sorted(live_files.items()):
        head_blob = head_blobs.get(rel)
        if head_blob is None:
            result.live_only.append(rel)
            continue
        if head_blob == live_bytes:
            result.at_head.append(rel)
            continue
        match = None
        for index, (commit, blob) in enumerate(ancestor_lookup(rel)):
            if blob == live_bytes:
                match = (commit, index)
                break
        if match is None:
            result.divergent.append(rel)
        else:
            commit, index = match
            result.behind.append((rel, commit, index))
    return result


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True
    ).stdout.strip()


def _blob(ref: str, rel: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=str(REPO_ROOT), capture_output=True
    ).stdout


def collect(live_root: Path, lookback: int) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, list]]:
    live_files: dict[str, bytes] = {}
    for path in sorted(live_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(live_root).as_posix()
        live_files[rel] = path.read_bytes()

    head_blobs: dict[str, bytes] = {}
    for rel in live_files:
        if (REPO_ROOT / rel).exists():
            head_blobs[rel] = _blob("HEAD", rel)
    tracked = set(_git("ls-files").splitlines())

    cache: dict[str, list] = {}

    def ancestor_lookup(rel: str) -> list:
        if rel in cache:
            return cache[rel]
        entries = []
        for commit in _git("log", f"-{lookback}", "--format=%H", "--", rel).split():
            entries.append((commit, _blob(commit, rel)))
        cache[rel] = entries
        return entries

    return live_files, {rel: head_blobs[rel] for rel in head_blobs}, ancestor_lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--lookback", type=int, default=120, help="commits to search per file")
    args = parser.parse_args()

    if not args.live.is_dir():
        print(f"live plugin directory not found: {args.live}", file=sys.stderr)
        return 2

    live_files, head_blobs, ancestor_lookup = collect(args.live, args.lookback)
    closure = classify(live_files, head_blobs, ancestor_lookup)

    print(f"live: {args.live}")
    print(f"repo: {REPO_ROOT} @ {_git('rev-parse', '--short', 'HEAD')}")
    print(f"files compared: {len(live_files)}")
    print(f"\n  at HEAD (already current): {len(closure.at_head)}")
    print(f"  behind (safe to copy)    : {len(closure.behind)}")
    for name, commit, count in sorted(closure.behind, key=lambda row: row[2]):
        print(f"      {name:28} live == {commit[:8]}  {count:>3} commits behind")
    print(f"  DIVERGENT (do NOT copy)  : {len(closure.divergent)}")
    for name in closure.divergent:
        print(f"      {name}  <- reconcile by hand")
    print(f"  live-only (left alone)   : {len(closure.live_only)}")

    if not closure.ok:
        print("\nRESULT: blocked — divergent files need manual reconciliation.")
        return 1
    print(f"\nRESULT: safe — {len(closure.deployable)} file(s) would be updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
