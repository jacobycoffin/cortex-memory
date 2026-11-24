"""Is the running plugin the code in the repository? (#221)

The deployed plugin is a **snapshot**, not a checkout: files copied in, no `.git`, mtimes from the day
of the copy. So "the repo says X" and "the service does X" can be different sentences for weeks, and
every live measurement taken in between describes the old build while looking like current evidence.

That is not hypothetical. Measured 2026-11-24: the deployed copy had **66** Python files, all stamped
2026-10-20, while `main` had **167** top-level modules — 122 absent, including `boost_caps.py` and
`usage_weight.py`. Which is why 10,058 traced candidate rows belonged to memories with a live `used`
count and not one carried `use_boost`: the boosts do not exist in the running code.

This module answers that question and nothing else. It is **read-only** — it never copies, deploys or
restarts; a redeploy is a decision for the owner of the machine, not a side effect of a check.

Four verdicts, deliberately distinguishable:

- `current` — every module the repo ships is present and identical;
- `behind` — modules are missing or differ, and the differences are listed;
- `ahead` — the deployed copy has modules the repo does not (someone patched the live tree);
- `unreadable` — the deployed path is missing or not readable, which is a finding, not an error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CURRENT = "current"
BEHIND = "behind"
AHEAD = "ahead"
UNREADABLE = "unreadable"


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001 - an unreadable file is reported, never raised
        return ""


def _modules(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    try:
        for entry in sorted(root.iterdir()):
            if entry.is_dir() or entry.suffix != ".py":
                continue
            found[entry.name] = entry
    except Exception:  # noqa: BLE001
        return {}
    return found


@dataclass
class Drift:
    verdict: str = UNREADABLE
    deployed: str = ""
    repo: str = ""
    deployed_count: int = 0
    repo_count: int = 0
    missing: list[str] = field(default_factory=list)
    differing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "deployed": self.deployed,
            "repo": self.repo,
            "deployed_count": self.deployed_count,
            "repo_count": self.repo_count,
            "missing": self.missing,
            "differing": self.differing,
            "extra": self.extra,
            "reason": self.reason,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if self.verdict == UNREADABLE:
            return f"unreadable: {self.reason}"
        if self.verdict == CURRENT:
            return (f"current: {self.deployed_count} module(s) identical to {self.repo}")
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.differing:
            parts.append(f"{len(self.differing)} differing")
        if self.extra:
            parts.append(f"{len(self.extra)} the repo does not have")
        return (f"{self.verdict}: {', '.join(parts)} — deployed {self.deployed_count}, "
                f"repo {self.repo_count}")


def compare(repo_root: str | Path, deployed: str | Path, *,
            limit: int = 400) -> Drift:
    """Compare a repo tree against the deployed copy. Reads both; writes neither."""
    repo_path = Path(repo_root)
    deployed_path = Path(deployed)
    found = Drift(deployed=str(deployed_path), repo=str(repo_path))
    if not deployed_path.is_dir():
        found.reason = f"{deployed_path} is not a directory"
        return found
    repo_modules = _modules(repo_path)
    if not repo_modules:
        found.reason = f"{repo_path} has no Python modules"
        return found
    deployed_modules = _modules(deployed_path)
    found.repo_count = len(repo_modules)
    found.deployed_count = len(deployed_modules)
    if not deployed_modules:
        # An empty deployed tree means *everything* is missing; saying "no modules" and stopping
        # would hide that behind a one-line reason.
        found.verdict = BEHIND
        found.reason = "the deployed copy holds no Python modules, so all of them are missing"
        found.missing = sorted(repo_modules)[: int(limit)]
        return found
    for name, path in repo_modules.items():
        target = deployed_modules.get(name)
        if target is None:
            found.missing.append(name)
            continue
        if _digest(path) != _digest(target):
            found.differing.append(name)
    found.extra = sorted(set(deployed_modules) - set(repo_modules))
    found.missing.sort()
    found.differing.sort()
    if found.extra and not (found.missing or found.differing):
        found.verdict = AHEAD
    elif found.missing or found.differing:
        found.verdict = BEHIND
    else:
        found.verdict = CURRENT
    if found.verdict != CURRENT:
        found.reason = (f"{len(found.missing)} module(s) the deployed copy does not have, "
                       f"{len(found.differing)} it has a different version of")
    found.missing = found.missing[: int(limit)]
    found.differing = found.differing[: int(limit)]
    found.extra = found.extra[: int(limit)]
    return found


def samples(drift: Drift, count: int = 5) -> dict[str, list[str]]:
    """The first few names in each bucket, for a report that a human can actually read."""
    return {
        "missing": drift.missing[:count],
        "differing": drift.differing[:count],
        "extra": drift.extra[:count],
    }
