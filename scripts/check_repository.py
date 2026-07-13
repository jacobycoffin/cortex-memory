#!/usr/bin/env python3
"""Portable release checks for public tracked files and XML-based assets."""

from __future__ import annotations

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH_SUFFIXES = (
    ".db",
    ".db-shm",
    ".db-wal",
    ".key",
    ".pem",
    ".private.md",
    ".private.jsonl",
)
PRIVATE_FILENAMES = {".env", "TWITTER_LAUNCH_KIT.md", "dashboard-auth.json"}
CONTENT_RULES = {
    "macOS home-directory path": re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    "root Hermes home path": re.compile(rb"/root/\.hermes(?:/|\b)"),
    "private key material": re.compile(rb"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY"),
    "OpenAI-style secret token": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub secret token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    "Slack secret token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "JWT bearer token": re.compile(rb"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def check_public_files(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.name in PRIVATE_FILENAMES or relative.endswith(PRIVATE_PATH_SUFFIXES):
            failures.append(f"private file is tracked: {relative}")
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for label, pattern in CONTENT_RULES.items():
            if pattern.search(data):
                failures.append(f"{label} found in tracked file: {relative}")
    return failures


def check_xml_assets(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    assets = [path for path in paths if path.suffix.casefold() in {".svg", ".xml"}]
    for path in assets:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            failures.append(f"invalid XML in {path.relative_to(ROOT).as_posix()}: {exc}")
            continue
        if path.suffix.casefold() == ".svg" and not root.tag.casefold().endswith("svg"):
            failures.append(f"SVG root element is not <svg>: {path.relative_to(ROOT).as_posix()}")
    if not assets:
        failures.append("no tracked SVG or XML assets were found")
    return failures


def main() -> int:
    paths = tracked_files()
    failures = check_public_files(paths) + check_xml_assets(paths)
    if failures:
        for failure in failures:
            print(f"release check failed: {failure}", file=sys.stderr)
        return 1
    xml_count = sum(path.suffix.casefold() in {".svg", ".xml"} for path in paths)
    print(f"Repository checks passed: {len(paths)} tracked files, {xml_count} XML assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
