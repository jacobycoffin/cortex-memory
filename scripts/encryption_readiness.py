#!/usr/bin/env python3
"""Encryption at rest: work out what this machine can actually do, and refuse to pretend.

Backlog item 194 is not a feature you write once; it is a property of the deployment.
This script reports the three things that decide whether the capability exists at all:

  1. does the SQLite build in use support an encryption extension (SQLCipher)?
  2. is a key material file present, and is it private enough to trust?
  3. is the current database plaintext (readable magic bytes) or already encrypted?

It never writes to the database, never prints key material, and exits non-zero when the
answer is "not ready", so a deploy script cannot mistake silence for success.

    python3 scripts/encryption_readiness.py --db ~/.hermes/cortex/cortex.db
    python3 scripts/encryption_readiness.py --db ... --key-file /etc/cortex/store.key --json

Exit codes: 0 ready to encrypt, 1 not ready, 2 could not inspect.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path

SQLITE_MAGIC = b"SQLite format 3\x00"


def cipher_support() -> tuple[bool, str]:
    """True when this SQLite build can encrypt a database file."""
    conn = sqlite3.connect(":memory:")
    try:
        row = conn.execute("PRAGMA cipher_version").fetchone()
    except sqlite3.Error as exc:
        return False, f"PRAGMA cipher_version failed: {exc}"
    finally:
        conn.close()
    if row and row[0]:
        return True, str(row[0])
    return False, (f"sqlite {sqlite3.sqlite_version} has no encryption extension "
                   "(PRAGMA cipher_version returned NULL)")


def database_state(db: Path) -> tuple[str, str]:
    """'plaintext', 'encrypted', or 'missing' — decided by the file header, not by hope."""
    if not db.exists():
        return "missing", f"no such file: {db}"
    with db.open("rb") as fh:
        head = fh.read(len(SQLITE_MAGIC))
    if head == SQLITE_MAGIC:
        return "plaintext", "file begins with the SQLite header: readable without a key"
    if head and head.startswith(b"\x00"):
        return "encrypted", "file header is not SQLite's: likely encrypted"
    return "unknown", f"unrecognised header: {head!r}"


def key_file_state(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"no key file at {path}"
    mode = stat.S_IMODE(path.stat().st_mode)
    size = path.stat().st_size
    if mode & 0o077:
        return False, f"key file mode is {mode:o}; it must be 0600 (no group/other bits)"
    if size < 32:
        return False, f"key file is {size} bytes; 32+ bytes of entropy required"
    return True, f"key file ok ({size} bytes, mode {mode:o})"


def report(db: Path, key_file: Path) -> dict:
    supported, cipher_note = cipher_support()
    state, state_note = database_state(db)
    key_ok, key_note = key_file_state(key_file)
    ready = supported and key_ok and state in ("plaintext", "encrypted")
    return {
        "cipher_supported": supported,
        "cipher_note": cipher_note,
        "database_state": state,
        "database_note": state_note,
        "key_ok": key_ok,
        "key_note": key_note,
        "ready": ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--key-file", type=Path,
                        default=Path(os.environ.get("CORTEX_KEY_FILE", "/etc/cortex/store.key")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = report(args.db, args.key_file)
    except OSError as exc:
        print(f"encryption_readiness: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"database      {args.db}: {result['database_state']} — {result['database_note']}")
        print(f"cipher        {'yes' if result['cipher_supported'] else 'NO'} — {result['cipher_note']}")
        print(f"key file      {'ok' if result['key_ok'] else 'NOT OK'} — {result['key_note']}")
        print(f"verdict       {'READY to encrypt' if result['ready'] else 'NOT READY'}")

    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
