#!/usr/bin/env python3
"""Deploy the repository onto a running plugin tree without losing local patches.

One tool for the whole procedure in `docs/DEPLOY.md`, because the parts are useless apart: an audit without a staged
verification is a rumour, and a swap without an audit overwrites code that exists nowhere else.

    # 1. what is on the box that git has never seen?
    python3 scripts/deploy_plugin.py audit   --plugin <dir> --repo <dir>

    # 2. back up the plugin tree and (optionally) a database
    python3 scripts/deploy_plugin.py backup  --plugin <dir> --out <dir> [--db <file>]

    # 3. stage the additive tree
    python3 scripts/deploy_plugin.py stage   --plugin <dir> --repo <dir> --stage <dir> --ref origin/main

    # 4. prove it imports and runs before anything moves
    python3 scripts/deploy_plugin.py verify  --stage <dir> [--db <file>]

    # 5. swap, re-verify at the live path, roll back automatically on failure
    python3 scripts/deploy_plugin.py swap    --plugin <dir> --stage <dir> --db <file>

Every path is an argument: nothing here assumes a machine layout, and no default is baked in.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

BOX_ONLY_HINT = "a module only the box has is kept, never replaced"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()


def _known_blobs(repo: Path) -> set[str]:
    out = subprocess.run(["git", "rev-list", "--objects", "--all"], cwd=repo,
                         capture_output=True, text=True, check=False).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def _hash(repo: Path, path: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=repo,
                          capture_output=True, text=True, check=False).stdout.strip()


def cmd_audit(args: argparse.Namespace) -> int:
    plugin, repo = Path(args.plugin), Path(args.repo)
    known = _known_blobs(repo)
    on_ref = set(_git(repo, "ls-tree", "-r", "--name-only", args.ref).split())
    local_patches, box_only, old = [], [], []
    for path in sorted(plugin.glob("*.py")):
        if path.name not in on_ref:
            box_only.append(path.name)
        elif _hash(repo, path) in known:
            old.append(path.name)
        else:
            local_patches.append(path.name)
    print(json.dumps({
        "deployed_py": len(list(plugin.glob("*.py"))),
        "local_patches_not_in_git": local_patches,
        "merely_old": old,
        "box_only": box_only,
        "note": BOX_ONLY_HINT,
    }, indent=1))
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    plugin, out = Path(args.plugin), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "plugin"
    shutil.copytree(plugin, target, symlinks=True)
    manifest = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(target.rglob("*")) if path.is_file()}
    (out / "plugin-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    report = {"plugin_copy": str(target), "files": len(manifest)}
    if args.db:
        # WAL mode: a plain file copy would miss everything still in the -wal.
        destination = out / "cortex.db"
        source = sqlite3.connect(f"file:{Path(args.db)}?mode=ro", uri=True)
        target_db = sqlite3.connect(destination)
        with target_db:
            source.backup(target_db)
        source.close()
        target_db.close()
        check = sqlite3.connect(f"file:{destination}?mode=ro", uri=True)
        report["db"] = str(destination)
        report["db_bytes"] = destination.stat().st_size
        report["db_readable"] = check.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        check.close()
    print(json.dumps(report, indent=1))
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    plugin, repo, stage = Path(args.plugin), Path(args.repo), Path(args.stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "--format=tar", args.ref],
                             cwd=repo, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(stage)], input=archive.stdout, check=True)

    known = _known_blobs(repo)
    kept, took_repo, added = [], [], []
    for path in sorted(stage.glob("*.py")):
        live_file = plugin / path.name
        if not live_file.exists():
            added.append(path.name)
        elif _hash(repo, live_file) in known:
            took_repo.append(path.name)
        else:
            shutil.copy2(live_file, path)
            kept.append(path.name)

    live_names = {item.name for item in plugin.iterdir()}
    for name in sorted(live_names):
        source = plugin / name
        if name == "__pycache__" or (stage / name).exists():
            continue
        if source.is_dir():
            shutil.copytree(source, stage / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, stage / name)

    print(json.dumps({"stage": str(stage), "py_total": len(list(stage.glob("*.py"))),
                      "added": len(added), "took_repo_version": len(took_repo),
                      "kept_local_patches": kept}, indent=1))
    return 0


def _import_as_package(tree: Path) -> dict:
    parent = Path(tempfile.mkdtemp())
    os.symlink(tree, parent / "cortex")
    sys.path.insert(0, str(parent))
    report: dict = {"package": None, "modules": {}, "counts": {}}
    try:
        importlib.import_module("cortex")
        report["package"] = "ok"
    except Exception as error:
        report["package"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()[-800:]
        return report
    ok = failed = 0
    for path in sorted(tree.glob("*.py")):
        if path.stem.startswith("__"):
            continue
        try:
            importlib.import_module(f"cortex.{path.stem}")
            ok += 1
        except Exception as error:
            failed += 1
            report["modules"][path.stem] = f"{type(error).__name__}: {error}"
    report["counts"] = {"ok": ok, "failed": failed}
    return report


SMOKE_CALLS = (
    ("summary_regen", "plan", lambda m, s: len(m.plan(s, limit=5).items)),
    ("interference_pruning", "plan", lambda m, s: len(m.plan(s, limit=5).items)),
    ("cross_domain", "plan", lambda m, s: m.plan(s, limit=5).projects),
    ("procedure_validation", "known_procedures", lambda m, s: len(m.known_procedures(s))),
    ("failure_modes", "lookup", lambda m, s: len(m.lookup(s).modes)),
)


def cmd_verify(args: argparse.Namespace) -> int:
    tree = Path(args.stage)
    imported = _import_as_package(tree)
    report: dict[str, object] = {"import": imported, "smoke": {}}
    failures = list(imported.get("modules", {}).keys())

    smoke: dict[str, str] = {}
    if args.db:
        copy = Path(tempfile.mkdtemp()) / "smoke.db"
        source = sqlite3.connect(f"file:{Path(args.db)}?mode=ro", uri=True)
        target = sqlite3.connect(copy)
        with target:
            source.backup(target)
        source.close()
        target.close()
        store_module = importlib.import_module("cortex.store")
        store = store_module.CortexStore(copy)
        for module_name, function, call in SMOKE_CALLS:
            key = f"{module_name}.{function}"
            try:
                module = importlib.import_module(f"cortex.{module_name}")
                smoke[key] = f"ok -> {call(module, store)}"
            except Exception as error:
                smoke[key] = f"{type(error).__name__}: {error}"
                failures.append(key)
        store.close()

    report["smoke"] = smoke
    report["failures"] = failures
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


def cmd_swap(args: argparse.Namespace) -> int:
    plugin, stage = Path(args.plugin), Path(args.stage)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    previous = plugin.parent / f"{plugin.name}.previous-{stamp}"
    if not stage.exists():
        print(json.dumps({"ok": False, "reason": "nothing staged"}))
        return 1
    os.rename(plugin, previous)
    os.rename(stage, plugin)

    verify = subprocess.run([sys.executable, __file__, "verify", "--stage", str(plugin)]
                            + (["--db", args.db] if args.db else []),
                            capture_output=True, text=True, timeout=1800)
    print(json.dumps({"swapped": True, "previous": str(previous), "verified": verify.returncode == 0,
                      "verify_tail": (verify.stdout or verify.stderr)[-800:]}, indent=1))
    if verify.returncode != 0:
        rejected = plugin.parent / f"{plugin.name}.rejected-{stamp}"
        os.rename(plugin, rejected)
        os.rename(previous, plugin)
        print(json.dumps({"rolled_back": True, "rejected": str(rejected), "live": str(plugin)}, indent=1))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--plugin", required=True)
    audit.add_argument("--repo", required=True)
    audit.add_argument("--ref", default="origin/main")
    audit.set_defaults(func=cmd_audit)

    backup = sub.add_parser("backup")
    backup.add_argument("--plugin", required=True)
    backup.add_argument("--out", required=True)
    backup.add_argument("--db")
    backup.set_defaults(func=cmd_backup)

    stage = sub.add_parser("stage")
    stage.add_argument("--plugin", required=True)
    stage.add_argument("--repo", required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--ref", default="origin/main")
    stage.set_defaults(func=cmd_stage)

    verify = sub.add_parser("verify")
    verify.add_argument("--stage", required=True)
    verify.add_argument("--db")
    verify.set_defaults(func=cmd_verify)

    swap = sub.add_parser("swap")
    swap.add_argument("--plugin", required=True)
    swap.add_argument("--stage", required=True)
    swap.add_argument("--db")
    swap.set_defaults(func=cmd_swap)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
