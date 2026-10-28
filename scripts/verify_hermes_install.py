#!/usr/bin/env python3
"""Verify a Cortex install inside Hermes, using Hermes's own plugin loader.

Run with Hermes's Python (the installer does this):

    HERMES_HOME=~/.hermes <hermes-python> scripts/verify_hermes_install.py [--write-test]

Checks that Hermes discovers and loads the ``cortex`` memory provider, that
config.yaml selects it, that background preload and semantic fusion are live,
and that recall answers within Hermes's prefetch budget. ``--write-test`` also
records one explicit test turn and recalls it (it is then archived again).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    report: dict = {"hermes_home": str(home), "python": sys.executable, "checks": {}}
    checks = report["checks"]

    def check(name: str, ok: bool, detail=None) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    try:
        import yaml

        config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    except ImportError:  # current Hermes ships ruamel.yaml only
        from ruamel.yaml import YAML

        config = YAML(typ="safe", pure=True).load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    provider_name = ((config.get("memory") or {}).get("provider"))
    check("config_selects_cortex", provider_name == "cortex", provider_name)
    plugin_cfg = ((config.get("plugins") or {}).get("cortex") or {})
    check("background_preload_enabled", bool(plugin_cfg.get("background_preload")), plugin_cfg.get("background_preload"))
    check("plugin_files_present", (home / "plugins" / "cortex" / "plugin.yaml").is_file())

    provider = None
    try:
        from plugins.memory import load_memory_provider  # Hermes's real loader

        provider = load_memory_provider("cortex")
        check("hermes_loader_finds_cortex", provider is not None, type(provider).__name__ if provider else None)
    except Exception as exc:  # noqa: BLE001
        check("hermes_loader_finds_cortex", False, f"{type(exc).__name__}: {exc}")
    if provider is None:
        sys.path.insert(0, str(home / "plugins"))
        from cortex.hermes_provider import CortexMemoryProvider  # type: ignore

        provider = CortexMemoryProvider()

    provider.initialize("cortex-install-verify", hermes_home=str(home))
    retriever = provider._retriever  # noqa: SLF001 - verification only
    from_embedder = None
    try:
        import importlib

        emb = importlib.import_module(type(retriever).__module__.rsplit(".", 1)[0] + ".embeddings")
        embedder = emb.get_embedder()
        from_embedder = {"model_dir": str(embedder.model_dir), "files_present": embedder.available}
        started = time.perf_counter()
        vector = embedder.embed_one("verify")
        from_embedder["vector_dim"] = len(vector)
        from_embedder["first_embed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:  # noqa: BLE001
        from_embedder = {"error": f"{type(exc).__name__}: {exc}"}
    check("semantic_fusion_active",
          float(retriever.semantic_weight) > 0 and bool(from_embedder.get("vector_dim")),
          {"semantic_weight": retriever.semantic_weight, **from_embedder})
    check("preloader_running", provider._preloader is not None, provider.preload_stats())  # noqa: SLF001

    started = time.perf_counter()
    provider.queue_prefetch("what do I know about this project", session_id="cortex-install-verify")
    queued_ms = (time.perf_counter() - started) * 1000
    check("queue_prefetch_non_blocking", queued_ms < 250, round(queued_ms, 2))
    time.sleep(0.5)
    started = time.perf_counter()
    provider.prefetch("what do I know about this project", session_id="cortex-install-verify")
    prefetch_ms = (time.perf_counter() - started) * 1000
    check("prefetch_under_hermes_budget", prefetch_ms < 8000, round(prefetch_ms, 1))

    if args.write_test:
        marker = "cortex-verify-7f3a"
        provider.sync_turn(
            f"Remember: we decided to use PostgreSQL for the {marker} project because it supports JSONB.",
            "Noted, I will remember that.",
            session_id="cortex-install-verify",
        )
        # By design Hermes-captured memories are staged for review (the local
        # dashboard, or the opt-in auto-judge). Approve the test proposal the
        # way the dashboard would, then recall it in a fresh session.
        store = provider._store  # noqa: SLF001
        staged = store._conn.execute(  # noqa: SLF001
            "SELECT proposal_id FROM memory_creation_proposals WHERE content LIKE ? AND status='pending'",
            (f"%{marker}%",)).fetchall()
        check("capture_staged_for_review", bool(staged), len(staged))
        for row in staged:
            store.review_memory_creation(row["proposal_id"], "remember", actor="install-verify",
                                         reason_text="install verification")
        recalled = provider.prefetch(f"which database did we decide to use for the {marker} project?",
                                     session_id="cortex-install-verify-2")
        found = marker in (recalled or "")
        check("explicit_memory_round_trip", found, (recalled or "")[:300])
        # Clean up the verification memory.
        try:
            rows = provider._store._conn.execute(  # noqa: SLF001
                "SELECT id FROM memories WHERE content LIKE ?", (f"%{marker}%",)).fetchall()
            for row in rows:
                provider._store.set_state(row["id"], "archived", reason="install verification cleanup")  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass

    provider.shutdown()
    report["ok"] = all(item["ok"] for item in checks.values())
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for name, item in checks.items():
            print(f"{'ok ' if item['ok'] else 'FAIL'} {name}: {json.dumps(item['detail'], default=str)[:200]}")
        print("PASS: Cortex is live in Hermes" if report["ok"] else "FAIL: see checks above")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
