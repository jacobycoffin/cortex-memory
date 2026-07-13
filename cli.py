"""Small local inspection CLI for a Cortex database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .retrieval import MemoryRetriever
from .security import sanitize_memory
from .store import CortexStore


def main() -> int:
    parser = argparse.ArgumentParser(prog="cortex", description="Inspect and test Cortex adaptive memory")
    parser.add_argument(
        "--db",
        default=os.environ.get("CORTEX_DB", str(Path.home() / ".hermes" / "cortex" / "cortex.db")),
        help="Path to cortex.db",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    sub.add_parser("audit")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=6)
    search.add_argument("--include-archived", action="store_true")
    search.add_argument("--historical", action="store_true")
    search.add_argument("--as-of", help="ISO timestamp used by historical retrieval")
    remember = sub.add_parser("remember")
    remember.add_argument("content")
    remember.add_argument("--kind", default="semantic")
    remember.add_argument("--pin", action="store_true")
    explain = sub.add_parser("explain")
    explain.add_argument("memory_id")
    maintenance = sub.add_parser("maintenance")
    maintenance.add_argument("--apply", action="store_true")
    consolidate = sub.add_parser("consolidate", help="Preview or apply reversible near-duplicate folding")
    consolidate.add_argument("--apply", action="store_true")
    consolidate.add_argument("--threshold", type=float, default=0.78)
    undo = sub.add_parser("undo-consolidation", help="Restore members from an applied consolidation run")
    undo.add_argument("run_id")
    sub.add_parser("recall-stats", help="Show attention-gate latency and context-budget evidence")
    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--no-open", action="store_true")
    vault_index = sub.add_parser("vault-index", help="Plan or apply an incremental Obsidian vault import")
    vault_index.add_argument("vault_path")
    vault_index.add_argument("--apply", action="store_true", help="Write the planned import to Cortex")
    vault_index.add_argument("--max-chars", type=int, default=1600, help="Maximum source characters per chunk")
    vault_index.add_argument("--max-file-bytes", type=int, default=1_000_000)
    sub.add_parser("vault-status", help="Show indexed vault sources and chunks")
    args = parser.parse_args()

    if args.command == "dashboard":
        from .dashboard import serve_dashboard

        serve_dashboard(args.db, port=args.port, open_browser=not args.no_open)
        return 0

    store = CortexStore(args.db)
    try:
        if args.command == "stats":
            result = store.stats()
        elif args.command == "audit":
            result = store.audit()
        elif args.command == "search":
            result = [
                r.as_dict()
                for r in MemoryRetriever(store).search(
                    args.query,
                    limit=args.limit,
                    include_archived=args.include_archived,
                    temporal_mode="historical" if args.historical or args.as_of else "current",
                    as_of=args.as_of,
                )
            ]
        elif args.command == "remember":
            sanitized = sanitize_memory(args.content)
            memory_id, created = store.add_memory(
                sanitized.text,
                kind=args.kind,
                pinned=args.pin,
                source_type="cli",
                quarantine_reason=sanitized.quarantine_reason,
            )
            result = {"memory_id": memory_id, "created": created, "quarantined": bool(sanitized.quarantine_reason)}
        elif args.command == "explain":
            memory_id = store.resolve_id(args.memory_id)
            result = store.explain(memory_id) if memory_id else None
        elif args.command == "vault-index":
            from .vault import VaultIndexer

            indexer = VaultIndexer(
                store,
                args.vault_path,
                max_chars=args.max_chars,
                max_file_bytes=args.max_file_bytes,
            )
            if args.apply:
                result = indexer.apply()
            else:
                _scan, result = indexer.plan()
                result = {**result, "applied": False}
        elif args.command == "vault-status":
            sources = list(store.document_manifest().values())
            result = {
                "sources": sources,
                "active_sources": sum(source["status"] == "active" for source in sources),
                "missing_sources": sum(source["status"] == "missing" for source in sources),
                "active_chunks": sum(
                    len(store.document_chunks(source["source_path"], active_only=True)) for source in sources
                ),
            }
        elif args.command == "consolidate":
            result = store.consolidate(dry_run=not args.apply, similarity_threshold=args.threshold)
        elif args.command == "undo-consolidation":
            result = store.undo_consolidation(args.run_id)
        elif args.command == "recall-stats":
            snapshot = store.dashboard_snapshot(memory_limit=1)
            result = {
                "summary": snapshot["recall_summary"],
                "modes": snapshot["recall_modes"],
                "by_day": snapshot["recall_by_day"],
            }
        else:
            result = store.maintenance(dry_run=not args.apply)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
