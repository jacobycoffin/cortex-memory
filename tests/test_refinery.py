from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


from tests._bootstrap import ROOT  # noqa: F401 - loads the cortex package

from cortex.refinery import (
    PRESENTATION_VERSION,
    ROLE_CLASSIFIER_VERSION,
    build_presentation,
    classify_record_role,
    deterministic_split_preview,
)
from cortex.retrieval import MemoryRetriever, SHADOW_ROLE_POLICY_VERSION
from cortex.store import REFINERY_BACKFILL_KEY, CortexStore
from cortex.vault import VaultIndexer


CODE_CONTENT = (
    "```python\ndef restart_service(name):\n    return subprocess.run(['systemctl', 'restart', name])\n```\n"
    "Run it after every deploy."
)
TABLE_CONTENT = (
    "| Service | Port | Host |\n| --- | --- | --- |\n| grafana | 3000 | mon-01 |\n"
    "| prometheus | 9090 | mon-01 |\n| loki | 3100 | mon-02 |"
)
CONFIG_CONTENT = (
    "server_host = 10.0.0.12\nserver_port = 8443\ntls_enabled = true\nretry_limit = 4\n"
    "queue_backend = redis"
)
DIAGRAM_CONTENT = (
    "infrastructure layout\n├── proxy\n│   ├── caddy\n│   └── acme\n└── apps\n    ├── grafana\n    └── cortex"
)
VAULT_SECTION_CONTENT = (
    "Vault note: Backup Runbook\nPath: manual/Backup Runbook.md\nSection: Nightly restic job\n"
    "The nightly restic job copies the app volumes to the offsite bucket and prunes snapshots "
    "older than ninety days. Verify the log before rotating credentials."
)


class RefineryClassifierTests(unittest.TestCase):
    def test_code_tables_configs_and_diagrams_become_reference(self) -> None:
        for content in (CODE_CONTENT, TABLE_CONTENT, CONFIG_CONTENT, DIAGRAM_CONTENT):
            result = classify_record_role({"content": content, "kind": "semantic"})
            self.assertEqual(result["record_role"], "reference", content[:40])
            self.assertTrue(result["reasons"])

    def test_document_sections_default_to_reference_evidence(self) -> None:
        result = classify_record_role(
            {
                "content": VAULT_SECTION_CONTENT,
                "kind": "procedure",
                "source_type": "vault_markdown",
                "source_category": "DOCUMENT_EXTRACTED",
            }
        )
        self.assertEqual(result["record_role"], "reference")

    def test_explicit_user_statements_stay_canonical(self) -> None:
        result = classify_record_role(
            {
                "content": "Jacoby prefers dashboards rendered in dark mode with dense tables.",
                "kind": "preference",
                "source_category": "USER_EXPLICIT",
            }
        )
        self.assertEqual(result["record_role"], "canonical")
        self.assertEqual(
            [flag for flag in result["readability_flags"] if flag != "uncertain_language"], []
        )

    def test_transient_status_and_fragments_are_flagged_not_discarded(self) -> None:
        status = classify_record_role({"content": "Deploy job completed successfully.", "kind": "semantic"})
        self.assertIn("transient_status", status["readability_flags"])
        fragment = classify_record_role({"content": "This one should use the newer path.", "kind": "semantic"})
        self.assertIn("unresolved_reference", fragment["readability_flags"])

    def test_episodes_become_events_and_unsupported_inferences_become_claims(self) -> None:
        episode = classify_record_role({"content": "We debugged the proxy for an hour.", "kind": "episode"})
        self.assertEqual(episode["record_role"], "event")
        claim = classify_record_role(
            {
                "content": "The operator appears to favor asynchronous updates.",
                "kind": "semantic",
                "source_category": "AGENT_INFERENCE",
            },
            has_active_dependencies=False,
        )
        self.assertEqual(claim["record_role"], "claim")
        self.assertIn("unconfirmed_claim", claim["readability_flags"])
        supported = classify_record_role(
            {
                "content": "The operator favors asynchronous updates for long jobs.",
                "kind": "semantic",
                "source_category": "AGENT_INFERENCE",
                "source_context": "Stated while reviewing the deploy pipeline.",
            },
            has_active_dependencies=True,
        )
        self.assertEqual(supported["record_role"], "canonical")

    def test_presentation_preserves_negation_anchors_and_uncertainty(self) -> None:
        memory = {
            "content": (
                "Do not upgrade PostgreSQL past 16.3 on athena before 2026-09-01; "
                "the replication bridge probably breaks above that version."
            ),
            "kind": "decision",
            "source_category": "USER_EXPLICIT",
        }
        classification = classify_record_role(memory)
        presentation = build_presentation(memory, classification)
        self.assertIn("Do not upgrade", presentation["display_summary"])
        self.assertIn("16.3", presentation["display_summary"])
        self.assertIn("2026-09-01", presentation["display_summary"])
        self.assertIn("probably", presentation["display_summary"])
        self.assertIn("uncertain_language", presentation["readability_flags"])

    def test_reference_presentation_describes_evidence_without_paths(self) -> None:
        memory = {
            "content": VAULT_SECTION_CONTENT,
            "kind": "procedure",
            "source_type": "vault_markdown",
            "source_category": "DOCUMENT_EXTRACTED",
            "object_value": "Nightly restic job",
            "entities": ["Backup Runbook", "Nightly restic job"],
        }
        classification = classify_record_role(memory)
        presentation = build_presentation(memory, classification)
        self.assertEqual(presentation["display_title"], "Backup Runbook › Nightly restic job")
        self.assertNotIn("manual/Backup Runbook.md", presentation["display_title"])
        self.assertNotIn("manual/Backup Runbook.md", presentation["display_summary"])
        self.assertIn("raw evidence", presentation["display_summary"])

    def test_deterministic_split_uses_bullets_and_sentences(self) -> None:
        parts = deterministic_split_preview(
            "- The proxy listens on 8443.\n- The dashboard binds to localhost only.\n- Backups run nightly."
        )
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(part.strip() for part in parts))


class RefineryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "refinery.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_new_writes_are_classified_and_presented(self) -> None:
        preference_id, _ = self.store.add_memory(
            "Jacoby prefers concise commit messages under seventy characters.",
            kind="preference",
            source_category="USER_EXPLICIT",
        )
        code_id, _ = self.store.add_memory(CODE_CONTENT, kind="procedure")
        self.assertEqual(self.store.get_memory(preference_id)["record_role"], "canonical")
        self.assertEqual(self.store.get_memory(code_id)["record_role"], "reference")
        detail = self.store.explain(code_id)
        self.assertIsNotNone(detail["presentation"])
        self.assertIn("Code reference", detail["presentation"]["display_summary"])
        summary = self.store.refinery_summary()
        self.assertEqual(summary["view_counts"]["readable"], 1)
        self.assertEqual(summary["view_counts"]["reference"], 1)
        self.assertTrue(summary["backfill"]["complete"])

    def test_automatic_capture_with_clarity_flags_stays_a_claim(self) -> None:
        memory_id, created = self.store.add_memory(
            "Kaya observed that the operator currently favors reviewing pull requests in the morning hours.",
            kind="semantic",
            source_category="AGENT_INFERENCE",
            source_context="Observed across three morning review sessions.",
            confidence=0.9,
            importance=0.9,
            storage_policy="automatic",
        )
        self.assertTrue(created)
        memory = self.store.get_memory(memory_id)
        self.assertEqual(memory["record_role"], "claim")

    def test_presentations_rebuild_idempotently_and_invalidate_on_change(self) -> None:
        memory_id, _ = self.store.add_memory(
            "The staging cluster runs Kubernetes 1.31 and must not be upgraded before the audit.",
            kind="operational",
            source_category="USER_EXPLICIT",
        )
        first = self.store.explain(memory_id)["presentation"]
        result = self.store.rebuild_presentations()
        self.assertGreaterEqual(result["rebuilt"], 1)
        second = self.store.explain(memory_id)["presentation"]
        self.assertEqual(first["display_summary"], second["display_summary"])
        self.assertEqual(first["source_digest"], second["source_digest"])
        self.store.correct_memory(memory_id, "The staging cluster now runs Kubernetes 1.32 after the audit.")
        third = self.store.explain(memory_id)["presentation"]
        self.assertNotEqual(first["source_digest"], third["source_digest"])
        self.assertIn("1.32", third["display_summary"])
        self.assertEqual(self.store.audit()["stale_memory_presentations"], 0)

    def test_summary_report_and_items_do_not_leak_content_or_paths(self) -> None:
        private_text = "The vault backup passphrase hint lives beside the offsite key."
        self.store.add_memory(
            f"Vault note: Secret Note\nPath: private/Secret Note.md\nSection: Hint\n{private_text}",
            kind="semantic",
            source_type="vault_markdown",
            source_category="DOCUMENT_EXTRACTED",
            record_role="reference",
        )
        summary_json = json.dumps(self.store.refinery_summary())
        report_json = json.dumps(self.store.refinery_classification_report())
        for payload in (summary_json, report_json):
            self.assertNotIn(private_text, payload)
            self.assertNotIn("private/Secret Note.md", payload)
            self.assertNotIn("Secret Note", payload)

    def test_clarity_items_enter_review_inbox_with_flags(self) -> None:
        self.store.add_memory(
            "Deploy job completed successfully.", kind="semantic", source_category="USER_EXPLICIT"
        )
        inbox = self.store.review_inbox_snapshot()
        clarity = [item for item in inbox["items"] if item["category"] == "clarity"]
        self.assertTrue(clarity)
        self.assertIn("transient status", clarity[0]["question"])
        self.assertGreaterEqual(inbox["counts"].get("clarity", 0), 1)

    def test_role_change_action_is_item_only_audited_and_reversible(self) -> None:
        memory_id, _ = self.store.add_memory(CODE_CONTENT, kind="procedure")
        self.assertEqual(self.store.get_memory(memory_id)["record_role"], "reference")
        result = self.store.apply_refinery_action(
            "keep_canonical", memory_id, actor="tester"
        )
        self.assertEqual(result["decision_scope"], "item_only")
        self.assertEqual(result["affected_memory_ids"], [memory_id])
        changed = self.store.get_memory(memory_id)
        self.assertEqual(changed["record_role"], "canonical")
        self.assertEqual(changed["role_method"], "operator_review")
        self.assertIsNotNone(changed["role_reviewed_at"])
        self.assertTrue(self.store.undo_refinery_action(result["review_id"], actor="tester"))
        restored = self.store.get_memory(memory_id)
        self.assertEqual(restored["record_role"], "reference")
        self.assertIsNone(restored["role_reviewed_at"])
        self.assertFalse(self.store.undo_refinery_action(result["review_id"], actor="tester"))

    def test_rewrite_preview_and_action_link_source_dependencies(self) -> None:
        source_id, _ = self.store.add_memory(
            "This one should be handled by the archiver after the run finishes and someone checks it.",
            kind="semantic",
        )
        prior_role = self.store.get_memory(source_id)["record_role"]
        preview = self.store.refinery_preview("rewrite", source_id)
        self.assertTrue(preview["requires_confirmation"])
        self.assertEqual(preview["source_dependencies"], [source_id])
        self.assertEqual(len(preview["proposed_records"]), 1)
        result = self.store.apply_refinery_action(
            "rewrite",
            source_id,
            proposed_records=[
                {"content": "The nightly archiver handles completed export runs after an operator check."}
            ],
            reason_code="clearer_wording",
            actor="tester",
        )
        self.assertEqual(len(result["created_memory_ids"]), 1)
        created = self.store.get_memory(result["created_memory_ids"][0])
        self.assertEqual(created["record_role"], "canonical")
        self.assertEqual(created["source_category"], "USER_EXPLICIT")
        dependencies = self.store.dependencies(created["id"])
        self.assertEqual([row["evidence_id"] for row in dependencies], [source_id])
        source = self.store.get_memory(source_id)
        self.assertEqual(source["record_role"], "reference")
        self.assertEqual(source["state"], "active")
        self.assertTrue(self.store.undo_refinery_action(result["review_id"], actor="tester"))
        self.assertEqual(self.store.get_memory(created["id"])["state"], "tombstoned")
        self.assertEqual(self.store.get_memory(source_id)["record_role"], prior_role)

    def test_rewrite_preserves_source_scope_and_applicability(self) -> None:
        source_id, _ = self.store.add_memory(
            "This deployment setting applies only to the Brain staging service during migration.",
            kind="operational",
            source_category="USER_EXPLICIT",
            context_mode="context_dependent",
            scope={"project": "Brain", "environment": "staging"},
            entities=["Brain staging service"],
            preconditions={"migration": "in progress"},
            source_context="Recorded while preparing the staging migration.",
            applicable_systems=["cortex-dashboard"],
            applicable_versions=["18"],
            valid_from="2026-07-01T00:00:00+00:00",
            valid_to="2026-08-01T00:00:00+00:00",
        )
        source = self.store.get_memory(source_id)
        result = self.store.apply_refinery_action(
            "rewrite",
            source_id,
            proposed_records=[
                {
                    "content": (
                        "During the migration, Brain staging uses the temporary dashboard setting."
                    )
                }
            ],
            actor="tester",
        )
        created = self.store.get_memory(result["created_memory_ids"][0])
        for field in (
            "context_mode",
            "scope",
            "entities",
            "preconditions",
            "applicable_systems",
            "applicable_versions",
            "valid_from",
            "valid_to",
            "metadata_completeness",
        ):
            self.assertEqual(created[field], source[field], field)
        self.assertIn(source["source_context"], created["source_context"])
        self.assertIn("Clarity review", created["source_context"])

    def test_split_creates_bounded_standalone_memories(self) -> None:
        source_id, _ = self.store.add_memory(
            "- The proxy listens on port 8443 for external traffic.\n"
            "- The dashboard binds to localhost only.\n"
            "- Backups run nightly at three.",
            kind="semantic",
        )
        preview = self.store.refinery_preview("split", source_id)
        self.assertGreaterEqual(len(preview["proposed_records"]), 2)
        with self.assertRaises(ValueError):
            self.store.apply_refinery_action(
                "split", source_id, proposed_records=[{"content": "only one part"}]
            )
        result = self.store.apply_refinery_action(
            "split",
            source_id,
            proposed_records=[
                {"content": "The proxy listens on port 8443 for external traffic."},
                {"content": "The dashboard binds to localhost only."},
            ],
            actor="tester",
        )
        self.assertEqual(len(result["created_memory_ids"]), 2)
        for created_id in result["created_memory_ids"]:
            self.assertEqual(
                [row["evidence_id"] for row in self.store.dependencies(created_id)], [source_id]
            )
        with self.assertRaises(ValueError):
            self.store.apply_refinery_action(
                "split",
                source_id,
                proposed_records=[{"content": "part"}],
                decision_scope="exact_duplicates",
            )

    def test_exact_duplicate_scope_and_teach_scope_stay_distinct(self) -> None:
        first_id, _ = self.store.add_memory(TABLE_CONTENT, kind="semantic", session_id="one")
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (first_id,)).fetchone()
            duplicate_id = "duplicate-" + first_id[:8]
            values = dict(row)
            values["id"] = duplicate_id
            columns = ",".join(values.keys())
            placeholders = ",".join("?" for _ in values)
            conn.execute(
                f"INSERT INTO memories({columns}) VALUES({placeholders})", tuple(values.values())
            )
            conn.execute("INSERT INTO memory_fts(memory_id,content) VALUES(?,?)", (duplicate_id, values["content"]))
        result = self.store.apply_refinery_action(
            "keep_canonical", first_id, decision_scope="exact_duplicates", actor="tester"
        )
        self.assertEqual(sorted(result["affected_memory_ids"]), sorted([first_id, duplicate_id]))
        self.assertEqual(self.store.get_memory(duplicate_id)["record_role"], "canonical")
        with self.store._lock:
            decision = self.store._conn.execute(
                "SELECT decision_scope,learning_signal_json FROM operator_review_decisions WHERE review_id=?",
                (result["review_id"],),
            ).fetchone()
        self.assertEqual(decision["decision_scope"], "exact_duplicates")
        teach = self.store.apply_refinery_action(
            "keep_reference", first_id, decision_scope="policy_evidence", actor="tester"
        )
        with self.store._lock:
            teach_row = self.store._conn.execute(
                "SELECT decision_scope FROM operator_review_decisions WHERE review_id=?",
                (teach["review_id"],),
            ).fetchone()
        self.assertEqual(teach_row["decision_scope"], "policy_evidence")
        self.assertEqual(teach["affected_memory_ids"], [first_id])

    def test_reason_is_optional_and_archive_trash_are_reversible(self) -> None:
        memory_id, _ = self.store.add_memory(CONFIG_CONTENT, kind="operational")
        archived = self.store.apply_refinery_action("archive", memory_id, actor="tester")
        self.assertEqual(self.store.get_memory(memory_id)["state"], "archived")
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT reason_code,reason_text FROM operator_review_decisions WHERE review_id=?",
                (archived["review_id"],),
            ).fetchone()
        self.assertEqual(row["reason_code"], "unspecified")
        self.assertIsNone(row["reason_text"])
        self.assertTrue(self.store.undo_refinery_action(archived["review_id"], actor="tester"))
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")
        trashed = self.store.apply_refinery_action("trash", memory_id, actor="tester")
        self.assertEqual(self.store.get_memory(memory_id)["state"], "tombstoned")
        self.assertTrue(self.store.undo_refinery_action(trashed["review_id"], actor="tester"))
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")

    def test_undo_never_overwrites_later_changes(self) -> None:
        memory_id, _ = self.store.add_memory(DIAGRAM_CONTENT, kind="semantic")
        archived = self.store.apply_refinery_action("archive", memory_id, actor="tester")
        self.store.set_state(memory_id, "active", reason="operator restored it manually")
        self.assertTrue(self.store.undo_refinery_action(archived["review_id"], actor="tester"))
        memory = self.store.get_memory(memory_id)
        self.assertEqual(memory["state"], "active")
        events = [
            event
            for event in self.store._conn.execute(
                "SELECT reason FROM lifecycle_events WHERE memory_id=? ORDER BY event_id", (memory_id,)
            ).fetchall()
        ]
        self.assertNotIn("refinery undo by tester", [str(event["reason"]) for event in events][-1:])

    def test_backfill_assigns_roles_without_touching_records(self) -> None:
        code_id, _ = self.store.add_memory(CODE_CONTENT, kind="procedure")
        preference_id, _ = self.store.add_memory(
            "Jacoby prefers signal-dense weekly summaries.",
            kind="preference",
            source_category="USER_EXPLICIT",
        )
        before = {
            memory_id: {
                key: self.store.get_memory(memory_id)[key]
                for key in ("content", "state", "created_at", "content_hash")
            }
            for memory_id in (code_id, preference_id)
        }
        version_count = self.store.stats()["memory_presentations"]
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE memories SET record_role='canonical',role_method='legacy_default',role_version=NULL"
            )
            self.store._conn.execute("DELETE FROM memory_presentations")
            self.store._conn.execute("DELETE FROM meta WHERE key=?", (REFINERY_BACKFILL_KEY,))
            self.store._conn.commit()
        path = self.store.path
        self.store.close()
        self.store = CortexStore(path)
        self.assertEqual(self.store.get_memory(code_id)["record_role"], "reference")
        self.assertEqual(self.store.get_memory(preference_id)["record_role"], "canonical")
        self.assertEqual(self.store.stats()["memory_presentations"], version_count)
        for memory_id, values in before.items():
            after = self.store.get_memory(memory_id)
            for key, value in values.items():
                self.assertEqual(after[key], value, key)
        audit = self.store.audit()
        self.assertTrue(audit["ok"], audit)

    def test_operator_reviewed_roles_survive_backfill(self) -> None:
        memory_id, _ = self.store.add_memory(CODE_CONTENT, kind="procedure")
        self.store.apply_refinery_action("keep_canonical", memory_id, actor="tester")
        with self.store._lock:
            self.store._conn.execute("DELETE FROM meta WHERE key=?", (REFINERY_BACKFILL_KEY,))
            self.store._conn.execute("UPDATE memories SET role_version='stale' WHERE id=?", (memory_id,))
            self.store._conn.commit()
        path = self.store.path
        self.store.close()
        self.store = CortexStore(path)
        memory = self.store.get_memory(memory_id)
        self.assertEqual(memory["record_role"], "canonical")
        self.assertEqual(memory["role_method"], "operator_review")


class RefineryMigrationTests(unittest.TestCase):
    def test_legacy_database_gains_roles_and_presentations_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL,
                    source_type TEXT NOT NULL, source_ref TEXT, session_id TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, valid_from TEXT, valid_to TEXT, confidence REAL NOT NULL,
                    importance REAL NOT NULL, volatility REAL NOT NULL, trust REAL NOT NULL, state TEXT NOT NULL,
                    pinned INTEGER NOT NULL, quarantine_reason TEXT, retrieved_count INTEGER NOT NULL,
                    injected_count INTEGER NOT NULL, used_count INTEGER NOT NULL, success_count INTEGER NOT NULL,
                    confirmed_count INTEGER NOT NULL, correction_count INTEGER NOT NULL,
                    false_positive_count INTEGER NOT NULL, duplicate_count INTEGER NOT NULL,
                    last_retrieved_at TEXT, last_injected_at TEXT, last_used_at TEXT
                );
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            rows = [
                ("legacy-pref", "preference", "Jacoby prefers dense dashboards.", "hash-pref"),
                ("legacy-code", "procedure", CODE_CONTENT, "hash-code"),
            ]
            for memory_id, kind, content, digest in rows:
                conn.execute(
                    "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        memory_id, kind, content, digest, "conversation", None, "legacy", now, now,
                        None, None, 0.7, 0.6, 0.4, 0.7, "active", 0, None, 0, 0, 0, 0, 0, 0, 0, 0,
                        None, None, None,
                    ),
                )
            conn.commit()
            conn.close()

            store = CortexStore(db_path)
            try:
                preference = store.get_memory("legacy-pref")
                code = store.get_memory("legacy-code")
                self.assertEqual(preference["content"], "Jacoby prefers dense dashboards.")
                self.assertEqual(preference["state"], "active")
                # Legacy rows migrate with the AGENT_INFERENCE default source
                # category and no dependencies, so they classify as claims —
                # the same records today's unsupported-inference queue shows.
                self.assertEqual(preference["record_role"], "claim")
                self.assertEqual(code["record_role"], "reference")
                self.assertEqual(store.stats()["memory_presentations"], 2)
                summary = store.refinery_summary()
                self.assertTrue(summary["backfill"]["complete"])
                self.assertEqual(summary["role_counts"].get("reference"), 1)
                with store._lock:
                    states = {
                        str(row["id"]): str(row["state"])
                        for row in store._conn.execute("SELECT id,state FROM memories").fetchall()
                    }
                self.assertEqual(set(states.values()), {"active"})
            finally:
                store.close()


class RefineryVaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.vault = root / "vault"
        self.vault.mkdir()
        (self.vault / "Runbook.md").write_text(
            "# Runbook\n\n## Restart procedure\n\nRestart the collector with `systemctl restart collector` "
            "after every config change, then confirm the health endpoint responds.\n",
            encoding="utf-8",
        )
        self.store = CortexStore(root / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_vault_imports_default_to_reference_and_stay_idempotent(self) -> None:
        indexer = VaultIndexer(self.store, self.vault)
        first = indexer.apply()
        self.assertTrue(first["audit"]["ok"], first["audit"])
        chunks = self.store.document_chunks("Runbook.md", active_only=True)
        self.assertTrue(chunks)
        for chunk in chunks:
            memory = self.store.get_memory(chunk["memory_id"])
            self.assertEqual(memory["record_role"], "reference")
        memory_count = self.store.stats()["memories"]
        presentation_count = self.store.stats()["memory_presentations"]
        second = indexer.apply()
        self.assertEqual(second["memories_created"], 0)
        self.assertEqual(self.store.stats()["memories"], memory_count)
        self.assertEqual(self.store.stats()["memory_presentations"], presentation_count)
        self.assertTrue(second["audit"]["ok"])

    def test_revised_vault_section_keeps_id_and_refreshes_presentation(self) -> None:
        indexer = VaultIndexer(self.store, self.vault)
        indexer.apply()
        chunk = self.store.document_chunks("Runbook.md", active_only=True)[0]
        first_presentation = self.store.explain(chunk["memory_id"])["presentation"]
        (self.vault / "Runbook.md").write_text(
            "# Runbook\n\n## Restart procedure\n\nRestart the collector with `systemctl restart collector` "
            "after every config change, wait thirty seconds, then confirm the health endpoint responds.\n",
            encoding="utf-8",
        )
        indexer.apply()
        revised = self.store.document_chunks("Runbook.md", active_only=True)[0]
        self.assertEqual(revised["memory_id"], chunk["memory_id"])
        second_presentation = self.store.explain(chunk["memory_id"])["presentation"]
        self.assertNotEqual(first_presentation["source_digest"], second_presentation["source_digest"])
        self.assertEqual(self.store.audit()["stale_memory_presentations"], 0)


class RefineryRetrievalSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "retrieval.db")
        self.queries = (
            "restart collector service",
            "grafana port",
            "commit message preferences",
            "prowlarr_get_indexers usage",
        )
        self.store.add_memory(
            "Jacoby prefers concise commit messages under seventy characters.",
            kind="preference",
            source_category="USER_EXPLICIT",
        )
        self.store.add_memory(TABLE_CONTENT, kind="operational")
        self.store.add_memory(CODE_CONTENT, kind="procedure")
        self.store.add_memory(
            "Vault note: Tools\nPath: tools/Tools.md\nSection: MCP\n"
            "`prowlarr_get_indexers` lists indexers and `prowlarr_test_indexer` verifies one by id.",
            kind="procedure",
            source_type="vault_markdown",
            source_category="DOCUMENT_EXTRACTED",
            record_role="reference",
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _frozen_results(self) -> str:
        retriever = MemoryRetriever(self.store)
        output = []
        for query in self.queries:
            results, _diagnostics = retriever.search_detailed(query, limit=6, token_budget=900)
            output.append(
                [
                    {"id": result.memory["id"], "score": round(result.score, 6)}
                    for result in results
                ]
            )
        return json.dumps(output, sort_keys=True)

    def test_stage1_selection_and_order_are_identical_before_and_after_classification(self) -> None:
        classified = self._frozen_results()
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE memories SET record_role='canonical',role_method='legacy_default',role_version=NULL"
            )
            self.store._conn.execute("DELETE FROM memory_presentations")
            self.store._conn.commit()
        legacy = self._frozen_results()
        self.assertEqual(classified, legacy)
        with self.store._lock:
            self.store._conn.execute("DELETE FROM meta WHERE key=?", (REFINERY_BACKFILL_KEY,))
            self.store._conn.commit()
        path = self.store.path
        self.store.close()
        self.store = CortexStore(path)
        reclassified = self._frozen_results()
        self.assertEqual(classified, reclassified)

    def test_shadow_comparison_is_versioned_and_does_not_mutate(self) -> None:
        retriever = MemoryRetriever(self.store)
        with self.store._lock:
            before = self.store._conn.execute(
                "SELECT COUNT(*) n FROM access_log"
            ).fetchone()["n"]
        comparison = retriever.shadow_tiered_comparison("restart collector service")
        self.assertEqual(comparison["policy_version"], SHADOW_ROLE_POLICY_VERSION)
        self.assertIn("live_selected_ids", comparison)
        with self.store._lock:
            after = self.store._conn.execute("SELECT COUNT(*) n FROM access_log").fetchone()["n"]
        self.assertEqual(before, after)

    def test_shadow_keeps_directly_supported_technical_reference(self) -> None:
        retriever = MemoryRetriever(self.store)
        comparison = retriever.shadow_tiered_comparison("prowlarr_get_indexers usage")
        live = set(comparison["live_selected_ids"])
        shadow = set(comparison["shadow_selected_ids"])
        technical = [
            memory_id
            for memory_id in live
            if "prowlarr_get_indexers" in str(self.store.get_memory(memory_id)["content"])
        ]
        self.assertTrue(technical)
        for memory_id in technical:
            self.assertIn(memory_id, shadow)

    def test_shadow_gates_unsupported_claims(self) -> None:
        claim_id, _ = self.store.add_memory(
            "The operator appears to prefer reviewing restart collector service logs late at night.",
            kind="semantic",
            source_category="AGENT_INFERENCE",
        )
        self.assertEqual(self.store.get_memory(claim_id)["record_role"], "claim")
        retriever = MemoryRetriever(self.store)
        comparison = retriever.shadow_tiered_comparison("restart collector service logs")
        self.assertNotIn(claim_id, comparison["shadow_selected_ids"])
        gates = {row["memory_id"]: row for row in comparison["role_gates"]}
        if claim_id in gates:
            self.assertFalse(gates[claim_id]["eligible"])

    def test_shadow_gate_blocks_graph_only_reference(self) -> None:
        gate = MemoryRetriever._shadow_role_gate(
            "reference",
            {"lexical": 0.02, "phrase": 0.0, "graph": 0.8, "currentness": 0.7},
            content_preview="unrelated stored reference material",
            query_technical_tokens=set(),
        )
        self.assertFalse(gate["eligible"])
        supported = MemoryRetriever._shadow_role_gate(
            "reference",
            {"lexical": 0.4, "phrase": 0.0, "graph": 0.8, "currentness": 0.7},
            content_preview="matching reference",
            query_technical_tokens=set(),
        )
        self.assertTrue(supported["eligible"])


if __name__ == "__main__":
    unittest.main()
