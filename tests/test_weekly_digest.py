"""The weekly digest reports, ranks and refuses to flatter (#181)."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import weekly_digest as wd
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    # -- fixtures ---------------------------------------------------------------------------
    def memory(self, text: str, *, days_ago: int = 0, category: str = "USER_STATED",
               entities: list[str] | None = None, retired_days_ago: int | None = None) -> str:
        memory_id = self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category=category,
        )[0]
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        retired = ((NOW - timedelta(days=retired_days_ago)).isoformat()
                   if retired_days_ago is not None else None)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ?, valid_to = ? WHERE id = ?",
                         (stamp, retired, memory_id))
            if entities is not None:
                conn.execute("UPDATE memories SET entities_json = ? WHERE id = ?",
                             (json.dumps(entities), memory_id))
        return memory_id

    def run_row(self, *, prepare_ms: float, abstained: int = 0, days_ago: int = 0) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO recall_runs(recall_id, mode, requested_limit, token_budget, "
                "candidate_count, selected_count, prepare_ms, abstained, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (f"run-{prepare_ms}-{days_ago}-{abstained}", "focused", 6, 620, 50, 4,
                 prepare_ms, abstained, (NOW - timedelta(days=days_ago)).isoformat()),
            )

    def usage_row(self, memory_id: str, *, selected: int = 1, used: int = 0,
                  days_ago: int = 0) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO usage_records(usage_id, task_id, memory_id, selected, used, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid.uuid4()), f"task-{memory_id}", memory_id,
                 selected, used, (NOW - timedelta(days=days_ago)).isoformat()),
            )

    def maintenance(self, *, dry_run: int, action: str = "decay", days_ago: int = 0) -> None:
        with self.store.transaction() as conn:
            conn.execute("INSERT INTO maintenance_log(action, details, dry_run, created_at) "
                         "VALUES(?,?,?,?)",
                         (action, "{}", dry_run, (NOW - timedelta(days=days_ago)).isoformat()))

    def section(self, described, name):
        for section in described.sections:
            if section.name == name:
                return section
        raise AssertionError(f"no section {name}")

    # -- window and figures -----------------------------------------------------------------
    def test_the_window_is_stated_in_bounds_not_now(self) -> None:
        described = wd.digest(self.store, days=7, now=NOW)
        self.assertEqual(described.start[:10], "2026-11-17")
        self.assertEqual(described.end[:10], "2026-11-24")
        self.assertIn("2026-11-17", described.window)
        self.assertIn("2026-11-24", described.window)

    def test_writes_are_counted_by_source(self) -> None:
        self.memory("a fact")
        self.memory("another fact", category="DOCUMENT_EXTRACTED")
        section = self.section(wd.digest(self.store, days=7, now=NOW), "memories written")
        self.assertEqual(section.count, 2)
        self.assertEqual(section.detail.get("DOCUMENT_EXTRACTED"), 1)

    def test_a_memory_outside_the_window_is_not_counted(self) -> None:
        self.memory("an old fact", days_ago=40)
        section = self.section(wd.digest(self.store, days=7, now=NOW), "memories written")
        self.assertEqual(section.count, 0)

    def test_every_figure_names_its_table_and_its_window(self) -> None:
        self.memory("a fact")
        self.run_row(prepare_ms=90)
        described = wd.digest(self.store, days=7, now=NOW)
        for section in described.sections:
            self.assertTrue(section.table, section.name)
            self.assertIn("2026-11-17", section.window)
            self.assertIn("2026-11-24", section.window)
            self.assertIn(section.table, section.sentence())
        for issue in described.issues:
            self.assertTrue(issue.table, issue.key)
            self.assertIn("2026-11-17", issue.window)
            self.assertIn(issue.table, issue.line())

    def test_corrections_and_retirements_are_counted(self) -> None:
        self.memory("the port is 8101", category="USER_CORRECTION")
        self.memory("the port was 8100", retired_days_ago=1)
        described = wd.digest(self.store, days=7, now=NOW)
        self.assertEqual(self.section(described, "corrections recorded").count, 1)
        self.assertEqual(self.section(described, "beliefs retired").count, 1)

    def test_a_retirement_outside_the_window_is_not_counted(self) -> None:
        self.memory("an old belief", days_ago=40, retired_days_ago=30)
        section = self.section(wd.digest(self.store, days=7, now=NOW), "beliefs retired")
        self.assertEqual(section.count, 0)

    def test_new_entities_are_names_not_seen_before_the_window(self) -> None:
        self.memory("an old note about Mack", days_ago=30, entities=["Mack"])
        self.memory("a note about the NAS", entities=["Mack", "Synology NAS"])
        section = self.section(wd.digest(self.store, days=7, now=NOW), "new entities")
        self.assertEqual(section.count, 1)

    def test_an_absent_topic_table_says_unavailable_not_zero(self) -> None:
        section = self.section(wd.digest(self.store, days=7, now=NOW), "new topics")
        self.assertTrue(section.unavailable)
        self.assertIn("unavailable", section.sentence())

    def test_retrieval_quality_reports_abstention_and_use(self) -> None:
        memory_id = self.memory("a fact")
        self.run_row(prepare_ms=400)
        self.run_row(prepare_ms=600, abstained=1)
        self.usage_row(memory_id, selected=1, used=0)
        described = wd.digest(self.store, days=7, now=NOW)
        runs = self.section(described, "retrieval runs")
        self.assertEqual(runs.count, 2)
        self.assertEqual(runs.detail.get("abstained"), 1)
        self.assertEqual(runs.detail.get("p50 ms"), 500)
        self.assertEqual(self.section(described, "retrieval use").detail.get("selected"), 1)

    def test_maintenance_dry_runs_are_separated_from_real_actions(self) -> None:
        self.maintenance(dry_run=1)
        self.maintenance(dry_run=0, action="prune")
        section = self.section(wd.digest(self.store, days=7, now=NOW), "maintenance actions")
        self.assertEqual(section.count, 2)
        self.assertEqual(section.detail.get("dry runs"), 1)

    # -- quiet, unavailable, empty ----------------------------------------------------------
    def test_a_quiet_week_is_reported_as_quiet(self) -> None:
        described = wd.digest(self.store, days=7, now=NOW)
        self.assertEqual(described.status, "quiet")
        self.assertTrue(described.headline().startswith("week 2026-11-17 → 2026-11-24: quiet"))
        self.assertNotIn("unavailable", described.headline())

    def test_a_quiet_week_is_not_an_empty_page(self) -> None:
        lines = wd.digest(self.store, days=7, now=NOW).render()
        self.assertGreater(len(lines), 2)
        self.assertTrue(any("quiet" in line for line in lines))
        self.assertTrue(lines[-1].startswith("next: "))

    def test_a_quiet_week_still_proposes_one_concrete_action(self) -> None:
        action = wd.digest(self.store, days=7, now=NOW).action
        self.assertIn("memories", action)
        self.assertIn("2026-11-17", action)

    def test_a_broken_store_says_unavailable_and_never_quiet(self) -> None:
        class Broken:
            _conn = None

        described = wd.digest(Broken(), days=7, now=NOW)
        self.assertEqual(described.status, "unavailable")
        self.assertIn("unavailable", described.headline())
        self.assertTrue(all(section.unavailable for section in described.sections))
        self.assertIn("could not be read", described.sections[0].sentence())

    def test_the_render_of_a_broken_store_is_not_blank(self) -> None:
        class Broken:
            _conn = None

        rendered = "\n".join(wd.render(Broken(), days=7, now=NOW))
        self.assertIn("unavailable", rendered)
        self.assertTrue(rendered.strip().endswith(rendered.strip().splitlines()[-1]))
        self.assertTrue(rendered.strip().splitlines()[-1].startswith("next: "))

    def test_an_empty_section_is_omitted_from_the_page_but_kept_in_the_data(self) -> None:
        self.memory("a fact")
        described = wd.digest(self.store, days=7, now=NOW)
        rendered = "\n".join(described.render())
        self.assertNotIn("retrieval runs", rendered)
        self.assertEqual(self.section(described, "retrieval runs").count, 0)

    # -- worst metric ordering --------------------------------------------------------------
    def test_the_worst_number_leads_the_page(self) -> None:
        self.memory("a fact")
        self.maintenance(dry_run=1)
        self.maintenance(dry_run=1)
        self.run_row(prepare_ms=900)  # (900-150)/150 = 5.0 past the limit
        described = wd.digest(self.store, days=7, now=NOW)
        self.assertEqual(described.worst.key, "latency")
        self.assertGreater(described.worst.severity, 1.0)
        self.assertTrue(described.headline().endswith(
            f"({described.worst.table}, {described.window})"))
        self.assertIn("p50 900 ms", described.render()[1])

    def test_a_rehearsal_week_outranks_a_fast_but_unused_store(self) -> None:
        memory_id = self.memory("a fact")
        self.maintenance(dry_run=1)
        for _ in range(4):
            self.usage_row(memory_id, selected=1, used=0, days_ago=1)
        self.run_row(prepare_ms=20)
        described = wd.digest(self.store, days=7, now=NOW)
        self.assertEqual(described.worst.key, "maintenance_rehearsal")
        self.assertEqual(described.issues[0].key, "maintenance_rehearsal")

    def test_issues_are_ordered_by_severity_then_key(self) -> None:
        memory_id = self.memory("a fact")
        self.maintenance(dry_run=1)
        self.usage_row(memory_id, selected=1, used=0)
        self.run_row(prepare_ms=1500)
        issues = wd.digest(self.store, days=7, now=NOW).issues
        rankings = [(-issue.severity, issue.key) for issue in issues]
        self.assertEqual(rankings, sorted(rankings))

    def test_the_next_action_belongs_to_the_worst_issue(self) -> None:
        self.memory("a fact")
        self.maintenance(dry_run=1)
        self.run_row(prepare_ms=900)
        described = wd.digest(self.store, days=7, now=NOW)
        self.assertEqual(described.action, described.worst.action)
        self.assertIn("prepare_ms", described.action)

    def test_the_action_changes_with_the_week(self) -> None:
        """A static list of tips would print the same sentence for both of these."""
        self.maintenance(dry_run=1)
        rehearsal = wd.digest(self.store, days=7, now=NOW)

        other = CortexStore(Path(self.tmp.name) / "other.db")
        try:
            memory_id = other.add_memory(content="a fact", kind="semantic",
                                         source_type="user_turn",
                                         source_category="USER_STATED")[0]
            # A correction in the window disqualifies the "never corrected" issue, so the two
            # stores differ in exactly one axis: rehearsal versus an unused retrieval pool.
            corrected = other.add_memory(content="the port is 8101", kind="semantic",
                                         source_type="user_turn",
                                         source_category="USER_CORRECTION")[0]
            with other.transaction() as conn:
                conn.execute("UPDATE memories SET created_at = ? WHERE id IN (?, ?)",
                             (NOW.isoformat(), memory_id, corrected))
                conn.execute(
                    "INSERT INTO usage_records(usage_id, task_id, memory_id, selected, used, "
                    "created_at) VALUES(?,?,?,?,?,?)",
                    ("u1", "t1", memory_id, 1, 0, NOW.isoformat()),
                )
            unused = wd.digest(other, days=7, now=NOW)
        finally:
            other.close()
        self.assertEqual(rehearsal.worst.key, "maintenance_rehearsal")
        self.assertEqual(unused.worst.key, "unused_retrievals")
        self.assertNotEqual(rehearsal.action, unused.action)

    # -- determinism ------------------------------------------------------------------------
    def test_the_output_is_deterministic_for_a_fixed_window(self) -> None:
        memory_id = self.memory("a fact")
        self.usage_row(memory_id, selected=1, used=0)
        self.run_row(prepare_ms=800, abstained=1)
        self.maintenance(dry_run=1)
        first = wd.digest(self.store, days=7, now=NOW)
        second = wd.digest(self.store, days=7, now=NOW)
        self.assertEqual(first.render(), second.render())
        self.assertEqual(json.dumps(first.as_dict(), sort_keys=True),
                         json.dumps(second.as_dict(), sort_keys=True))

    def test_no_clock_is_embedded_beyond_the_window_bounds(self) -> None:
        self.memory("a fact")
        self.run_row(prepare_ms=800)
        described = wd.digest(self.store, days=7, now=NOW)
        blob = json.dumps(described.as_dict(), sort_keys=True) + "\n".join(described.render())
        stamps = set(_STAMP.findall(blob))
        self.assertEqual(stamps, {described.start[:16], described.end[:16]})

    def test_a_different_window_gives_a_different_answer(self) -> None:
        self.memory("a fact", days_ago=20)
        week = wd.digest(self.store, days=7, now=NOW)
        month = wd.digest(self.store, days=30, now=NOW)
        self.assertEqual(self.section(week, "memories written").count, 0)
        self.assertEqual(self.section(month, "memories written").count, 1)

    def test_the_data_carries_every_section_even_when_the_page_does_not(self) -> None:
        described = wd.digest(self.store, days=7, now=NOW)
        payload = described.as_dict()
        names = [section["name"] for section in payload["sections"]]
        for expected in ("memories written", "corrections recorded", "beliefs retired",
                         "new entities", "new topics", "retrieval runs", "retrieval use",
                         "maintenance actions", "lifecycle changes"):
            self.assertIn(expected, names)
        self.assertIn("status", payload)
        self.assertIn("action", payload)


if __name__ == "__main__":
    unittest.main()
