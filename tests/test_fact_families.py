"""Group variant phrasings of one fact so they age together (#56)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import fact_families
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, confidence: float = 0.8,
               used: datetime | None = None, uses: int = 0) -> str:
        memory_id = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", confidence=confidence,
        )[0]
        if used is not None or uses:
            with self.store.transaction() as conn:
                conn.execute(
                    "UPDATE memories SET last_used_at = ?, used_count = ? WHERE id = ?",
                    (used.isoformat() if used else None, uses, memory_id),
                )
        return memory_id

    def test_two_phrasings_of_one_fact_share_a_family(self) -> None:
        first = self.memory("The backup window is 2am every night.")
        second = self.memory("The backup window is 2am every night")
        families = fact_families.build(self.store, persist=False)
        grouped = [family for family in families if family.size == 2]
        self.assertEqual(len(grouped), 1)
        self.assertEqual(set(grouped[0].members), {first, second})

    def test_unrelated_facts_stay_apart(self) -> None:
        self.memory("The backup window is 2am every night.")
        self.memory("The dashboard listens on port 8100 behind nginx.")
        families = fact_families.build(self.store, persist=False)
        self.assertEqual([family for family in families if family.size > 1], [])

    def test_a_restatement_joins_the_family(self) -> None:
        first = self.memory("Backups run at 2am on the r630 host every night.")
        second = self.memory("Backups run at 2am on the r630 host.")
        families = fact_families.build(self.store, persist=False)
        grouped = [family for family in families if family.size > 1]
        self.assertEqual(len(grouped), 1, [family.as_dict() for family in families])
        self.assertEqual(set(grouped[0].members), {first, second})

    def test_the_representative_is_the_strongest_member(self) -> None:
        weak = self.memory("The backup window is 2am every night.", confidence=0.4)
        strong = self.memory("The backup window is 2am every night", confidence=0.95)
        family = [f for f in fact_families.build(self.store, persist=False) if f.size == 2][0]
        self.assertEqual(family.representative, strong)
        self.assertIn(weak, family.members)

    def test_the_anchor_is_the_freshest_use_in_the_family(self) -> None:
        stale = self.memory("The backup window is 2am every night.",
                            used=NOW - timedelta(days=90), uses=1)
        fresh = self.memory("The backup window is 2am every night", used=NOW, uses=3)
        fact_families.build(self.store)
        found = fact_families.anchor(self.store, stale)
        self.assertEqual(found["size"], 2)
        self.assertEqual(found["last_used"][:10], NOW.isoformat()[:10])
        self.assertEqual(found["uses"], 4)
        self.assertIn(fresh, found["members"])

    def test_family_of_returns_self_for_a_singleton(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100 behind nginx.")
        fact_families.build(self.store)
        self.assertEqual(fact_families.family_of(self.store, memory_id), [memory_id])

    def test_family_of_an_unknown_memory_returns_itself(self) -> None:
        self.assertEqual(fact_families.family_of(self.store, "no-such-id"), ["no-such-id"])

    def test_building_twice_changes_nothing(self) -> None:
        first = self.memory("The backup window is 2am every night.")
        second = self.memory("The backup window is 2am every night")
        fact_families.build(self.store)
        before = fact_families.family_of(self.store, first)
        fact_families.build(self.store)
        self.assertEqual(fact_families.family_of(self.store, first), before)
        self.assertEqual(set(before), {first, second})

    def test_an_archived_member_is_included(self) -> None:
        first = self.memory("The backup window is 2am every night.")
        second = self.memory("The backup window is 2am every night")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (second,))
        family = [f for f in fact_families.build(self.store, persist=False) if f.size == 2]
        self.assertEqual(len(family), 1)

    def test_aging_is_off_by_default(self) -> None:
        self.assertFalse(fact_families.enabled({}))

    def test_aging_switches_on_by_environment(self) -> None:
        self.assertTrue(fact_families.enabled({"CORTEX_FAMILY_AGING": "1"}))

    def test_aging_does_nothing_when_off(self) -> None:
        stale = self.memory("The backup window is 2am every night.",
                            used=NOW - timedelta(days=90), uses=1)
        self.memory("The backup window is 2am every night", used=NOW, uses=3)
        fact_families.build(self.store)
        own = NOW - timedelta(days=90)
        self.assertEqual(fact_families.apply_to(self.store, stale, own, environ={}), own)

    def test_aging_uses_the_family_anchor_when_on(self) -> None:
        stale = self.memory("The backup window is 2am every night.",
                            used=NOW - timedelta(days=90), uses=1)
        self.memory("The backup window is 2am every night", used=NOW, uses=3)
        fact_families.build(self.store)
        own = NOW - timedelta(days=90)
        aged = fact_families.apply_to(self.store, stale, own, environ={"CORTEX_FAMILY_AGING": "1"})
        self.assertEqual(aged, NOW)

    def test_aging_never_makes_a_fact_older(self) -> None:
        fresh = self.memory("The backup window is 2am every night.", used=NOW, uses=2)
        self.memory("The backup window is 2am every night", used=NOW - timedelta(days=30), uses=1)
        fact_families.build(self.store)
        self.assertEqual(
            fact_families.apply_to(self.store, fresh, NOW, environ={"CORTEX_FAMILY_AGING": "1"}),
            NOW,
        )

    def test_aging_without_a_built_table_is_harmless(self) -> None:
        memory_id = self.memory("The dashboard listens on port 8100.")
        self.assertEqual(
            fact_families.apply_to(self.store, memory_id, NOW, environ={"CORTEX_FAMILY_AGING": "1"}),
            NOW,
        )

    def test_the_report_counts_grouped_families(self) -> None:
        self.memory("The backup window is 2am every night.")
        self.memory("The backup window is 2am every night")
        self.memory("The dashboard listens on port 8100 behind nginx.")
        described = fact_families.report(self.store)
        self.assertEqual(described["families"], 1)
        self.assertEqual(described["memories"], 2)
        self.assertEqual(described["largest"], 2)

    def test_a_family_describes_itself(self) -> None:
        self.memory("The backup window is 2am every night.", used=NOW, uses=1)
        self.memory("The backup window is 2am every night")
        family = [f for f in fact_families.build(self.store, persist=False) if f.size == 2][0]
        self.assertIn("phrasing", family.describe())

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(fact_families.build(self.store, persist=False), [])
        self.assertEqual(fact_families.report(self.store)["families"], 0)


if __name__ == "__main__":
    unittest.main()
