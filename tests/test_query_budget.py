"""A per-PR gate on how many SQL statements one recall costs (#219).

The latency regression that #219 exists to clean up did not arrive as one big change — it
arrived as *one extra query per turn, thirty-nine times*. Wall-clock timing cannot gate that
in CI: a shared runner is too noisy, so a p50 budget either flakes or gets raised until it
means nothing.

Statement count can. It is deterministic, it does not depend on the machine, and the thing it
measures is exactly the mistake being made. Cached inputs do not appear here at all — that is
the point: the fix for a query-count regression is usually to cache, not to micro-optimise.

This is a ceiling, not a target: raising it should require saying which new query earned the
slot and why the store revision could not cover it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore

# Application-level statements per recall, measured after the precompute work (#44, #219).
#
# Raw `set_trace_callback` counts 102 for this fixture, which sounds alarming and is not: 91
# of those are SQLite's own FTS internals (`memory_fts_idx` probes, `data_version` reads).
# The number that reflects a design mistake is the one the application issues: 11.
STATEMENT_BUDGET = 12


def _is_application_statement(statement: str) -> bool:
    """SQLite's own FTS probes and the module's data_version reads are not our queries."""
    text = " ".join(statement.split())
    if text.startswith("--"):
        return False
    return not text.startswith(("PRAGMA data_version", "PRAGMA 'main'.data_version"))

FACTS = [
    ("The maintenance window is Sunday 02:00 on CT116.", ["CT116"]),
    ("Seerr notifications route through the Hermes bot.", ["Seerr", "Hermes"]),
    ("The Temple app keeps a crimson accent on the workout screen.", ["Temple"]),
    ("Cortex stores memories in SQLite with a WAL journal.", ["Cortex"]),
    ("Mack's access uses a shared vault and a per-user key.", ["Mack"]),
]


class QueryBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        for index in range(6):
            for content, entities in FACTS:
                self.store.add_memory(
                    content=f"{content} (variant {index})",
                    kind="semantic",
                    source_type="user_turn",
                    source_category="USER_EXPLICIT",
                    entities=entities,
                )
        self.retriever = MemoryRetriever(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _count(self, query: str) -> int:
        counter = {"n": 0}

        def trace(statement: str) -> None:
            if _is_application_statement(statement):
                counter["n"] += 1

        self.store._conn.set_trace_callback(trace)
        try:
            self.retriever.search_detailed(query, limit=6, token_budget=700)
        finally:
            self.store._conn.set_trace_callback(None)
        return counter["n"]

    def test_one_recall_stays_within_the_statement_budget(self) -> None:
        # Warm first: the first call legitimately builds the caches.
        self._count("maintenance window")
        measured = self._count("maintenance window")
        self.assertLessEqual(
            measured,
            STATEMENT_BUDGET,
            f"one recall now costs {measured} statements (budget {STATEMENT_BUDGET}); "
            "a new per-turn query needs to be cached per store revision instead",
        )

    def test_invalidation_is_temporary_not_accumulating(self) -> None:
        """A write may cost one recompute; it must not make every later turn dearer.

        The first recall after a write legitimately re-reads the memoised sets (7 extra
        statements here). The second turn must be back at the budget — otherwise the cache is
        not a cache, it is a slower copy of the database.
        """
        self._count("maintenance window")
        self.store.add_memory(
            content="The maintenance window moved to Tuesday 03:00 on CT116.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        self._count("maintenance window")  # pays the recompute
        recovered = self._count("maintenance window")
        self.assertLessEqual(
            recovered, STATEMENT_BUDGET, f"the turn after a write costs {recovered} statements"
        )

    def test_the_count_is_stable_across_repeats(self) -> None:
        self._count("seerr notifications")
        counts = {self._count("seerr notifications") for _ in range(3)}
        self.assertEqual(len(counts), 1, f"statement count varies: {counts}")


if __name__ == "__main__":
    unittest.main()
