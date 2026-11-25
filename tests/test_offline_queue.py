"""Writes that happen when the store doesn't (#157)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import offline_queue as oq
from cortex.store import CortexStore


class QueueFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "queue.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_absent_queue_reads_as_empty_not_as_an_error(self) -> None:
        found = oq.load(self.path)
        self.assertEqual(found.entries, [])
        self.assertIn("no queue file", found.note)

    def test_enqueueing_writes_the_file_before_saying_yes(self) -> None:
        """A crash between 'queued' and 'stored' must leave a replayable record."""
        result = oq.enqueue(self.path, content="the dashboard listens on port 8100")
        self.assertTrue(result["queued"])
        self.assertTrue(self.path.exists())
        self.assertEqual(len(oq.load(self.path).entries), 1)

    def test_the_same_thought_is_not_queued_twice(self) -> None:
        oq.enqueue(self.path, content="a fact")
        second = oq.enqueue(self.path, content="a fact")
        self.assertFalse(second["queued"])
        self.assertIn("already queued", second["reason"])

    def test_whitespace_does_not_defeat_the_dedupe(self) -> None:
        oq.enqueue(self.path, content="a  fact")
        self.assertFalse(oq.enqueue(self.path, content="a fact")["queued"])

    def test_an_empty_write_is_refused(self) -> None:
        self.assertFalse(oq.enqueue(self.path, content="   ")["queued"])

    def test_a_full_queue_refuses_the_newest_and_says_so(self) -> None:
        for index in range(3):
            oq.enqueue(self.path, content=f"fact {index}", max_entries=3)
        result = oq.enqueue(self.path, content="fact 4", max_entries=3)
        self.assertFalse(result["queued"])
        self.assertIn("queue is full", result["reason"])
        self.assertEqual(result["size"], 3)

    def test_a_corrupt_line_is_skipped_and_counted(self) -> None:
        oq.enqueue(self.path, content="a good fact")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        found = oq.load(self.path)
        self.assertEqual(len(found.entries), 1)
        self.assertEqual(found.refused, 1)

    def test_the_key_is_stable(self) -> None:
        self.assertEqual(oq.dedupe_key("semantic", "a fact"), oq.dedupe_key("semantic", "a  fact"))

    def test_the_key_separates_kinds(self) -> None:
        self.assertNotEqual(oq.dedupe_key("semantic", "a fact"),
                            oq.dedupe_key("preference", "a fact"))


class DrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "queue.jsonl"
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def count(self) -> int:
        return int(self.store._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"])

    def test_a_drain_stores_everything_and_empties_the_queue(self) -> None:
        oq.enqueue(self.path, content="the dashboard listens on port 8100")
        oq.enqueue(self.path, content="the reverse proxy sits in front of it")
        result = oq.drain(self.store, self.path)
        self.assertEqual(result["drained"], 2)
        self.assertEqual(result["remaining"], 0)
        self.assertEqual(self.count(), 2)

    def test_draining_twice_stores_nothing_twice(self) -> None:
        """A replayed write is idempotent, or a flaky network doubles the store."""
        oq.enqueue(self.path, content="the dashboard listens on port 8100")
        oq.drain(self.store, self.path)
        oq.enqueue(self.path, content="the dashboard listens on port 8100")
        second = oq.drain(self.store, self.path)
        self.assertEqual(second["drained"], 0)
        self.assertEqual(self.count(), 1)

    def test_an_empty_queue_drains_to_nothing(self) -> None:
        result = oq.drain(self.store, self.path)
        self.assertEqual(result["drained"], 0)

    def test_order_is_kept_when_replay_stops(self) -> None:
        oq.enqueue(self.path, content="first fact about the dashboard")
        oq.enqueue(self.path, content="second fact about the dashboard")
        original = self.store.add_memory
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("the store went away mid-replay")
            return original(**kwargs)

        self.store.add_memory = flaky  # type: ignore[assignment]
        try:
            result = oq.drain(self.store, self.path)
        finally:
            self.store.add_memory = original  # type: ignore[assignment]
        self.assertTrue(result["stopped_on"])
        self.assertEqual(result["remaining"], 2)
        remaining = oq.load(self.path).entries
        self.assertIn("first fact", remaining[0].content)
        self.assertIn("second fact", remaining[1].content)

    def test_the_attempt_count_is_recorded(self) -> None:
        oq.enqueue(self.path, content="a fact about the dashboard")
        original = self.store.add_memory
        self.store.add_memory = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore[assignment]
        try:
            oq.drain(self.store, self.path)
        finally:
            self.store.add_memory = original  # type: ignore[assignment]
        self.assertEqual(oq.status(self.path)["attempts"], 1)

    def test_status_and_render_report_the_queue(self) -> None:
        oq.enqueue(self.path, content="a fact")
        self.assertEqual(oq.status(self.path)["entries"], 1)
        rendered = "\n".join(oq.render(self.path))
        self.assertIn("offline queue", rendered)
        self.assertIn("ordered", rendered)

    def test_a_broken_store_stops_the_drain_without_losing_the_queue(self) -> None:
        oq.enqueue(self.path, content="a fact about the dashboard")

        class Broken:
            _conn = None

            def add_memory(self, **kwargs):
                raise RuntimeError("no store")

        result = oq.drain(Broken(), self.path)
        self.assertEqual(result["remaining"], 1)


if __name__ == "__main__":
    unittest.main()
