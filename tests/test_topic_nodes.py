"""A dense region of the graph deserves a name, not another thousand edges (#74)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import topic_nodes
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=1)

CONTAINER_WORDS = ("docker", "compose", "nginx", "proxy", "container")
SHELF_WORDS = ("bookshelf", "shelving", "walnut")


class Broken:
    """A store whose reads fail: every public helper has to stay quiet, not raise."""

    @property
    def _conn(self):
        raise RuntimeError("database is locked")

    def transaction(self):
        raise RuntimeError("database is locked")


class StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def chain(self, members: list[str]) -> None:
        for left, right in zip(members, members[1:]):
            self.store.add_edge(left, right, "related", weight=0.3)

    def cluster(self, words: tuple[str, ...], count: int) -> list[str]:
        made = [
            self.memory(f"{' '.join(words)} note number {index} about {words[index % len(words)]}")
            for index in range(count)
        ]
        self.chain(made)
        return made


class ComponentTests(StoreCase):
    def test_two_clusters_are_two_components(self) -> None:
        first = self.cluster(CONTAINER_WORDS, 8)
        second = self.cluster(SHELF_WORDS, 3)
        found = topic_nodes.components(self.store)
        self.assertEqual(len(found), 2)
        self.assertEqual(sorted(found[0]), sorted(first))
        self.assertEqual(sorted(found[1]), sorted(second))

    def test_the_largest_component_comes_first(self) -> None:
        self.cluster(SHELF_WORDS, 3)
        self.cluster(CONTAINER_WORDS, 8)
        found = topic_nodes.components(self.store)
        self.assertGreater(len(found[0]), len(found[1]))

    def test_members_inside_a_component_are_sorted(self) -> None:
        self.cluster(CONTAINER_WORDS, 5)
        for group in topic_nodes.components(self.store):
            self.assertEqual(group, sorted(group))

    def test_an_edge_written_either_way_round_still_joins(self) -> None:
        late = self.memory("Late in the alphabet.")
        early = self.memory("Early in the alphabet.")
        # `supports` keeps the caller's order, so this row runs backwards through the table.
        self.store.add_edge(late, early, "supports", weight=0.4)
        self.assertEqual(len(topic_nodes.components(self.store)), 1)
        self.assertEqual(sorted(topic_nodes.components(self.store)[0]), sorted([late, early]))

    def test_a_store_with_no_edges_has_no_components(self) -> None:
        self.memory("An island with nothing attached.")
        self.assertEqual(topic_nodes.components(self.store), [])

    def test_a_scan_is_a_bound_not_a_guess(self) -> None:
        first = self.cluster(CONTAINER_WORDS, 6)
        scanned = topic_nodes.components(self.store, scan=1)
        self.assertEqual(len(scanned), 1)
        self.assertEqual(len(scanned[0]), 2)
        self.assertLess(len(scanned[0]), len(first))

    def test_a_scan_of_zero_reads_nothing(self) -> None:
        self.cluster(CONTAINER_WORDS, 4)
        self.assertEqual(topic_nodes.components(self.store, scan=0), [])

    def test_components_are_repeatable(self) -> None:
        self.cluster(CONTAINER_WORDS, 6)
        self.assertEqual(topic_nodes.components(self.store), topic_nodes.components(self.store))


class RegionTests(StoreCase):
    def test_only_regions_at_or_above_the_floor_are_named(self) -> None:
        self.cluster(CONTAINER_WORDS, 8)
        self.cluster(SHELF_WORDS, 3)
        regions = topic_nodes.dense_regions(self.store, min_size=8)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].size, 8)

    def test_a_lower_floor_admits_the_island(self) -> None:
        self.cluster(CONTAINER_WORDS, 8)
        self.cluster(SHELF_WORDS, 3)
        self.assertEqual(len(topic_nodes.dense_regions(self.store, min_size=3)), 2)

    def test_a_region_reports_the_terms_that_repeat_inside_it(self) -> None:
        self.cluster(CONTAINER_WORDS, 8)
        region = topic_nodes.dense_regions(self.store, min_size=8)[0]
        self.assertIn("docker", region.terms)
        self.assertIn("nginx", region.terms)

    def test_a_region_is_named_from_its_own_words(self) -> None:
        self.cluster(SHELF_WORDS, 6)
        self.cluster(CONTAINER_WORDS, 8)
        region = topic_nodes.dense_regions(self.store, min_size=8)[0]
        self.assertTrue(region.name)
        self.assertIn("docker", region.name)

    def test_a_region_describes_itself(self) -> None:
        self.cluster(CONTAINER_WORDS, 8)
        region = topic_nodes.dense_regions(self.store, min_size=8)[0]
        self.assertIn("8 memories", region.describe())
        self.assertEqual(region.as_dict()["size"], 8)

    def test_an_empty_store_has_no_regions(self) -> None:
        self.assertEqual(topic_nodes.dense_regions(self.store), [])


class LabelTests(StoreCase):
    def test_the_label_is_a_short_title(self) -> None:
        self.cluster(CONTAINER_WORDS, 8)
        words = topic_nodes.label(self.store, self.cluster_ids()).split()
        self.assertTrue(1 <= len(words) <= 6)

    def cluster_ids(self) -> list[str]:
        return topic_nodes.components(self.store)[0]

    def test_the_same_members_always_give_the_same_label(self) -> None:
        self.cluster(CONTAINER_WORDS, 8)
        members = self.cluster_ids()
        self.assertEqual(topic_nodes.label(self.store, members), topic_nodes.label(self.store, members))
        self.assertEqual(
            topic_nodes.label(self.store, list(reversed(members))),
            topic_nodes.label(self.store, members),
        )

    def test_a_word_every_memory_uses_is_not_distinctive(self) -> None:
        members = [
            self.memory(f"cortex docker compose nginx proxy note {index}") for index in range(8)
        ]
        self.chain(members)
        for index in range(7):
            self.memory(f"cortex unrelated note {index}")
        members = topic_nodes.components(self.store)[0]
        title = topic_nodes.label(self.store, members)
        self.assertNotIn("cortex", title)
        self.assertIn("docker", title)

    def test_equally_ranked_words_break_alphabetically(self) -> None:
        members = [self.memory("zebra apple") for _ in range(4)]
        self.chain(members)
        self.assertEqual(topic_nodes.label(self.store, members), "apple zebra")

    def test_function_words_never_become_a_label(self) -> None:
        members = [
            self.memory("the and for note about the docker compose of the nginx proxy")
            for _ in range(6)
        ]
        self.chain(members)
        words = topic_nodes.label(self.store, members).split()
        for filler in ("the", "and", "for", "of", "note", "about"):
            self.assertNotIn(filler, words)
        self.assertIn("docker", words)

    def test_no_members_means_no_label(self) -> None:
        self.assertEqual(topic_nodes.label(self.store, []), "")

    def test_unknown_members_do_not_invent_a_label(self) -> None:
        self.assertEqual(topic_nodes.label(self.store, ["no-such-memory"]), "")


class CreateTests(StoreCase):
    def test_a_topic_holds_its_members(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        topic = topic_nodes.create(self.store, members, at=NOW)
        self.assertIsNotNone(topic)
        self.assertEqual(topic.size, 8)
        self.assertEqual(list(topic.members), sorted(members))
        self.assertTrue(topic.name)

    def test_the_same_member_set_is_the_same_topic(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        first = topic_nodes.create(self.store, members, at=NOW)
        second = topic_nodes.create(self.store, list(reversed(members)), at=LATER)
        self.assertEqual(first.topic_id, second.topic_id)
        self.assertEqual(second.created_at, first.created_at)
        self.assertGreater(second.updated_at, first.updated_at)
        self.assertEqual(len(topic_nodes.topics(self.store)), 1)
        rows = self.store._conn.execute("SELECT COUNT(*) FROM topic_members").fetchone()[0]
        self.assertEqual(rows, 8)

    def test_a_supplied_name_is_kept(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        topic = topic_nodes.create(self.store, members, name="Homelab containers", at=NOW)
        self.assertEqual(topic.name, "Homelab containers")
        again = topic_nodes.create(self.store, members, at=LATER)
        self.assertEqual(again.name, "Homelab containers")

    def test_a_supplied_name_can_be_corrected(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        topic_nodes.create(self.store, members, name="First guess", at=NOW)
        renamed = topic_nodes.create(self.store, members, name="Second guess", at=LATER)
        self.assertEqual(renamed.name, "Second guess")

    def test_a_member_that_does_not_exist_is_dropped_not_stored(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        topic = topic_nodes.create(self.store, members + ["ghost-memory"], at=NOW)
        self.assertEqual(topic.size, 8)
        self.assertNotIn("ghost-memory", topic.members)
        rows = self.store._conn.execute("SELECT COUNT(*) FROM topic_members").fetchone()[0]
        self.assertEqual(rows, 8)

    def test_no_real_members_means_no_topic(self) -> None:
        self.assertIsNone(topic_nodes.create(self.store, []))
        self.assertIsNone(topic_nodes.create(self.store, ["ghost-memory"], at=NOW))
        self.assertEqual(topic_nodes.topics(self.store), [])

    def test_creating_a_topic_does_not_touch_the_graph(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        before_edges = self.store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        before_memories = self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        topic_nodes.create(self.store, members, at=NOW)
        self.assertEqual(
            self.store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0], before_edges
        )
        self.assertEqual(
            self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            before_memories,
        )


class LookupTests(StoreCase):
    def test_topics_come_back_biggest_first(self) -> None:
        small = self.cluster(SHELF_WORDS, 3)
        large = self.cluster(CONTAINER_WORDS, 8)
        topic_nodes.create(self.store, small, name="Shelf", at=NOW)
        topic_nodes.create(self.store, large, name="Containers", at=NOW)
        names = [topic.name for topic in topic_nodes.topics(self.store)]
        self.assertEqual(names, ["Containers", "Shelf"])

    def test_topics_respect_their_limit(self) -> None:
        self.cluster(SHELF_WORDS, 3)
        self.cluster(CONTAINER_WORDS, 8)
        for members, name in (
            (topic_nodes.components(self.store)[1], "Shelf"),
            (topic_nodes.components(self.store)[0], "Containers"),
        ):
            topic_nodes.create(self.store, members, name=name, at=NOW)
        self.assertEqual(len(topic_nodes.topics(self.store, limit=1)), 1)

    def test_topic_of_finds_the_landmark_a_memory_belongs_to(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        topic = topic_nodes.create(self.store, members, at=NOW)
        found = topic_nodes.topic_of(self.store, members[3])
        self.assertEqual(found.topic_id, topic.topic_id)

    def test_topic_of_an_unanchored_memory_is_none(self) -> None:
        self.cluster(CONTAINER_WORDS, 8)
        island = self.memory("Nothing points here.")
        self.assertIsNone(topic_nodes.topic_of(self.store, island))

    def test_get_returns_none_for_an_unknown_id(self) -> None:
        self.assertIsNone(topic_nodes.get(self.store, "topic:000000000000"))


class ReportTests(StoreCase):
    def test_the_report_counts_topics_and_members(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        topic_nodes.create(self.store, members, at=NOW)
        described = topic_nodes.report(self.store)
        self.assertEqual(described["topics"], 1)
        self.assertEqual(described["members"], 8)
        self.assertEqual(described["largest"]["size"], 8)

    def test_the_report_names_what_is_still_uncovered(self) -> None:
        members = self.cluster(CONTAINER_WORDS, 8)
        other = self.cluster(SHELF_WORDS, 3)
        topic_nodes.create(self.store, members, at=NOW)
        described = topic_nodes.report(self.store)
        self.assertEqual(described["connected"], 11)
        self.assertEqual(described["uncovered"], len(other))

    def test_an_empty_store_reports_zeroes(self) -> None:
        described = topic_nodes.report(self.store)
        self.assertEqual(described["topics"], 0)
        self.assertEqual(described["members"], 0)
        self.assertIsNone(described["largest"])
        self.assertEqual(described["uncovered"], 0)


class FailureTests(unittest.TestCase):
    """A store error is an empty answer. A caller must never have to catch for that."""

    def test_components_are_empty(self) -> None:
        self.assertEqual(topic_nodes.components(Broken()), [])

    def test_regions_are_empty(self) -> None:
        self.assertEqual(topic_nodes.dense_regions(Broken()), [])

    def test_a_label_is_empty(self) -> None:
        self.assertEqual(topic_nodes.label(Broken(), ["anything"]), "")

    def test_a_topic_is_not_created(self) -> None:
        self.assertIsNone(topic_nodes.create(Broken(), ["anything"]))

    def test_lookups_are_empty(self) -> None:
        self.assertEqual(topic_nodes.topics(Broken()), [])
        self.assertIsNone(topic_nodes.topic_of(Broken(), "anything"))

    def test_the_report_is_zeroed(self) -> None:
        self.assertEqual(topic_nodes.report(Broken())["topics"], 0)


if __name__ == "__main__":
    unittest.main()
