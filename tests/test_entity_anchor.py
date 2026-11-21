"""Entity anchoring: the names in the query are the strongest signal the store has (item #8)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.entity_anchor import (
    MAX_ENTITIES,
    entity_memory_ids,
    extract_entities,
    known_entities,
    parse_entities,
    usable,
)
from cortex.store import CortexStore

NAMES = ["Mack", "Mack › Access architecture", "Hermes Gateway", "Cortex Brain Mechanics", "Cortex"]


class ExtractTests(unittest.TestCase):
    def test_a_named_entity_is_found(self) -> None:
        self.assertIn("Mack", extract_entities("how is Mack's access set up?", NAMES))

    def test_the_longest_name_wins(self) -> None:
        found = extract_entities("cortex brain mechanics design doc", NAMES)
        self.assertEqual(found[0], "Cortex Brain Mechanics")
        self.assertNotIn("Cortex", found)  # covered by the longer match

    def test_heading_entities_are_ignored(self) -> None:
        self.assertNotIn(
            "Mack › Access architecture", extract_entities("mack access architecture", NAMES)
        )

    def test_no_entity_in_the_query_returns_nothing(self) -> None:
        self.assertEqual(extract_entities("what's the weather today", NAMES), [])

    def test_case_and_punctuation_do_not_matter(self) -> None:
        self.assertIn("Hermes Gateway", extract_entities("is the HERMES GATEWAY up?", NAMES))

    def test_matching_is_whole_name_only(self) -> None:
        self.assertEqual(extract_entities("mackerel fishing", ["Mack"]), [])

    def test_the_number_of_entities_is_capped(self) -> None:
        many = [f"Entity{index}" for index in range(20)]
        query = " ".join(many)
        self.assertLessEqual(len(extract_entities(query, many)), MAX_ENTITIES)

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(extract_entities("", NAMES), [])

    def test_dates_and_numbers_are_not_entities(self) -> None:
        self.assertFalse(usable("2026-09-15"))
        self.assertFalse(usable("42"))
        self.assertTrue(usable("Mack"))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.ids = []
        for content, entities in (
            ("Mack's access uses a shared vault.", ["Mack"]),
            ("The gateway sits behind Caddy.", ["Hermes Gateway"]),
            ("Nothing to do with names.", []),
        ):
            created = self.store.add_memory(
                content=content,
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                entities=entities,
            )
            self.ids.append(created[0] if isinstance(created, tuple) else created)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_known_entities_are_listed_longest_first(self) -> None:
        names = known_entities(self.store)
        self.assertIn("Mack", names)
        self.assertIn("Hermes Gateway", names)

    def test_entities_expand_to_their_memories(self) -> None:
        ids = entity_memory_ids(self.store, ["Mack"])
        self.assertEqual(ids, [self.ids[0]])

    def test_an_unknown_entity_expands_to_nothing(self) -> None:
        self.assertEqual(entity_memory_ids(self.store, ["Nobody"]), [])

    def test_no_entities_is_a_no_op(self) -> None:
        self.assertEqual(entity_memory_ids(self.store, []), [])

    def test_parse_entities_accepts_json_and_lists(self) -> None:
        self.assertEqual(parse_entities('["Mack"]'), ["Mack"])
        self.assertEqual(parse_entities(["Mack"]), ["Mack"])
        self.assertEqual(parse_entities("{bad json"), [])

    def test_a_store_without_a_connection_is_safe(self) -> None:
        class Broken:
            pass

        self.assertEqual(known_entities(Broken()), [])
        self.assertEqual(entity_memory_ids(Broken(), ["Mack"]), [])


if __name__ == "__main__":
    unittest.main()
