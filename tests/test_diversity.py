"""Diversity: one slot per fact, not five (#10)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.diversity import DEFAULT_CEILING, enforce_distinct, group_key, jaccard, token_set


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


SEERR = "Vault note: Seerr Telegram Notifications Path: homelab/Seerr Section: notifications"
SEERR_TWEAK = "Vault note: Seerr Telegram Notifications Path: homelab/Seerr Section: notifications setup"
OTHER = "Jacoby prefers Telegram progress updates as a single edited message, not a stream."


def make(content: str, score: float) -> Fake:
    return Fake({"id": content[:8], "content": content}, score)


class SimilarityTests(unittest.TestCase):
    def test_identical_content_is_one(self) -> None:
        self.assertEqual(jaccard(token_set(make(SEERR, 1.0)), token_set(make(SEERR, 1.0))), 1.0)

    def test_different_facts_are_low(self) -> None:
        a, b = token_set(make(SEERR, 1.0)), token_set(make(OTHER, 1.0))
        self.assertLess(jaccard(a, b), 0.35)

    def test_empty_content_is_zero(self) -> None:
        self.assertEqual(jaccard(frozenset(), token_set(make("x", 1.0))), 0.0)


def note(path: str, section: str, score: float) -> Fake:
    """Distinct prose that shares only a source path — the real Seerr case."""
    # No nested dict inside the f-string: PEP 701 syntax breaks Python 3.10/3.11.
    phrase = {
        "a": "notification routing and webhook setup",
        "b": "the api key rotation schedule",
        "c": "disk usage warnings from the container",
        "d": "the reverse proxy header rules",
    }.get(section, "misc")
    content = (
        f"Vault note: {section} Path: {path} Section: {section}. "
        f"Unique body text for {section} covering {phrase}."
    )
    return Fake({"id": section, "content": content, "source_ref": path}, score)


class GroupCapTests(unittest.TestCase):
    def test_a_section_anchor_does_not_create_a_new_group(self) -> None:
        a = Fake({"id": "1", "content": "x", "source_ref": "vault:a.md#one"}, 0.5)
        b = Fake({"id": "2", "content": "y", "source_ref": "vault:a.md#two"}, 0.4)
        self.assertEqual(group_key(a), group_key(b))
    """Distinct notes about the same vault path are still one topic eating the block."""

    def test_a_path_contributes_at_most_two_slots(self) -> None:
        selected = [
            note("homelab/Seerr", "a", 0.72),
            note("homelab/Seerr", "b", 0.71),
            note("homelab/Seerr", "c", 0.70),
            note("homelab/Seerr", "d", 0.69),
        ]
        kept, replaced = enforce_distinct(selected, [], limit=4, ceiling=1.0)
        self.assertEqual(replaced, 2)
        self.assertEqual(len(kept), 2)

    def test_lower_scoring_distinct_paths_win_those_slots(self) -> None:
        selected = [
            note("homelab/Seerr", "a", 0.72),
            note("homelab/Seerr", "b", 0.71),
            note("homelab/Seerr", "c", 0.70),
        ]
        spare = [Fake({"id": "x", "content": "Unrelated fact about trucks and trailers.", "source_ref": "homelab/trucks"}, 0.40)]
        kept, replaced = enforce_distinct(selected, spare, limit=3, ceiling=1.0)
        self.assertEqual(replaced, 1)
        self.assertIn("homelab/trucks", [r.memory.get("source_ref") for r in kept])

    def test_different_paths_are_untouched(self) -> None:
        selected = [note("a/one", "a", 0.7), note("b/two", "b", 0.6), note("c/three", "c", 0.5)]
        kept, replaced = enforce_distinct(selected, [], limit=3)
        self.assertEqual((replaced, len(kept)), (0, 3))

    def test_a_literal_memory_source_is_not_a_group(self) -> None:
        self.assertIsNone(group_key(Fake({"id": "1", "content": "x", "source_ref": "memory"}, 0.5)))

    def test_memories_without_a_group_are_not_capped(self) -> None:
        selected = [
            Fake({"id": "1", "content": "Short distinct fact one."}, 0.7),
            Fake({"id": "2", "content": "Other unrelated fact two."}, 0.6),
        ]
        kept, replaced = enforce_distinct(selected, [], limit=2)
        self.assertEqual((replaced, len(kept)), (0, 2))


class EnforceTests(unittest.TestCase):
    def test_near_duplicates_are_replaced_by_a_distinct_memory(self) -> None:
        selected = [
            make(SEERR, 0.72),
            make(SEERR_TWEAK, 0.70),
            make(SEERR + " extra note", 0.69),
        ]
        spare = [make(OTHER, 0.45)]
        # 0.85 does not catch a note that dropped a word; 0.75 does. The ceiling is a
        # parameter precisely because "the same fact" depends on the corpus.
        kept, replaced = enforce_distinct(selected, spare, limit=3, ceiling=0.75)
        self.assertEqual(replaced, 2)
        self.assertEqual(len(kept), 2)  # SEERR + the distinct spare
        self.assertIn(OTHER, [r.memory["content"] for r in kept])

    def test_a_duplicate_is_never_readded_to_fill_a_slot(self) -> None:
        """A shorter block of distinct facts beats a longer one that repeats itself."""
        selected = [make(SEERR, 0.72), make(SEERR_TWEAK, 0.70)]
        kept, replaced = enforce_distinct(selected, [], limit=2)
        self.assertEqual(replaced, 1)
        self.assertEqual(len(kept), 1)

    def test_distinct_memories_are_untouched(self) -> None:
        selected = [make(SEERR, 0.7), make(OTHER, 0.5)]
        kept, replaced = enforce_distinct(selected, [], limit=2)
        self.assertEqual((replaced, len(kept)), (0, 2))
        self.assertEqual([r.score for r in kept], [0.7, 0.5])

    def test_limit_is_respected(self) -> None:
        selected = [make(SEERR, 0.7), make(OTHER, 0.6), make("Another unrelated fact about trucks.", 0.5)]
        kept, _ = enforce_distinct(selected, [], limit=2)
        self.assertEqual(len(kept), 2)

    def test_ceiling_one_disables_the_rule(self) -> None:
        selected = [make(SEERR, 0.7), make(SEERR, 0.69)]
        kept, replaced = enforce_distinct(selected, [], limit=2, ceiling=1.0)
        self.assertEqual((replaced, len(kept)), (0, 2))

    def test_zero_limit_returns_nothing(self) -> None:
        self.assertEqual(enforce_distinct([make(SEERR, 0.7)], [], limit=0), ([], 0))

    def test_custom_similarity_function_is_used(self) -> None:
        a, b = make("alpha", 0.7), make("beta", 0.6)
        kept, replaced = enforce_distinct([a, b], [], limit=2, similarity=lambda _x, _y: 0.99)
        self.assertEqual(replaced, 1)
        self.assertEqual(len(kept), 1)

    def test_default_ceiling_is_strict_enough_to_catch_near_copies(self) -> None:
        self.assertLess(DEFAULT_CEILING, 0.9)
        a, b = token_set(make(SEERR, 1.0)), token_set(make(SEERR_TWEAK, 1.0))
        self.assertGreaterEqual(jaccard(a, b), DEFAULT_CEILING)


if __name__ == "__main__":
    unittest.main()
