"""A one-page profile of a person, assembled from the facts that mention them (#61)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import person_profiles
from cortex.store import CortexStore


class StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, kind: str = "semantic", entities: Any = None,
               created_at: str = "", state: str = "active") -> str:
        memory_id = self.store.add_memory(
            content=content, kind=kind, source_type="user_turn",
            source_category="USER_EXPLICIT", entities=entities, state=state,
        )[0]
        if created_at:
            with self.store.transaction() as conn:
                conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (created_at, memory_id))
        return memory_id

    def node(self, name: str, kind: str = "person") -> None:
        """Stand in for the entity layer's table; the profile reads it, never writes it."""
        with self.store.transaction() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS entity_nodes (
                       entity_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL,
                       kind TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
                       created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
            )
            conn.execute(
                "INSERT INTO entity_nodes VALUES (?, ?, ?, '', '2026-11-20', '2026-11-20')",
                (name, name, kind),
            )


class PeopleTests(StoreCase):
    def test_a_person_with_facts_is_listed(self) -> None:
        self.memory("Jacoby prefers terse answers.", kind="preference", entities=["Jacoby"])
        self.assertIn("Jacoby", person_profiles.people(self.store))

    def test_an_entity_that_is_not_a_person_is_left_out(self) -> None:
        self.memory("CT116 runs the nightly build.", entities=["CT116", "cortex"])
        self.memory("Jacoby owns the box.", entities=["Jacoby"])
        self.assertEqual(person_profiles.people(self.store), ["Jacoby"])

    def test_a_kind_corrected_to_person_is_listed(self) -> None:
        """The lexicon cannot know every name; a human correction on the node can."""
        self.node("Maria", "person")
        self.memory("Maria runs the design review.", entities=["Maria"])
        self.assertIn("Maria", person_profiles.people(self.store))


class FactsAboutTests(StoreCase):
    def test_facts_come_back_newest_first(self) -> None:
        self.memory("First thing about Jacoby.", entities=["Jacoby"],
                    created_at="2026-11-01T09:00:00+00:00")
        self.memory("Second thing about Jacoby.", entities=["Jacoby"],
                    created_at="2026-11-05T09:00:00+00:00")
        contents = [fact["content"] for fact in person_profiles.facts_about(self.store, "Jacoby")]
        self.assertEqual(contents, ["Second thing about Jacoby.", "First thing about Jacoby."])

    def test_only_facts_that_name_the_person(self) -> None:
        self.memory("Jacoby likes the long form.", entities=["Jacoby"])
        self.memory("Kaya likes the short form.", entities=["Kaya"])
        contents = [fact["content"] for fact in person_profiles.facts_about(self.store, "Jacoby")]
        self.assertEqual(contents, ["Jacoby likes the long form."])

    def test_a_name_matches_regardless_of_case(self) -> None:
        self.memory("jacoby works late.", entities=["jacoby"])
        self.assertEqual(len(person_profiles.facts_about(self.store, "Jacoby")), 1)

    def test_superseded_facts_are_not_counted_twice(self) -> None:
        self.memory("Jacoby lives in Austin.", entities=["Jacoby"])
        self.memory("Jacoby lives in Denver.", entities=["Jacoby"], state="superseded")
        contents = [fact["content"] for fact in person_profiles.facts_about(self.store, "Jacoby")]
        self.assertEqual(contents, ["Jacoby lives in Austin."])

    def test_the_limit_is_respected(self) -> None:
        for index in range(5):
            self.memory(f"Jacoby fact {index}.", entities=["Jacoby"],
                        created_at=f"2026-11-0{index + 1}T09:00:00+00:00")
        self.assertEqual(len(person_profiles.facts_about(self.store, "Jacoby", limit=2)), 2)


class ProfileTests(StoreCase):
    def setUp(self) -> None:
        super().setUp()
        self.memory("Jacoby prefers dark mode.", kind="preference", entities=["Jacoby"],
                    created_at="2026-11-01T09:00:00+00:00")
        self.memory("Jacoby plans to ship the profile view.", kind="prospective",
                    entities=["Jacoby"], created_at="2026-11-03T09:00:00+00:00")
        self.memory("Jacoby works from Knoxville.", kind="identity", entities=["Jacoby"],
                    created_at="2026-11-02T09:00:00+00:00")

    def test_the_profile_has_the_promised_shape(self) -> None:
        described = person_profiles.profile(self.store, "Jacoby")
        for key in ("name", "kind", "fact_count", "first_seen", "last_seen", "facts",
                    "by_kind", "open_questions", "sentence"):
            self.assertIn(key, described)
        self.assertEqual(described["name"], "Jacoby")
        self.assertEqual(described["kind"], "person")
        self.assertEqual(described["first_seen"][:10], "2026-11-01")
        self.assertEqual(described["last_seen"][:10], "2026-11-03")

    def test_fact_count_covers_every_fact_not_just_the_page(self) -> None:
        described = person_profiles.profile(self.store, "Jacoby", limit=1)
        self.assertEqual(described["fact_count"], 3)
        self.assertEqual(len(described["facts"]), 1)

    def test_by_kind_counts_the_kinds(self) -> None:
        described = person_profiles.profile(self.store, "Jacoby")
        self.assertEqual(described["by_kind"]["identity"], 1)
        self.assertEqual(sum(described["by_kind"].values()), 3)

    def test_open_questions_are_the_prospective_facts(self) -> None:
        described = person_profiles.profile(self.store, "Jacoby")
        self.assertEqual([fact["content"] for fact in described["open_questions"]],
                         ["Jacoby plans to ship the profile view."])

    def test_the_sentence_names_the_person_and_the_count(self) -> None:
        sentence = person_profiles.profile(self.store, "Jacoby")["sentence"]
        self.assertIn("Jacoby", sentence)
        self.assertIn("3 facts", sentence)
        self.assertNotIn("\n", sentence)

    def test_a_person_with_no_facts_has_no_profile(self) -> None:
        self.assertIsNone(person_profiles.profile(self.store, "Nobody"))
        self.assertIsNone(person_profiles.profile(self.store, ""))


class RenderTests(StoreCase):
    def test_markdown_starts_with_a_heading_and_lists_bullets(self) -> None:
        self.memory("Jacoby prefers dark mode.", kind="preference", entities=["Jacoby"])
        block = person_profiles.render(self.store, "Jacoby")
        lines = block.splitlines()
        self.assertEqual(lines[0], "## Jacoby")
        self.assertTrue(any(line.startswith("- ") for line in lines))
        self.assertIn("seen first", lines[1])

    def test_an_unknown_person_renders_nothing(self) -> None:
        self.assertEqual(person_profiles.render(self.store, "Nobody"), "")

    def test_the_block_is_capped(self) -> None:
        for index in range(40):
            self.memory(f"Jacoby fact number {index}.", entities=["Jacoby"],
                        created_at=f"2026-11-01T09:{index:02d}:00+00:00")
        block = person_profiles.render(self.store, "Jacoby")
        self.assertLessEqual(len(block.splitlines()), person_profiles.MAX_LINES)
        self.assertIn("more", block.splitlines()[-1])

    def test_style_text_has_no_markdown_heading(self) -> None:
        self.memory("Jacoby prefers dark mode.", entities=["Jacoby"])
        block = person_profiles.render(self.store, "Jacoby", style="text")
        self.assertFalse(block.startswith("## "))
        self.assertTrue(block.startswith("Jacoby"))


class IndexTests(StoreCase):
    def test_a_small_index_still_finds_people_among_many_entities(self) -> None:
        """Asking for few profiles must not mean looking at few names.

        The first version widened the scan to `limit * 5`, so `index(limit=8)` saw 40 names, found
        no person among them and returned an empty list while `people()` returned three — the live
        store reproduced exactly that.
        """
        for name in ("CT116", "CT117", "r630", "nginx", "proxmox", "redis", "docker", "sqlite",
                     "tailscale", "cloudflare", "immich", "jellyfin", "sonarr", "radarr",
                     "homebox", "grafana", "prometheus", "vaultwarden", "paperless", "gitea"):
            for index in range(3):
                self.memory(f"{name} note {index}.", entities=[name])
        self.memory("Jacoby prefers dark mode.", entities=["Jacoby"])
        self.memory("Kaya writes the briefs.", entities=["Kaya"])
        rows = person_profiles.index(self.store, limit=2)
        self.assertEqual([row["name"] for row in rows], ["Jacoby", "Kaya"])

    def test_the_index_is_sorted_by_fact_count_and_skips_non_people(self) -> None:
        for index in range(2):
            self.memory(f"Kaya fact {index}.", entities=["Kaya"])
        self.memory("Jacoby prefers dark mode.", entities=["Jacoby"])
        self.memory("CT116 runs the nightly build.", entities=["CT116"])
        rows = person_profiles.index(self.store)
        self.assertEqual([row["name"] for row in rows], ["Kaya", "Jacoby"])
        self.assertEqual(rows[0]["fact_count"], 2)
        self.assertTrue(rows[0]["last_seen"])


class FailureTests(unittest.TestCase):
    def test_a_broken_store_yields_an_empty_profile_and_no_people(self) -> None:
        """The profile is a view: a view that raises would take the caller down with it."""
        described = person_profiles.profile(object(), "Jacoby")
        self.assertEqual(described["fact_count"], 0)
        self.assertEqual(described["facts"], [])
        self.assertEqual(person_profiles.people(object()), [])
        self.assertEqual(person_profiles.index(object()), [])
        self.assertEqual(person_profiles.render(object(), "Jacoby"), "")


if __name__ == "__main__":
    unittest.main()
