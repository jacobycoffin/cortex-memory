"""Redundancy collapse: cluster near-duplicates and plan one copy (item #11)."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.redundancy import Memory, canonical, cluster, jaccard, memory_from_row, plan, tokens

FACT_A = (
    "Jacoby runs Cortex on the r630 at 192.168.1.50 and the store lives at "
    "~/.local/share/cortex/cortex.db with nightly backups."
)
FACT_A_VARIANT = (
    "Cortex runs on the r630 host 192.168.1.50, its store is "
    "~/.local/share/cortex/cortex.db, and nightly backups are taken."
)
FACT_B = "The maintenance window is Sunday 02:00 and lasts about an hour on CT116."


SAME_NOTE = "vault:plans/r630.md#host"


def mem(memory_id: str, content: str, **kwargs) -> Memory:
    kwargs.setdefault("source_ref", SAME_NOTE)
    return Memory(id=memory_id, content=content, **kwargs)


class TokenTests(unittest.TestCase):
    def test_vault_scaffolding_is_ignored(self) -> None:
        self.assertEqual(tokens("Vault note: Path: section: router"), {"router"})

    def test_similar_restatements_score_high(self) -> None:
        self.assertGreater(jaccard(tokens(FACT_A), tokens(FACT_A_VARIANT)), 0.5)

    def test_unrelated_facts_score_low(self) -> None:
        self.assertLess(jaccard(tokens(FACT_A), tokens(FACT_B)), 0.2)


class ClusterTests(unittest.TestCase):
    def test_near_duplicates_form_one_cluster(self) -> None:
        groups = cluster([mem("a", FACT_A), mem("b", FACT_A_VARIANT), mem("c", FACT_B)])
        self.assertEqual(len(groups), 1)
        self.assertEqual({m.id for m in groups[0]}, {"a", "b"})

    def test_near_verbatim_copies_always_merge(self) -> None:
        self.assertEqual(len(cluster([mem("a", FACT_A), mem("b", FACT_A)])), 1)

    def test_two_different_notes_are_not_merged_even_when_phrased_alike(self) -> None:
        """A TeslaMate note is not a duplicate of a Docker note because both say 'vault note'."""
        groups = cluster(
            [
                mem("a", FACT_A, source_ref="vault:homelab/Docker.md#docker"),
                mem("b", FACT_A_VARIANT, source_ref="vault:homelab/TeslaMate.md#tesla"),
            ]
        )
        self.assertEqual(groups, [])

    def test_a_partial_overlap_is_not_a_duplicate(self) -> None:
        shared = "The maintenance window is Sunday 02:00 and the store lives at srv/cortex"
        groups = cluster([mem("a", FACT_A), mem("b", shared + " and nothing else matters here")])
        self.assertEqual(groups, [])

    def test_empty_content_is_never_clustered(self) -> None:
        self.assertEqual(cluster([mem("a", ""), mem("b", "")]), [])

    def test_two_pinned_memories_are_not_merged(self) -> None:
        groups = cluster(
            [mem("a", FACT_A, pinned=True), mem("b", FACT_A_VARIANT, pinned=True)]
        )
        self.assertEqual(groups, [])

    def test_a_pinned_memory_absorbs_an_unpinned_duplicate(self) -> None:
        groups = cluster([mem("a", FACT_A, pinned=True), mem("b", FACT_A_VARIANT)])
        self.assertEqual(len(groups), 1)


class CanonicalTests(unittest.TestCase):
    def test_most_used_copy_is_kept(self) -> None:
        group = [mem("a", FACT_A, used_count=0), mem("b", FACT_A_VARIANT, used_count=7)]
        self.assertEqual(canonical(group).id, "b")

    def test_importance_breaks_a_use_tie(self) -> None:
        group = [mem("a", FACT_A, importance=0.2), mem("b", FACT_A_VARIANT, importance=0.9)]
        self.assertEqual(canonical(group).id, "b")

    def test_oldest_breaks_a_remaining_tie(self) -> None:
        group = [
            mem("a", FACT_A, created_at="2026-11-01"),
            mem("b", FACT_A_VARIANT, created_at="2026-05-01"),
        ]
        self.assertEqual(canonical(group).id, "b")

    def test_choice_is_deterministic(self) -> None:
        group = [mem("b", FACT_A_VARIANT), mem("a", FACT_A)]
        self.assertEqual(canonical(group).id, canonical(list(reversed(group))).id)


class PlanTests(unittest.TestCase):
    def test_plan_names_the_keeper_and_the_casualties(self) -> None:
        plans = plan([mem("a", FACT_A, used_count=4), mem("b", FACT_A_VARIANT), mem("c", FACT_B)])
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].keep.id, "a")
        self.assertEqual([m.id for m in plans[0].drop], ["b"])
        self.assertGreater(plans[0].tokens_saved, 0)

    def test_plans_are_ordered_by_savings(self) -> None:
        long_text = FACT_A + " " + FACT_A
        plans = plan(
            [
                mem("a", FACT_A, used_count=1),
                mem("b", FACT_A_VARIANT),
                mem("c", long_text, used_count=1),
                mem("d", long_text + " extra words here"),
            ],
        )
        savings = [p.tokens_saved for p in plans]
        self.assertEqual(savings, sorted(savings, reverse=True))

    def test_row_conversion_keeps_the_fields_that_matter(self) -> None:
        memory = memory_from_row(
            {"id": "x", "content": FACT_A, "importance": 0.5, "used_count": 3, "pinned": 1}
        )
        self.assertEqual((memory.id, memory.used_count, memory.pinned), ("x", 3, True))


if __name__ == "__main__":
    unittest.main()
