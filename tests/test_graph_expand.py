"""Cluster-then-expand retrieval (#160): the rules, and the cases where it must decline."""

from __future__ import annotations

import string
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import graph_expand
from cortex.store import CortexStore

HOMELAB = ("proxmox", "nginx", "container", "compose")
WOODWORK = ("walnut", "bookshelf", "shelving")
BAKING = ("sourdough", "starter", "flour")


class Broken:
    """A store whose reads fail. A retrieval aid that raises takes the turn down with it."""

    _conn = None

    def transaction(self):  # pragma: no cover - expansion never writes
        raise RuntimeError("database is locked")


class GraphExpandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    # -- fixtures ---------------------------------------------------------------
    def memory(self, content: str, *, entities: list[str] | None = None,
               state: str = "active") -> str:
        return self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_STATED",
            entities=entities,
            state=state,
        )[0]

    def link(self, left: str, right: str, weight: float = 0.6) -> None:
        self.store.add_edge(left, right, "related", weight=weight)

    def cluster(self, shared: tuple[str, ...], prefix: str, count: int = 2,
                *, state: str = "active") -> list[str]:
        """`count` memories sharing `shared` terms and one unique word each.

        Density for such a group is len(shared) / (len(shared) + 2): two members share every term but
        their own filler, so the overlap is exactly knowable instead of approximate.
        """
        made = [
            self.memory(" ".join(shared) + f" {prefix}word{string.ascii_lowercase[index]}",
                        state=state)
            for index in range(count)
        ]
        for left, right in zip(made, made[1:]):
            self.link(left, right)
        return made

    def fan_out(self, seed: str, ids: list[str]) -> None:
        for memory_id in ids:
            self.link(seed, memory_id)

    def noise(self, seed: str, count: int = 2) -> list[str]:
        """Unrelated memories, one cluster each, so the pool has something to be denser *than*.

        Each carries its own two words: sharing a word with its neighbour would make the "noise" a
        rival cluster, which is exactly the mistake the module is supposed to refuse to paper over.
        """
        words = ("lighthouse", "trumpet", "bicycle", "harbour", "clarinet", "workbench")
        made = [
            self.memory(f"{words[index]}s {words[index]}ward {index}")
            for index in range(count)
        ]
        self.fan_out(seed, made)
        return made

    # -- rules ------------------------------------------------------------------
    def test_empty_seed_expands_to_nothing(self) -> None:
        found = graph_expand.expand(self.store, [])
        self.assertFalse(found.expanded)
        self.assertEqual(found.ids, [])
        self.assertIn("empty", found.reason)

    def test_a_single_memory_has_nothing_to_cluster_with(self) -> None:
        only = self.memory("proxmox nginx container compose alpha")
        found = graph_expand.expand(self.store, [only])
        self.assertFalse(found.expanded)
        self.assertEqual(found.ids, [only])
        self.assertIn("fewer than two", found.reason)

    def test_a_dense_cluster_is_expanded(self) -> None:
        group = self.cluster(HOMELAB, "alpha", 4)
        noise = [self.memory(f"{word} {index}suffix") for index, word in enumerate(("bicycle", "lighthouse", "trumpet"))]
        self.fan_out(group[0], group[1:] + noise)
        found = graph_expand.expand(self.store, [group[0]])
        self.assertTrue(found.expanded)
        self.assertEqual(set(found.added), set(group[1:]))
        self.assertEqual(found.ids[0], group[0])

    def test_expansion_is_capped_and_the_cap_is_stated(self) -> None:
        group = self.cluster(HOMELAB, "bravo", 5)
        self.fan_out(group[0], group[1:])
        self.noise(group[0])
        found = graph_expand.expand(self.store, [group[0]], cap=2)
        self.assertEqual(len(found.added), 2)
        self.assertEqual(found.dropped, 2)
        self.assertEqual(found.cap, 2)
        self.assertIn("at most 2", found.describe())
        self.assertEqual(found.as_dict()["cap"], 2)

    def test_the_cap_cannot_be_exceeded_by_a_bigger_cluster(self) -> None:
        group = self.cluster(HOMELAB, "charlie", 8)
        self.fan_out(group[0], group[1:])
        self.noise(group[0])
        found = graph_expand.expand(self.store, [group[0]])
        self.assertLessEqual(len(found.added), graph_expand.CAP_PER_CLUSTER)
        self.assertEqual(len(found.added), graph_expand.CAP_PER_CLUSTER)

    def test_expansion_never_repeats_a_seed_id(self) -> None:
        group = self.cluster(HOMELAB, "delta", 4)
        self.fan_out(group[0], group[1:])
        self.noise(group[0])
        found = graph_expand.expand(self.store, [group[0]])
        self.assertTrue(found.expanded)
        self.assertFalse(set(found.added) & set(found.seed))
        self.assertEqual(len(found.ids), len(set(found.ids)))

    def test_the_answer_names_the_cluster_and_its_size(self) -> None:
        group = self.cluster(HOMELAB, "echo", 4)
        noise = [self.memory("lighthouse keeper log"), self.memory("trumpet mouthpiece brass")]
        self.fan_out(group[0], group[1:] + noise)
        found = graph_expand.expand(self.store, [group[0]])
        self.assertIsNotNone(found.cluster)
        self.assertEqual(found.cluster["size"], 4)
        self.assertEqual(found.cluster["already_seeded"], 1)
        self.assertTrue(found.cluster["name"])
        self.assertIn(found.cluster["name"], found.describe())
        self.assertIn("expanded", found.describe())
        self.assertEqual(found.cluster["name"], found.cluster["name"].strip())

    def test_the_threshold_is_computed_from_the_pool(self) -> None:
        # Shape one: the densest cluster competes with clusters that share nothing -> bar 0.0.
        group = self.cluster(HOMELAB, "foxtrot", 4)
        noise = [self.memory("lighthouse keeper log"), self.memory("trumpet mouthpiece brass")]
        self.fan_out(group[0], group[1:] + noise)
        loose = graph_expand.expand(self.store, [group[0]])

        # Shape two: competitors at 0.5 density -> bar 1.0, which the 0.6 best cannot clear.
        tight = tempfile.TemporaryDirectory()
        try:
            other = CortexStore(Path(tight.name) / "cortex.db")
            first = other.add_memory(content="proxmox nginx alpha", kind="semantic",
                                     source_type="user_turn", source_category="USER_STATED")[0]
            partner = other.add_memory(content="proxmox nginx bravo", kind="semantic",
                                       source_type="user_turn", source_category="USER_STATED")[0]
            left = other.add_memory(content="walnut shelving charlie", kind="semantic",
                                    source_type="user_turn", source_category="USER_STATED")[0]
            right = other.add_memory(content="walnut shelving delta", kind="semantic",
                                     source_type="user_turn", source_category="USER_STATED")[0]
            best = other.add_memory(content="sourdough starter flour echo", kind="semantic",
                                    source_type="user_turn", source_category="USER_STATED")[0]
            rest = other.add_memory(content="sourdough starter flour gulf", kind="semantic",
                                    source_type="user_turn", source_category="USER_STATED")[0]
            for src, dst in ((first, partner), (left, right), (best, rest)):
                other.add_edge(src, dst, "related", weight=0.6)
            for candidate in (first, partner, left, right, rest):
                other.add_edge(best, candidate, "related", weight=0.6)
            dense = graph_expand.expand(other, [best])
        finally:
            other.close()
            tight.cleanup()

        self.assertEqual(loose.threshold, 0.0)
        self.assertEqual(dense.threshold, 1.0)
        for found in (loose, dense):
            self.assertEqual(
                found.threshold,
                round(found.baseline + max(found.spread, found.baseline), 6),
            )
            self.assertIsNotNone(found.baseline)

    def test_all_one_cluster_does_not_expand(self) -> None:
        group = self.cluster(HOMELAB, "golf", 3)
        self.fan_out(group[0], group[1:])
        found = graph_expand.expand(self.store, [group[0]])
        self.assertFalse(found.expanded)
        self.assertIn("one cluster", found.reason)
        self.assertEqual(found.ids, [group[0]])

    def test_a_tie_between_clusters_does_not_expand(self) -> None:
        first = self.cluster(("proxmox", "nginx", "container", "compose"), "hotel", 2)
        second = self.cluster(("walnut", "shelving", "varnish", "dovetail"), "india", 2)
        third = self.cluster(("sourdough", "starter", "flour", "levain"), "juliet", 2)
        self.fan_out(first[0], [first[1], second[0], second[1], third[0], third[1]])
        found = graph_expand.expand(self.store, [first[0]])
        self.assertFalse(found.expanded)
        self.assertIn("equally dense", found.reason)

    def test_no_standout_cluster_keeps_the_seed(self) -> None:
        weak = self.cluster(("proxmox", "nginx"), "kilo", 2)
        pair = self.cluster(("walnut", "shelving"), "lima", 2)
        strong = self.cluster(("sourdough", "starter", "flour"), "mike", 2)
        self.fan_out(strong[0], [strong[1], weak[0], weak[1], pair[0], pair[1]])
        found = graph_expand.expand(self.store, [strong[0]])
        self.assertFalse(found.expanded)
        self.assertEqual(found.ids, [strong[0]])
        self.assertIn("denser than the rest", found.reason)
        self.assertEqual(found.cluster, None)

    def test_a_broken_store_returns_the_seed_and_does_not_raise(self) -> None:
        found = graph_expand.expand(Broken(), ["aaaaaaaa", "bbbbbbbb"])
        self.assertFalse(found.expanded)
        self.assertEqual(found.ids, ["aaaaaaaa", "bbbbbbbb"])
        self.assertTrue(found.reason)

    def test_expansion_is_deterministic(self) -> None:
        group = self.cluster(HOMELAB, "november", 4)
        self.fan_out(group[0], group[1:])
        first = graph_expand.expand(self.store, [group[0]])
        second = graph_expand.expand(self.store, [group[0]])
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_archived_neighbours_are_not_added(self) -> None:
        seed = self.memory("proxmox nginx container compose zulu")
        archived = self.cluster(HOMELAB, "oscar", 2, state="archived")
        noise = [self.memory("lighthouse keeper log"), self.memory("trumpet mouthpiece brass")]
        self.fan_out(seed, archived + noise)
        found = graph_expand.expand(self.store, [seed])
        self.assertFalse(found.expanded)
        self.assertEqual(found.added, [])
        self.assertIsNotNone(found.cluster)
        self.assertEqual(found.cluster["addable"], 0)
        self.assertIn("no addable", found.reason)

    def test_duplicate_seed_ids_collapse(self) -> None:
        group = self.cluster(HOMELAB, "papa", 3)
        self.fan_out(group[0], group[1:])
        found = graph_expand.expand(self.store, [group[0], group[0]])
        self.assertEqual(found.seed, [group[0]])
        self.assertEqual(len(found.ids), len(set(found.ids)))

    def test_a_zero_cap_adds_nothing(self) -> None:
        group = self.cluster(HOMELAB, "quebec", 4)
        self.fan_out(group[0], group[1:])
        self.noise(group[0])
        found = graph_expand.expand(self.store, [group[0]], cap=0)
        self.assertFalse(found.expanded)
        self.assertEqual(found.added, [])
        self.assertIn("cap is 0", found.reason)

    def test_a_seed_id_the_store_does_not_have_is_not_fatal(self) -> None:
        group = self.cluster(HOMELAB, "romeo", 3)
        self.fan_out(group[0], group[1:])
        found = graph_expand.expand(self.store, ["not-a-memory", group[0]])
        self.assertEqual(found.seed[0], "not-a-memory")
        for memory_id in found.added:
            self.assertNotEqual(memory_id, "not-a-memory")

    def test_entities_join_memories_that_share_no_content_words(self) -> None:
        first = self.memory("the backup ran at four", entities=["Proxmox"])
        second = self.memory("the box rebooted cleanly", entities=["Proxmox"])
        self.link(first, second)
        shared = graph_expand.terms_of("", ["Proxmox"]) & graph_expand.terms_of("", ["Proxmox"])
        self.assertIn("proxmox", shared)
        self.assertIn("proxmox", graph_expand.terms_of("", ["Proxmox"]))

    def test_vault_scaffolding_is_not_a_cluster_name(self) -> None:
        first = self.memory(
            "Vault note: Proxmox box Path: work/box.md Section: Setup proxmox nginx container alpha"
        )
        second = self.memory(
            "Vault note: Proxmox box Path: work/box.md Section: Setup proxmox nginx container bravo"
        )
        self.link(first, second)
        found = graph_expand.expand(self.store, [first])
        # One cluster only, so it declines - but the cluster it reported must not be named after the
        # importer's header.
        terms = graph_expand.terms_of(
            "Vault note: Proxmox box Path: work/box.md Section: Setup proxmox nginx container alpha"
        )
        self.assertNotIn("vault", terms)
        self.assertNotIn("section", terms)
        self.assertIn("proxmox", terms)
        self.assertFalse(found.expanded)

    def test_render_lists_what_was_added(self) -> None:
        group = self.cluster(HOMELAB, "sierra", 3)
        self.fan_out(group[0], group[1:])
        self.noise(group[0])
        found = graph_expand.expand(self.store, [group[0]])
        lines = graph_expand.render(found)
        self.assertIn("expanded", lines[0])
        self.assertEqual(sum(1 for line in lines if line.startswith("  + ")), len(found.added))

    def test_describe_says_so_when_nothing_was_added(self) -> None:
        found = graph_expand.Expansion(seed=["a"])
        self.assertIn("kept as-is", graph_expand.expand(self.store, ["a"]).describe())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
