"""A document must not quietly outrank what the user said (backlog item #196)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_authority import AUTHORITY, CAP_RATIO, authority_for, cap_conflicts  # noqa: E402


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


def mem(mid: str, category: str, subject: str = "vehicle", predicate: str = "max_tow") -> dict:
    return {
        "id": mid,
        "source_category": category,
        "subject": subject,
        "predicate": predicate,
        "content": f"{subject} {predicate} spoken by {category}",
    }


class AuthorityTests(unittest.TestCase):
    def test_user_outranks_every_document_category(self) -> None:
        self.assertEqual(authority_for(mem("a", "USER_EXPLICIT")), 1.0)
        self.assertGreater(
            authority_for(mem("a", "USER_EXPLICIT")),
            authority_for(mem("b", "DOCUMENT_EXTRACTED")),
        )
        # every declared category keeps its ordering against a scraped document
        for category in AUTHORITY:
            if category == "DOCUMENT_EXTRACTED":
                continue
            self.assertGreaterEqual(
                authority_for(mem("x", category)),
                authority_for(mem("y", "DOCUMENT_EXTRACTED")),
                category,
            )

    def test_source_type_fallback(self) -> None:
        self.assertEqual(authority_for({"source_type": "user_turn"}), 1.0)
        self.assertEqual(authority_for({"source_type": "vault_markdown"}), AUTHORITY["DOCUMENT_EXTRACTED"])
        self.assertEqual(authority_for({}), 0.5)

    def test_document_is_capped_below_the_user_statement(self) -> None:
        user = Fake(mem("user", "USER_EXPLICIT"), 0.60)
        doc = Fake(mem("doc", "DOCUMENT_EXTRACTED"), 0.95)
        results, capped = cap_conflicts([user, doc])
        self.assertEqual(capped, 1)
        by_id = {r.memory["id"]: r for r in results}
        self.assertEqual(by_id["user"].score, 0.60)
        self.assertAlmostEqual(by_id["doc"].score, 0.60 * CAP_RATIO, places=6)
        self.assertIn("authority_cap", by_id["doc"].components)
        self.assertEqual(doc.score, 0.95)  # the input object is never mutated

    def test_different_facts_are_untouched(self) -> None:
        user = Fake(mem("user", "USER_EXPLICIT", subject="truck"), 0.20)
        doc = Fake(mem("doc", "DOCUMENT_EXTRACTED", subject="trailer"), 0.99)
        results, capped = cap_conflicts([user, doc])
        self.assertEqual(capped, 0)
        self.assertEqual(doc.score, 0.99)

    def test_nothing_is_lifted(self) -> None:
        user = Fake(mem("user", "USER_EXPLICIT"), 0.10)
        doc = Fake(mem("doc", "DOCUMENT_EXTRACTED"), 0.12)
        results, capped = cap_conflicts([user, doc])
        self.assertEqual(capped, 1)
        by_id = {r.memory["id"]: r for r in results}
        self.assertAlmostEqual(by_id["doc"].score, 0.09, places=6)
        self.assertEqual(by_id["user"].score, 0.10)  # the user statement never moves

    def test_a_document_that_is_the_only_evidence_still_wins(self) -> None:
        doc = Fake(mem("doc", "DOCUMENT_EXTRACTED"), 0.88)
        results, capped = cap_conflicts([doc])
        self.assertEqual(capped, 0)
        self.assertEqual(doc.score, 0.88)

    def test_missing_slot_is_ignored(self) -> None:
        blank = {"id": "a", "source_category": "DOCUMENT_EXTRACTED", "subject": "", "predicate": ""}
        user = Fake({"id": "u", "source_category": "USER_EXPLICIT", "subject": "", "predicate": ""}, 0.5)
        results, capped = cap_conflicts([Fake(blank, 0.9), user])
        self.assertEqual(capped, 0)

    def test_equal_authority_is_untouched(self) -> None:
        a = Fake(mem("a", "USER_EXPLICIT"), 0.90)
        b = Fake(mem("b", "USER_EXPLICIT"), 0.89)
        results, capped = cap_conflicts([a, b])
        self.assertEqual(capped, 0)
        self.assertEqual(b.score, 0.89)


if __name__ == "__main__":
    unittest.main()
