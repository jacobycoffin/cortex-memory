"""Tests for temporal validity classification and its presentation wiring.

The three probe texts are trimmed from the REAL memories the operator flagged
on 2026-09-11, so these tests fail if the classifier regresses on the cases that
motivated it.
"""
import unittest

from tests._bootstrap import ROOT  # noqa: F401  (registers the flat `cortex` package)

from cortex.refinery import build_presentation
from cortex.temporal import (
    DURABLE,
    MIXED,
    MOSTLY_DURABLE,
    VOLATILE,
    classify_temporal,
    find_as_of,
)

# --- trimmed from the live corpus (operator-flagged) --------------------------
PROXMOX_HOST_SPECS = (
    "Proxmox › Host Specs | Attribute | Value | |---|---|---| | Hostname | r630 | "
    "| IP | 192.168.1.208 | | PVE version | 8.4.19 (pve-manager) | | CPU | Intel Xeon | "
    "| RAM | 188.5 GiB total — 36.3 GB used (17.9%) | | Uptime | 36.5 days | "
    "| Load avg | 1.64, 1.56, 1.92 |"
)
DEFAULT_MODEL = (
    "Hermes Agent › Default model → Muse Spark 1.3 Contributor — 2026-09-02 › "
    "Where to change models in the FUTURE (full checklist, verified 2026-09-02) "
    "| `/root/.hermes/config.yaml` → `model:` | `default: muse-spark-1.3-contributor` "
    "| ALWAYS — this is the single source of truth |"
)
DEBT_OVERVIEW = (
    "Debt Overview — Consolidated view of Jacoby's debt (Plaid-connected + manual "
    "Apple Card). Live snapshot via `~/.hermes/scripts/plaid_api.py snapshot`. "
    "## Current balances (as of 2026-08-07 — live Plaid snapshot) "
    "| Debt | Balance | APR | Source | |---|---|---|---| "
    "| Frontier Airlines Mastercard | $382.24 | 29.49% | Plaid live |"
)
DURABLE_EXAMPLE = (
    "For self-hosted Jarvis services, Jacoby prefers placing them in a dedicated "
    "unprivileged LXC on the physical Proxmox homelab rather than on the VPS. "
    "Policy: never on the bare Proxmox host."
)


class TestTemporalClassifier(unittest.TestCase):
    def test_flags_the_three_memories_the_operator_flagged(self):
        """Each was a correct retrieval whose volatile values were unmarked."""
        for name, text in (
            ("proxmox host specs", PROXMOX_HOST_SPECS),
            ("default model", DEFAULT_MODEL),
            ("debt overview", DEBT_OVERVIEW),
        ):
            with self.subTest(memory=name):
                verdict = classify_temporal(text)
                self.assertEqual(
                    verdict.classification,
                    MIXED,
                    f"{name} should be MIXED, got {verdict.classification} "
                    f"(markers={verdict.volatile_markers})",
                )
                self.assertTrue(verdict.needs_stamp)

    def test_uptime_and_load_are_caught_without_an_explicit_as_of(self):
        verdict = classify_temporal(PROXMOX_HOST_SPECS)
        self.assertIn("Uptime", verdict.volatile_markers)
        self.assertIn("Load avg", verdict.volatile_markers)
        # the worst class: volatile values and nothing to date them from
        self.assertIsNone(verdict.as_of)
        self.assertTrue(verdict.undated)
        self.assertIn("temporal_undated", verdict.flags())

    def test_recovers_as_of_and_stamps_the_action(self):
        verdict = classify_temporal(DEBT_OVERVIEW)
        self.assertEqual(verdict.as_of, "2026-08-07")
        self.assertIn("as_of=2026-08-07", verdict.action)

    def test_version_pin_is_case_sensitive(self):
        """re.I at the call site once made 'the 5.49' read as a product version."""
        self.assertNotIn(
            "the 5.49", classify_temporal("we should look at the 5.49 figure").volatile_markers
        )
        self.assertTrue(classify_temporal("PVE version | 8.4.19").volatile_score > 0)

    def test_plain_preference_is_durable(self):
        verdict = classify_temporal(DURABLE_EXAMPLE)
        self.assertEqual(verdict.classification, DURABLE)
        self.assertEqual(verdict.flags(), [])
        self.assertFalse(verdict.needs_stamp)

    def test_empty_text_is_safe(self):
        for value in ("", "   ", None):
            with self.subTest(value=value):
                self.assertEqual(classify_temporal(value).action, "empty")

    def test_find_as_of_formats(self):
        self.assertEqual(find_as_of("as of 2026-08-07 done"), "2026-08-07")
        self.assertEqual(find_as_of("snapshot Aug 21, 2026"), "Aug 21, 2026")
        self.assertIsNone(find_as_of("no dates here"))

    def test_a_bare_number_is_not_a_dated_value(self):
        """A date is itself digits; matching bare counts flagged every dated record."""
        verdict = classify_temporal("Reviewed on 2026-08-07 with 42 checks passing")
        self.assertEqual(verdict.classification, DURABLE)


class TestPresentationWiring(unittest.TestCase):
    def _memory(self, content: str) -> dict:
        return {
            "content": content,
            "source_category": "AUTOMATIC_APPROVED",
            "record_role": "canonical",
        }

    def test_volatile_record_carries_flags_and_as_of(self):
        presentation = build_presentation(self._memory(DEBT_OVERVIEW), {"record_role": "canonical"})
        self.assertIn("temporal_mixed", presentation["readability_flags"])
        self.assertIn("as of 2026-08-07", presentation["applies_when"])

    def test_durable_record_is_untouched(self):
        presentation = build_presentation(
            self._memory(DURABLE_EXAMPLE), {"record_role": "canonical"}
        )
        temporal_flags = [f for f in presentation["readability_flags"] if f.startswith("temporal_")]
        self.assertEqual(temporal_flags, [])
        self.assertNotIn("as of", presentation["applies_when"].lower())

    def test_wiring_does_not_change_role_or_content(self):
        """Flags are additive: role and content must be untouched."""
        memory = self._memory(PROXMOX_HOST_SPECS)
        presentation = build_presentation(memory, {"record_role": "reference"})
        self.assertEqual(memory["record_role"], "canonical")
        self.assertTrue(presentation["display_summary"])
        # a temporal flag must never appear in CLARITY_FLAGS, which drives roles
        from cortex.refinery import CLARITY_FLAGS

        for flag in presentation["readability_flags"]:
            if flag.startswith("temporal_"):
                self.assertNotIn(flag, CLARITY_FLAGS)


if __name__ == "__main__":
    unittest.main()
