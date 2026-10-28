"""Temporal classification and presentation tests using invented fixtures.

The fixtures preserve table, version, timestamp, and preference shapes without
copying operator history. Names, addresses, amounts, and settings are synthetic.
"""
import unittest

from tests._bootstrap import ROOT  # noqa: F401  (registers the flat `cortex` package)

from cortex.refinery import build_presentation
from cortex.temporal import (
    DURABLE,
    MIXED,
    classify_temporal,
    find_as_of,
)

# Entirely synthetic examples; network addresses use a documentation range.
HOST_SPECS = (
    "Example Lab › Host Specs | Attribute | Value | |---|---|---| | Hostname | cedar | "
    "| IP | 192.0.2.20 | | PVE version | 8.2.1 (pve-manager) | | CPU | Example CPU | "
    "| RAM | 64 GiB total — 16 GiB used (25%) | | Uptime | 12.5 days | "
    "| Load avg | 0.20, 0.30, 0.40 |"
)
DEFAULT_MODEL = (
    "Example Agent › Default model → Example Model 2.0 — 2026-09-02 › "
    "Where to change models in the FUTURE (checklist, verified 2026-09-02) "
    "| `./config.yaml` → `model:` | `default: example-model-2.0` |"
)
ACCOUNT_SNAPSHOT = (
    "Example Accounts — Consolidated view of fictional account balances. "
    "Live snapshot via `./scripts/example_accounts.py snapshot`. "
    "## Current balances (as of 2026-08-07 — synthetic snapshot) "
    "| Account | Balance | APR | Source | |---|---|---|---| "
    "| Example Card | $125.00 | 12.50% | Example service |"
)
DURABLE_EXAMPLE = (
    "For self-hosted Example Lab services, Avery prefers placing them in a "
    "dedicated unprivileged container rather than on the shared host. "
    "Policy: never on the bare host."
)

class TestTemporalClassifier(unittest.TestCase):
    def test_flags_mixed_synthetic_tables_and_configuration(self):
        """Durable context and changing values remain distinguishable."""
        for name, text in (
            ("host specs", HOST_SPECS),
            ("default model", DEFAULT_MODEL),
            ("account snapshot", ACCOUNT_SNAPSHOT),
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
        verdict = classify_temporal(HOST_SPECS)
        self.assertIn("Uptime", verdict.volatile_markers)
        self.assertIn("Load avg", verdict.volatile_markers)
        # the worst class: volatile values and nothing to date them from
        self.assertIsNone(verdict.as_of)
        self.assertTrue(verdict.undated)
        self.assertIn("temporal_undated", verdict.flags())

    def test_recovers_as_of_and_stamps_the_action(self):
        verdict = classify_temporal(ACCOUNT_SNAPSHOT)
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

    def test_find_as_of_formats(self) -> None:
        self.assertEqual(find_as_of("as of 2026-08-07 done"), "2026-08-07")
        self.assertEqual(find_as_of("snapshot Aug 21, 2026"), "Aug 21, 2026")
        self.assertIsNone(find_as_of("no dates here"))

    def test_invalid_dates_never_become_as_of(self) -> None:
        """A date-shaped token must parse as a real date before it is trusted."""
        self.assertIsNone(find_as_of("Snapshot 2026-99-99 balance $40."))
        self.assertIsNone(find_as_of("as of 2026-13-01"))
        self.assertEqual(find_as_of("invalid 2026-99-99 then real 2026-08-07"), "2026-08-07")

    def test_invalid_date_snapshot_is_undated_not_stamped(self) -> None:
        verdict = classify_temporal("Snapshot 2026-99-99 balance $40.")
        self.assertIsNone(verdict.as_of)
        self.assertTrue(verdict.undated)
        self.assertIn("temporal_undated", verdict.flags())

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
        presentation = build_presentation(self._memory(ACCOUNT_SNAPSHOT), {"record_role": "canonical"})
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
        memory = self._memory(HOST_SPECS)
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
