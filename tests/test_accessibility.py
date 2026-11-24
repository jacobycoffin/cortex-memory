"""An accessibility pass that runs in CI instead of in a browser (#173)."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import accessibility as ax

PHONE = Path(__file__).resolve().parent.parent / "phone.html"


def rules(findings):
    return {item.rule for item in findings if item.severity != "manual"}


class HtmlTests(unittest.TestCase):
    def test_a_page_without_a_lang_is_flagged(self) -> None:
        self.assertIn("lang", rules(ax.audit_html("<html><body></body></html>")))

    def test_a_page_with_a_lang_is_not(self) -> None:
        self.assertNotIn("lang", rules(ax.audit_html('<html lang="en"></html>')))

    def test_a_placeholder_is_not_a_name(self) -> None:
        """The most common real failure: a search field that announces nothing once it has content."""
        found = ax.audit_html('<html lang="en"><input type="search" placeholder="Search"></html>')
        self.assertIn("field-name", rules(found))

    def test_an_aria_label_names_a_field(self) -> None:
        found = ax.audit_html('<html lang="en"><input type="search" aria-label="Search"></html>')
        self.assertNotIn("field-name", rules(found))

    def test_a_label_element_names_a_field(self) -> None:
        found = ax.audit_html(
            '<html lang="en"><label for="q">Search</label><input id="q" type="search"></html>')
        self.assertNotIn("field-name", rules(found))

    def test_a_hidden_input_needs_no_name(self) -> None:
        found = ax.audit_html('<html lang="en"><input type="hidden" name="x"></html>')
        self.assertNotIn("field-name", rules(found))

    def test_a_button_with_text_has_a_name(self) -> None:
        found = ax.audit_html('<html lang="en"><button>Recent</button></html>')
        self.assertNotIn("button-name", rules(found))

    def test_a_button_without_text_is_flagged(self) -> None:
        found = ax.audit_html('<html lang="en"><button></button></html>')
        self.assertIn("button-name", rules(found))

    def test_an_icon_button_with_an_aria_label_passes(self) -> None:
        found = ax.audit_html('<html lang="en"><button aria-label="Close"></button></html>')
        self.assertNotIn("button-name", rules(found))

    def test_an_image_without_alt_is_flagged(self) -> None:
        self.assertIn("img-alt", rules(ax.audit_html('<html lang="en"><img src="a.png"></html>')))

    def test_a_dialog_needs_a_name_and_aria_modal(self) -> None:
        found = rules(ax.audit_html('<html lang="en"><div role="dialog"></div></html>'))
        self.assertIn("dialog-name", found)
        self.assertIn("dialog-modal", found)

    def test_transitions_without_reduced_motion_are_flagged(self) -> None:
        found = ax.audit_html('<html lang="en"><style>#a { transition: opacity .2s; }</style></html>')
        self.assertIn("reduced-motion", rules(found))

    def test_reduced_motion_satisfies_the_rule(self) -> None:
        found = ax.audit_html('<html lang="en"><style>#a { transition: opacity .2s; }'
                              '@media (prefers-reduced-motion: reduce) { #a { transition: none; } }'
                              '</style></html>')
        self.assertNotIn("reduced-motion", rules(found))

    def test_missing_focus_style_is_flagged(self) -> None:
        found = ax.audit_html('<html lang="en"><button aria-label="x">a</button></html>')
        self.assertIn("focus-visible", rules(found))

    def test_tabs_without_aria_controls_warn(self) -> None:
        found = ax.audit_html('<html lang="en"><button role="tab" aria-label="a">a</button></html>')
        self.assertIn("tab-controls", rules(found))

    def test_the_manual_checks_are_always_listed(self) -> None:
        """A green report must not imply more coverage than it has."""
        found = ax.audit_html('<html lang="en"></html>')
        self.assertEqual(sum(1 for item in found if item.severity == "manual"), len(ax.MANUAL))


class ContrastTests(unittest.TestCase):
    def test_white_on_black_is_the_maximum(self) -> None:
        self.assertEqual(ax.contrast("#ffffff", "#000000"), 21.0)

    def test_a_short_hex_is_expanded(self) -> None:
        self.assertEqual(ax.contrast("#fff", "#000"), 21.0)

    def test_rgb_notation_is_understood(self) -> None:
        self.assertEqual(ax.contrast("rgb(255,255,255)", "#000000"), 21.0)

    def test_an_unresolvable_colour_is_unknown_not_pass(self) -> None:
        verdict = ax.contrast_verdict("var(--x)", "#000000")
        self.assertEqual(verdict["verdict"], "unknown")
        self.assertIsNone(verdict["ratio"])

    def test_body_text_needs_four_point_five(self) -> None:
        verdict = ax.contrast_verdict("#7d8ea1", "#0b0d10", font_px=16)
        self.assertEqual(verdict["threshold"], 4.5)
        self.assertEqual(verdict["verdict"], "pass")

    def test_large_text_needs_three(self) -> None:
        self.assertEqual(ax.contrast_verdict("#808080", "#000000", font_px=24)["threshold"], 3.0)

    def test_bold_at_18_66_counts_as_large(self) -> None:
        verdict = ax.contrast_verdict("#808080", "#000000", font_px=19, bold=True)
        self.assertEqual(verdict["threshold"], 3.0)

    def test_a_failing_pair_says_by_how_much(self) -> None:
        verdict = ax.contrast_verdict("#333333", "#000000")
        self.assertEqual(verdict["verdict"], "fail")
        self.assertIn(":1 against", verdict["detail"])

    def test_the_palette_audit_covers_every_pair(self) -> None:
        report = ax.palette_audit([("body", "#e8ecf1", "#0b0d10"), ("meta", "#7d8ea1", "#0b0d10")])
        self.assertEqual([item["name"] for item in report], ["body", "meta"])
        self.assertTrue(all(item["verdict"] == "pass" for item in report))


class TextUiTests(unittest.TestCase):
    def test_a_colour_only_line_is_flagged(self) -> None:
        found = ax.audit_text_ui(["\x1b[31m\x1b[0m"])
        self.assertIn("colour-only", rules(found))

    def test_a_word_carrying_the_colour_is_fine(self) -> None:
        self.assertEqual(ax.audit_text_ui(["\x1b[31mERROR\x1b[0m disk full"]), [])

    def test_plain_text_is_fine(self) -> None:
        self.assertEqual(ax.audit_text_ui(["no colour here"]), [])


class SummaryTests(unittest.TestCase):
    def test_the_verdict_fails_on_an_error(self) -> None:
        described = ax.summary([ax.Finding("x", "y", "z")])
        self.assertEqual(described["verdict"], "fail")

    def test_a_manual_check_does_not_fail_the_verdict(self) -> None:
        described = ax.summary([ax.Finding("manual", "-", "look", "manual")])
        self.assertEqual(described["verdict"], "pass")
        self.assertEqual(described["manual_checks"], 1)

    def test_the_note_states_the_limit_of_a_pass(self) -> None:
        self.assertIn("machine can decide", ax.summary([])["note"])

    def test_the_render_lists_the_manual_checks_last(self) -> None:
        lines = ax.render(ax.audit_html('<html lang="en"></html>'))
        self.assertTrue(lines[0].startswith("accessibility: pass"))
        self.assertIn("[manual]", lines[-1])


class ServedPageTests(unittest.TestCase):
    def test_the_phone_surface_passes_every_mechanical_rule(self) -> None:
        found = ax.audit_html(PHONE.read_text(encoding="utf-8"))
        errors = [item for item in found if item.severity == "error"]
        self.assertEqual(errors, [], f"{len(errors)} accessibility error(s) in phone.html")

    def test_the_phone_palette_meets_aa(self) -> None:
        report = ax.palette_audit([
            ("body", "#e8ecf1", "#0b0d10"),
            ("meta", "#7d8ea1", "#0b0d10"),
            ("heading", "#9fb0c3", "#10141a"),
            ("link", "#9ecbff", "#10141a"),
            ("tab", "#cbd6e2", "#141a22"),
            ("close", "#e8ecf1", "#1d2a3a"),
        ])
        self.assertTrue(all(item["verdict"] == "pass" for item in report),
                        [item for item in report if item["verdict"] != "pass"])

    def test_the_search_field_has_a_real_label(self) -> None:
        body = PHONE.read_text(encoding="utf-8")
        self.assertIn('<label class="sr" for="q">', body)

    def test_motion_can_be_switched_off(self) -> None:
        self.assertIn("prefers-reduced-motion", PHONE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
