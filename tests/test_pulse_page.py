"""Static contract for the glanceable pulse tile.

UI-003 owns pulse.html only. The server, including the 401 gate, is RS-003.
These checks load design/dot-glyphs.json and compare it to the page. They do
not import a Python or Rust renderer.
"""

from __future__ import annotations

import json
import re
import unittest

from tests._bootstrap import ROOT

PAGE = ROOT / "pulse.html"
GLYPHS = ROOT / "design" / "dot-glyphs.json"
LIMIT = 20 * 1024


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _js_int(name: str, text: str) -> int:
    match = re.search(rf"const {name} = (\d+);", text)
    if match is None:
        raise AssertionError(f"{name} is missing from pulse.html")
    return int(match.group(1))


def _js_array(name: str, text: str) -> list[str]:
    match = re.search(rf"const {name} = (\[.*?\]);", text)
    if match is None:
        raise AssertionError(f"{name} is missing from pulse.html")
    loaded = json.loads(match.group(1))
    if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
        raise AssertionError(f"{name} is not a list of strings")
    return loaded


class PulsePageTests(unittest.TestCase):
    def test_file_is_under_20kb_and_has_no_remote_url(self) -> None:
        data = PAGE.read_bytes()
        self.assertLess(len(data), LIMIT, f"pulse.html is {len(data)} bytes")
        text = data.decode("utf-8")
        self.assertNotIn("http://", text)
        self.assertNotIn("https://", text)
        self.assertNotIn("<script src", text.lower())
        self.assertNotIn("<link", text.lower())
        self.assertNotIn("@import", text)
        self.assertNotIn("@font-face", text)
        self.assertNotIn("<canvas", text.lower())
        self.assertNotIn("<svg", text.lower())
        self.assertNotIn("<img", text.lower())

    def test_glyph_mapping_matches_dot_glyphs_file(self) -> None:
        spec = json.loads(GLYPHS.read_text(encoding="utf-8"))
        text = _text()
        self.assertEqual(_js_array("GLYPHS", text), spec["grid"]["glyphs"])
        self.assertEqual(_js_int("COLS", text), spec["layout"]["cols"])
        self.assertEqual(_js_int("ROWS", text), spec["layout"]["rows"])
        self.assertIn("return GLYPHS[level];", text)
        self.assertIn('fetch("/api/pulse?cols=" + COLS + "&rows=" + ROWS', text)
        self.assertNotIn("/api/snapshot", text)

    def test_unreachable_branch_text_is_present(self) -> None:
        text = _text()
        self.assertIn('const UNREACHABLE = "server unreachable";', text)
        self.assertIn("hold(UNREACHABLE)", text)

    def test_page_is_a_pre_grid_that_polls_and_pauses_when_hidden(self) -> None:
        text = _text()
        self.assertGreaterEqual(text.count("<pre"), 2)
        self.assertIn('id="grid"', text)
        self.assertIn("clamp(", text)
        self.assertIn("ui-monospace", text)
        self.assertIn("monospace", text)
        self.assertEqual(_js_int("INTERVAL_MS", text), 2000)
        self.assertIn("setInterval(pull, INTERVAL_MS)", text)
        self.assertIn('document.visibilityState === "hidden"', text)
        self.assertIn("new AbortController", text)
        self.assertIn("pagehide", text)
        self.assertIn('const NO_DATA = "no data yet";', text)
        self.assertIn('const NOT_MEASURABLE = "not measurable yet";', text)
        self.assertIn("the store is empty", text)
        for word in ("not run yet", "running", "stale", "fresh", "failed"):
            self.assertIn(word, text)

    def test_dark_is_the_default_and_light_follows_the_color_scheme(self) -> None:
        text = _text()
        prelude, _match, rest = text.partition("@media (prefers-color-scheme: light)")
        self.assertTrue(_match)
        self.assertIn("--bg: #090b10", prelude)
        self.assertIn("--text: #f5f6f8", prelude)
        self.assertNotIn("--bg: #f4f5f8", prelude)
        self.assertIn("--bg: #f4f5f8", rest)
        self.assertIn("--text: #171a22", rest)
        self.assertIn("color-scheme: dark", prelude)
        self.assertIn("color-scheme: light", rest)
