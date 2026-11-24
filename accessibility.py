"""An accessibility pass that runs in CI instead of in a browser (#173)

An accessibility review done by eye is done once. This is the same review as a **checker**, so a rule
that was satisfied in October stays satisfied in January:

- **a placeholder is not a name.** `placeholder="Search memories"` gives a screen reader nothing to
  announce once the field has content, so every field needs a label or an `aria-label`.
- **a colour is not a signal.** Anything that means "error" or "pinned" must carry a word or a shape;
  a red dot alone is invisible to a reader who cannot see red, and to every text surface that has no
  colour at all (the TUI panel, a pasted receipt).
- **contrast is computed, not eyeballed.** WCAG ratio from the actual palette, with the threshold stated
  per text size — and `unknown` when a colour cannot be resolved, never a confident `pass`.
- **the checker says what it cannot check.** Focus traps, screen-reader announcements and the rendered
  result are named as *manual* checks rather than silently omitted; a green report that implies more
  coverage than it has is worse than no report.
- **motion is optional.** A transition that cannot be switched off with `prefers-reduced-motion` is a
  rule violation, not a taste question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# WCAG 2.1 AA: 4.5:1 for body text, 3:1 for large text (>=18.66px bold or >=24px).
AA_BODY = 4.5
AA_LARGE = 3.0
MIN_TARGET_PX = 44
_LARGE_TEXT_PX = 24.0

_TAG = re.compile(r"<(input|button|a|img|select|textarea)\b([^>]*)>", re.I)
_ATTR = re.compile(r"([a-zA-Z-]+)\s*=\s*\"([^\"]*)\"")
_HEX = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_RGB = re.compile(r"^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", re.I)

# Rules that cannot be decided from source. Stated so the report is not mistaken for full coverage.
MANUAL = (
    "focus order and focus trapping inside the sheet",
    "what a screen reader actually announces",
    "whether the rendered contrast matches the declared palette",
    "touch accuracy on a real device",
)


@dataclass
class Finding:
    rule: str
    element: str
    detail: str
    severity: str = "error"   # error | warn | manual

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "element": self.element, "detail": self.detail,
                "severity": self.severity}


def _attrs(blob: str) -> dict[str, str]:
    return {name.lower(): value for name, value in _ATTR.findall(blob)}


def _labelled_ids(html: str) -> set[str]:
    """Ids that a <label for=...> points at. A visible label is the best kind of name."""
    return {value for value in re.findall(r"<label\b[^>]*\bfor\s*=\s*\"([^\"]+)\"", html, re.I)}


def _has_name(tag: str, attrs: dict[str, str], *, labelled: set[str] | None = None,
              text: str = "") -> bool:
    if attrs.get("aria-label", "").strip() or attrs.get("aria-labelledby", "").strip():
        return True
    if attrs.get("title", "").strip():
        return True
    if attrs.get("id", "").strip() and attrs.get("id", "") in (labelled or set()):
        return True
    if tag == "button" and text.strip():
        return True
    return False


def audit_html(html: str) -> list[Finding]:
    """Mechanical rules over one served page. Findings name what to change, not just what is wrong."""
    body = str(html or "")
    labelled = _labelled_ids(body)
    found: list[Finding] = []
    if not re.search(r"<html[^>]*\blang\s*=", body, re.I):
        found.append(Finding("lang", "<html>", "set lang so a screen reader uses the right voice"))
    for match in _TAG.finditer(body):
        tag = match.group(1).lower()
        attrs = _attrs(match.group(2))
        if tag == "img" and "alt" not in attrs:
            found.append(Finding("img-alt", "<img>", "an image without alt is announced as its filename"))
        # The text between a button's tags is its name, so it has to be looked at, not assumed absent.
        closing = body.find("</" + tag + ">", match.end())
        inner = body[match.end():closing] if closing > match.end() else ""
        if tag in {"input", "select", "textarea"}:
            kind = attrs.get("type", "text").lower()
            if kind not in {"hidden", "submit", "button"} and not _has_name(
                    tag, attrs, labelled=labelled):
                found.append(Finding(
                    "field-name", f"<{tag} id={attrs.get('id', '?')}>",
                    "a placeholder is not a name: add a <label for> or aria-label",
                ))
        if tag == "button" and not _has_name(tag, attrs, labelled=labelled, text=inner):
            found.append(Finding("button-name", "<button>", "a button needs text or aria-label"))
        if tag == "a" and "href" in attrs and not attrs["href"].strip():
            found.append(Finding("link-href", "<a>", "an empty href is not a link"))
    if re.search(r'role\s*=\s*"dialog"', body, re.I):
        if not re.search(r'role\s*=\s*"dialog"[^>]*aria-label', body, re.I):
            found.append(Finding("dialog-name", '<div role="dialog">',
                                 "a dialog needs aria-labelledby or aria-label"))
        if not re.search(r'aria-modal\s*=\s*"true"', body, re.I):
            found.append(Finding("dialog-modal", '<div role="dialog">',
                                 "mark the dialog aria-modal so the list behind it is not read"))
    if re.search(r"transition\s*:", body, re.I) and not re.search(
            r"prefers-reduced-motion", body, re.I):
        found.append(Finding("reduced-motion", "<style>",
                             "add @media (prefers-reduced-motion: reduce) to disable transitions"))
    if re.search(r":focus\b", body) is None and re.search(r"<button|<input|<a\b", body, re.I):
        found.append(Finding("focus-visible", "<style>",
                             "no :focus style: keyboard users cannot see where they are"))
    if re.search(r'role\s*=\s*"tab"', body, re.I) and not re.search(r"aria-controls", body, re.I):
        found.append(Finding("tab-controls", 'role="tab"',
                             "tabs need aria-controls pointing at their panel", "warn"))
    if re.search(r'id\s*=\s*"out"', body) and not re.search(r'aria-live', body, re.I):
        found.append(Finding("live-region", "<main id=out>",
                             "results that change without navigation need aria-live", "warn"))
    found.extend(Finding("manual", "-", item, "manual") for item in MANUAL)
    return found


def _luminance(color: str) -> float | None:
    text = str(color or "").strip()
    match = _HEX.match(text)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(char * 2 for char in digits)
        red, green, blue = (int(digits[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    else:
        match = _RGB.match(text)
        if not match:
            return None
        red, green, blue = (int(match.group(index)) / 255.0 for index in (1, 2, 3))

    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def contrast(foreground: str, background: str) -> float | None:
    """WCAG contrast ratio, or None when a colour cannot be resolved."""
    front, back = _luminance(foreground), _luminance(background)
    if front is None or back is None:
        return None
    high, low = max(front, back), min(front, back)
    return round((high + 0.05) / (low + 0.05), 2)


def contrast_verdict(foreground: str, background: str, *, font_px: float = 16.0,
                     bold: bool = False) -> dict[str, Any]:
    """`unknown` is not `pass`: an unresolvable colour fails the check by being unmeasured."""
    ratio = contrast(foreground, background)
    large = font_px >= _LARGE_TEXT_PX or (bold and font_px >= 18.66)
    threshold = AA_LARGE if large else AA_BODY
    if ratio is None:
        return {"ratio": None, "threshold": threshold, "verdict": "unknown",
                "detail": "a colour could not be resolved, so contrast was not measured"}
    return {
        "ratio": ratio,
        "threshold": threshold,
        "verdict": "pass" if ratio >= threshold else "fail",
        "detail": f"{ratio}:1 against {threshold}:1 for {'large' if large else 'body'} text",
    }


def palette_audit(pairs: list[tuple[str, str, str]], **options: Any) -> list[dict[str, Any]]:
    """Check the declared palette, not the pixels. Pairs are (name, foreground, background)."""
    report = []
    for name, foreground, background in pairs:
        checked = contrast_verdict(foreground, background, **options)
        report.append({"name": name, "foreground": foreground, "background": background, **checked})
    return report


def audit_text_ui(lines: list[str]) -> list[Finding]:
    """A terminal surface cannot show colour-only meaning: a reader may be piping it to a file."""
    found: list[Finding] = []
    for index, line in enumerate(lines, start=1):
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", str(line))
        if "\x1b[" in str(line) and not stripped.strip():
            found.append(Finding("colour-only", f"line {index}",
                                 "a line that is only colour carries no meaning when piped"))
        if re.search(r"\x1b\[(?:3[1-9]|4[1-9])m", str(line)) and not re.search(r"[a-zA-Z]", stripped):
            found.append(Finding("colour-only", f"line {index}",
                                 "the signal must be a word or a shape, not a hue"))
    return found


def summary(findings: list[Finding]) -> dict[str, Any]:
    counts = {"error": 0, "warn": 0, "manual": 0}
    for item in findings:
        counts[item.severity] = counts.get(item.severity, 0) + 1
    return {
        "errors": counts.get("error", 0),
        "warnings": counts.get("warn", 0),
        "manual_checks": counts.get("manual", 0),
        "verdict": "pass" if not counts.get("error") else "fail",
        "note": "a pass covers only the rules a machine can decide; the manual checks are listed",
    }


def render(findings: list[Finding]) -> list[str]:
    described = summary(findings)
    lines = [f"accessibility: {described['verdict']} "
             f"({described['errors']} error(s), {described['warnings']} warning(s), "
             f"{described['manual_checks']} manual)"]
    for item in findings:
        if item.severity != "manual":
            lines.append(f"  [{item.severity}] {item.rule}: {item.element} — {item.detail}")
    for item in findings:
        if item.severity == "manual":
            lines.append(f"  [manual] {item.detail}")
    return lines
