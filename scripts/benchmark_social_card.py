#!/usr/bin/env python3
"""Render a shareable SVG card from a Cortex aggregate benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Cortex benchmark social card as SVG.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(report), encoding="utf-8")
    print(args.output)
    return 0


def render_svg(report: dict) -> str:
    cortex = report["conditions"]["cortex"]
    default = report["conditions"]["default_built_in"]
    delta = report["paired_deltas_cortex_minus_default"]
    ci = delta["agent_ttft_ms_hierarchical_bootstrap_median_95_ci"]

    cortex_accuracy = cortex["accuracy"] * 100
    default_accuracy = default["accuracy"] * 100
    cortex_bar = 1010 * cortex["accuracy"]
    default_bar = max(14, 1010 * default["accuracy"])
    prompt_change = (
        cortex["reported_prompt_tokens_median"] / default["reported_prompt_tokens_median"] - 1
    ) * 100
    subtitle = (
        f"Hermes additive mode  /  {report['total_pairs']} paired questions  /  "
        f"{report['corpus_memories']} memories  /  {report['runs']} runs"
    )
    model = escape(str(report["model"]))

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
  <rect width="1600" height="900" fill="#0b0f14"/>
  <circle cx="1470" cy="-40" r="330" fill="#18211f"/>
  <circle cx="1510" cy="35" r="155" fill="#21332f" opacity=".75"/>

  <g font-family="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">
    <text x="92" y="90" fill="#ff7658" font-size="24" font-weight="750" letter-spacing="4">CORTEX</text>
    <text x="92" y="151" fill="#f5f0e8" font-size="54" font-weight="720">Memory benchmark</text>
    <text x="92" y="195" fill="#92a09b" font-size="22">{escape(subtitle)}</text>

    <rect x="72" y="236" width="1456" height="378" rx="28" fill="#11171d" stroke="#253039"/>
    <text x="110" y="291" fill="#92a09b" font-size="19" font-weight="650" letter-spacing="2">EXACT-ANSWER ACCURACY</text>

    <text x="110" y="361" fill="#f5f0e8" font-size="26" font-weight="650">Built-in + Cortex</text>
    <text x="1440" y="361" fill="#b6e3da" font-size="38" font-weight="760" text-anchor="end">{cortex_accuracy:.1f}%</text>
    <rect x="110" y="389" width="1010" height="42" rx="21" fill="#242d33"/>
    <rect x="110" y="389" width="{cortex_bar:.1f}" height="42" rx="21" fill="#68c9b7"/>

    <text x="110" y="500" fill="#c1c7c4" font-size="26" font-weight="650">Hermes built-in snapshot</text>
    <text x="1440" y="500" fill="#c1c7c4" font-size="38" font-weight="760" text-anchor="end">{default_accuracy:.1f}%</text>
    <rect x="110" y="528" width="1010" height="42" rx="21" fill="#242d33"/>
    <rect x="110" y="528" width="{default_bar:.1f}" height="42" rx="21" fill="#707a80"/>

    <rect x="72" y="638" width="705" height="172" rx="26" fill="#11171d" stroke="#253039"/>
    <text x="110" y="690" fill="#92a09b" font-size="18" font-weight="650" letter-spacing="2">WHOLE-AGENT TTFT</text>
    <text x="110" y="746" fill="#f5f0e8" font-size="40" font-weight="740">Effectively tied</text>
    <text x="110" y="782" fill="#b6bfbc" font-size="19">{cortex['agent_ttft']['p50_ms']:.0f} ms Cortex  /  {default['agent_ttft']['p50_ms']:.0f} ms built-in</text>

    <rect x="801" y="638" width="727" height="172" rx="26" fill="#11171d" stroke="#253039"/>
    <text x="839" y="690" fill="#92a09b" font-size="18" font-weight="650" letter-spacing="2">THE TRADEOFF</text>
    <text x="839" y="746" fill="#ffad93" font-size="40" font-weight="740">+{prompt_change:.1f}% prompt tokens</text>
    <text x="839" y="782" fill="#b6bfbc" font-size="19">Cortex retrieval: {cortex['memory_prepare']['p50_ms']:.1f} ms p50</text>

    <text x="92" y="858" fill="#6f7d78" font-size="17">No clear TTFT difference: paired median 95% CI {ci[0]:+.0f} to {ci[1]:+.0f} ms  /  synthetic corpus  /  model: {model}</text>
  </g>
</svg>
'''


if __name__ == "__main__":
    raise SystemExit(main())
