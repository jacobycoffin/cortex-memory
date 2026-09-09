# Adapter + dashboard split plan — `__init__.py` and dashboard assets

Measured 2026-09-09: `__init__.py` 2,045 lines (Hermes adapter +
`CortexMemoryProvider` + install/register helpers), `dashboard.py` 1,598
lines, `dashboard.html` 4,786 lines (single file, served from disk by
`dashboard.py:149` via `Path(__file__).with_name("dashboard.html")`).

No framework rewrite: the dashboard stays server-rendered vanilla
HTML/CSS/JS; the split is file organization only.

## `__init__.py` split (preserve public imports + plugin loading)

`register(ctx)`, `CortexMemoryProvider`, and the `cortex.*` re-exports used
by `tests/_bootstrap.py` (`import cortex` → `cortex.store`, `cortex.client`,
`cortex.harness`, …) must keep working untouched.

| Stage | Move | To |
|-------|------|----|
| A | Output-hook helpers (`_hermes_transform_llm_output_hook`, `_install_hermes_output_hook`) | `hermes_hooks.py` |
| B | Receipt/feedback helpers (`_receipt_feedback_outcome`, `_resolve_allowed_prefixes`, attribution scoring glue) | `hermes_receipts.py` |
| C | Config readers (`_read_config`, `_as_bool`, `_string_tuple`, `_json_ok/_json_error`) | `hermes_config.py` |
| D | `CortexMemoryProvider` itself (imports A–C) | `hermes_provider.py`, re-exported from `__init__.py` |

Acceptance per stage: `register(ctx)` import path unchanged, Hermes gateway
loads the plugin identically, `tests/test_provider.py` green, no behavior
diff (provider recall plumbing untouched — the legacy path keeps calling
`store.resolve_usage` directly with its own attribution).

## Dashboard split (keep installed dashboards working)

`dashboard.py` serves `dashboard.html` from disk plus `public_assets`
(path → content-type/payload map). `scripts/install_local.sh` installs both.

| Stage | Move | To |
|-------|------|----|
| E | Inline `<style>` → `dashboard.css`, served via `public_assets` | `dashboard.css` |
| F | Inline `<script>` feature logic → `dashboard.js` (+ `dashboard-api.js` for API helpers) | `dashboard.js`, `dashboard-api.js` |
| G | `dashboard.html` becomes a thin shell referencing the above | — |

Acceptance per stage: `scripts/check_repository.py` green (XML/SVG asset
gates unaffected), dashboard rendered page byte-equivalent modulo asset
tags, `tests/test_dashboard.py` green (incl. the cross-instance
password-reset regression test), installer copies the new assets
(`install_local.sh` + asset manifest updated together — never one without
the other), auth/session behavior unchanged.

## Explicitly out of scope

- No visual redesign, no framework migration, no API change.
- No auth-logic change beyond file moves (the `DashboardAuth` mtime/size
  reload fix stays covered by its regression test).
