# shellcheck shell=bash
# Sourced helper: print the Python interpreter Hermes imports plugins with.
#
# Order: an explicit path in $1 (callers pass PYTHON_BIN / CORTEX_PYTHON_BIN /
# HERMES_PYTHON), then current PM-managed Hermes (ask the launcher with
# `hermes --run-module site`), then an older console-script shebang, then the
# legacy venv locations, then python3 on PATH as a last resort.
# Note: PM-managed Hermes rebuilds its environment on `hermes update` /
# `hermes pm repair`, so systemd units that embed this path should be
# re-installed afterwards (re-running scripts/install.sh does that check).
cortex_find_hermes_python() {
  local explicit="${1:-}" home="${HERMES_HOME:-$HOME/.hermes}" bin="" out sp shebang candidate
  if [[ -n "$explicit" ]]; then echo "$explicit"; return 0; fi
  for candidate in "$(command -v hermes 2>/dev/null || true)" "$HOME/.local/bin/hermes" "$home/hermes-agent/.hermes/bin/hermes"; do
    [[ -n "$candidate" && -x "$candidate" ]] && { bin="$candidate"; break; }
  done
  if [[ -n "$bin" ]]; then
    out="$(HERMES_HOME="$home" timeout 90 "$bin" --run-module site 2>/dev/null || true)"
    sp="$(printf '%s\n' "$out" | grep -o "'[^']*/lib/python3[0-9.]*/site-packages'" | tr -d "'" | grep -v '/tools/python' | head -n1 || true)"
    if [[ -n "$sp" && -x "${sp%/lib/python*}/bin/python" ]]; then echo "${sp%/lib/python*}/bin/python"; return 0; fi
    shebang="$(head -n1 "$bin" 2>/dev/null | sed -n 's/^#! *//p' | awk '{print $1}')"
    if [[ "$shebang" == */python* && -x "$shebang" ]]; then echo "$shebang"; return 0; fi
  fi
  for candidate in "$home/hermes-agent/venv/bin/python" "$home/hermes-agent/.venv/bin/python" \
                   "$home/venv/bin/python" /usr/local/lib/hermes-agent/venv/bin/python; do
    [[ -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  command -v python3 2>/dev/null && return 0
  return 1
}
