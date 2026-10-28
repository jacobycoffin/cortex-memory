#!/usr/bin/env bash
# Cortex one-step installer. Detects the agent that is running it (Claude Code
# or Hermes) and performs the complete setup for that agent, end to end:
# dependencies, local embedding model, registration, fast recall, verification.
#
#   scripts/install.sh                 # auto-detect
#   scripts/install.sh --agent claude  # or: --agent hermes
#   scripts/install.sh --detect        # print what would be chosen and why, change nothing
#
# Safe to re-run: every step is idempotent and upgrades in place.
# Optional paid/remote components stay OFF unless you pass an --enable flag:
#   --enable-jev / --no-jev                 TypeSafe Jev judgment engine (Hermes auto-judge only)
#   --enable-auto-judge / --no-auto-judge   remote LLM auto-judge timer (Hermes only)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT=""
DETECT_ONLY=0
JEV=""            # "", on, off
AUTO_JUDGE=""     # "", on, off
WITH_HOOKS=1
WITH_MODEL=1
WITH_CLAUDE_MD=1
WITH_VERIFY=1

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; cat <<'USAGE'
Other options:
  --no-hooks        Claude Code: do not install SessionStart/UserPromptSubmit/Stop hooks
  --no-claude-md    Claude Code: do not add the Cortex block to ~/.claude/CLAUDE.md
  --no-model        skip the local embedding model (lexical recall only)
  --no-verify       skip the post-install verification
Environment:
  CORTEX_HOME       Claude Code data dir (default ~/.cortex)
  HERMES_HOME       Hermes home (default ~/.hermes)
  HERMES_PYTHON     Hermes's Python, if auto-discovery picks the wrong one
  CORTEX_PYTHON     Python (>=3.10) used to build the Claude Code venv
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) AGENT="${2:-}"; shift 2 ;;
    --agent=*) AGENT="${1#*=}"; shift ;;
    --detect) DETECT_ONLY=1; shift ;;
    --enable-jev) JEV=on; shift ;;
    --no-jev|--disable-jev) JEV=off; shift ;;
    --enable-auto-judge) AUTO_JUDGE=on; shift ;;
    --no-auto-judge|--disable-auto-judge) AUTO_JUDGE=off; shift ;;
    --no-hooks) WITH_HOOKS=0; shift ;;
    --no-claude-md) WITH_CLAUDE_MD=0; shift ;;
    --no-model) WITH_MODEL=0; shift ;;
    --no-verify) WITH_VERIFY=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ detection
DETECT_REASON=""
detect_agent() {
  local running_claude=0 running_hermes=0 has_claude=0 has_hermes=0
  [[ "${CLAUDECODE:-}" == "1" || -n "${CLAUDE_CODE_ENTRYPOINT:-}" ]] && running_claude=1
  [[ -n "${HERMES_SESSION_ID:-}${HERMES_CWD:-}${HERMES_HOME:-}" ]] && running_hermes=1
  command -v claude >/dev/null 2>&1 && has_claude=1
  if [[ -n "${HERMES_HOME:-}" && -d "${HERMES_HOME}" ]] || [[ -d "$HOME/.hermes" ]] || command -v hermes >/dev/null 2>&1; then
    has_hermes=1
  fi
  DETECT_SIGNALS="CLAUDECODE=${CLAUDECODE:-unset} CLAUDE_CODE_ENTRYPOINT=${CLAUDE_CODE_ENTRYPOINT:-unset} claude-cli=$has_claude"
  DETECT_SIGNALS+=" HERMES_HOME=${HERMES_HOME:-unset} HERMES_SESSION_ID=${HERMES_SESSION_ID:+set} hermes-install=$has_hermes"
  if [[ -n "$AGENT" ]]; then
    DETECT_REASON="--agent $AGENT given"
  elif [[ $running_claude == 1 ]]; then
    AGENT=claude; DETECT_REASON="this script is running inside Claude Code (CLAUDECODE/CLAUDE_CODE_ENTRYPOINT)"
  elif [[ $running_hermes == 1 ]]; then
    AGENT=hermes; DETECT_REASON="this script is running inside Hermes (HERMES_SESSION_ID/HERMES_CWD/HERMES_HOME)"
  elif [[ $has_claude == 1 && $has_hermes == 0 ]]; then
    AGENT=claude; DETECT_REASON="only Claude Code is installed (claude CLI on PATH)"
  elif [[ $has_hermes == 1 && $has_claude == 0 ]]; then
    AGENT=hermes; DETECT_REASON="only Hermes is installed (~/.hermes or hermes on PATH)"
  elif [[ $has_hermes == 1 && $has_claude == 1 ]]; then
    die "both Claude Code and Hermes are installed and neither is running this script. Re-run with --agent claude or --agent hermes. ($DETECT_SIGNALS)"
  else
    die "found neither Claude Code (claude CLI) nor Hermes (~/.hermes, hermes). Install your agent first, or pass --agent. ($DETECT_SIGNALS)"
  fi
  case "$AGENT" in claude|claude-code) AGENT=claude ;; hermes) ;; *) die "--agent must be claude or hermes" ;; esac
}

detect_agent
say "Agent: $AGENT ($DETECT_REASON)"
echo "    signals: $DETECT_SIGNALS"
[[ $DETECT_ONLY == 1 ]] && exit 0

# ------------------------------------------------------------------ helpers
pick_python() {
  local candidate
  for candidate in "${CORTEX_PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3; do
    [[ -z "$candidate" ]] && continue
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then
      command -v "$candidate"; return 0
    fi
  done
  return 1
}

download_model() {  # $1 python, $2 dir
  [[ $WITH_MODEL == 1 ]] || { warn "--no-model: semantic recall disabled (lexical recall still works)"; return 0; }
  say "Downloading/verifying the local embedding model (~67 MB, SHA-256 pinned) -> $2"
  "$1" "$REPO_DIR/scripts/download_embedding_model.py" --output "$2"
}

# ------------------------------------------------------------------ Claude Code
install_claude() {
  local home="${CORTEX_HOME:-$HOME/.cortex}"
  local venv="$home/venv" py
  mkdir -p "$home" && chmod 700 "$home"
  py="$(pick_python)" || die "Python 3.10+ is required (set CORTEX_PYTHON=/path/to/python3)"
  if [[ ! -x "$venv/bin/python" ]]; then
    say "Creating venv $venv (with $py)"
    "$py" -m venv "$venv" || die "python venv failed; on Debian/Ubuntu: sudo apt install python3-venv"
  fi
  local vpy="$venv/bin/python"
  say "Installing Cortex + local embedding runtime into the venv"
  "$vpy" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
  if [[ $WITH_MODEL == 1 ]]; then
    "$vpy" -m pip install --quiet --upgrade --force-reinstall --no-deps "$REPO_DIR" >/dev/null
    "$vpy" -m pip install --quiet --upgrade "$REPO_DIR[embeddings]"
  else
    "$vpy" -m pip install --quiet --upgrade --force-reinstall --no-deps "$REPO_DIR"
  fi
  rm -rf "$REPO_DIR/build" "$REPO_DIR"/*.egg-info 2>/dev/null || true
  download_model "$vpy" "$home/models/bge-small-en-v1.5"

  say "Registering the cortex MCP server with Claude Code (user scope)"
  if command -v claude >/dev/null 2>&1; then
    claude mcp remove -s user cortex >/dev/null 2>&1 || true
    claude mcp add -s user cortex -e "CORTEX_HOME=$home" -- "$vpy" -m cortex.mcp_server
  else
    warn "claude CLI not on PATH; writing ~/.claude.json directly"
    "$vpy" - "$HOME/.claude.json" "$vpy" "$home" <<'PY'
import json, sys, pathlib
path, py, home = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
data = json.loads(path.read_text()) if path.exists() else {}
data.setdefault("mcpServers", {})["cortex"] = {"type": "stdio", "command": py,
    "args": ["-m", "cortex.mcp_server"], "env": {"CORTEX_HOME": home}}
path.write_text(json.dumps(data, indent=2)); path.chmod(0o600)
PY
  fi

  local settings="$HOME/.claude/settings.json"
  mkdir -p "$HOME/.claude"
  say "$([[ $WITH_HOOKS == 1 ]] && echo Installing || echo Removing) Cortex hooks in $settings"
  "$vpy" - "$settings" "$vpy" "$home" "$WITH_HOOKS" <<'PY'
import json, shutil, sys, time, pathlib
path, py, home, enabled = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
data = json.loads(path.read_text() or "{}") if path.exists() else {}
if path.exists():
    shutil.copy2(path, path.with_name(f"settings.json.bak-cortex-{time.strftime('%Y%m%d%H%M%S')}"))
hooks = data.setdefault("hooks", {})
def strip(event):
    groups = []
    for group in hooks.get(event, []):
        kept = [h for h in group.get("hooks", []) if "cortex.mcp_server hook" not in str(h.get("command", ""))]
        if kept:
            groups.append({**group, "hooks": kept})
    if groups: hooks[event] = groups
    else: hooks.pop(event, None)
for event in ("SessionStart", "UserPromptSubmit", "Stop"):
    strip(event)
if enabled:
    def cmd(kind): return f'CORTEX_HOME="{home}" "{py}" -m cortex.mcp_server hook {kind}'
    hooks.setdefault("SessionStart", []).append({"matcher": "startup|resume|clear|compact",
        "hooks": [{"type": "command", "command": cmd("session-start"), "timeout": 20}]})
    hooks.setdefault("UserPromptSubmit", []).append({"hooks": [{"type": "command", "command": cmd("prompt"), "timeout": 15}]})
    hooks.setdefault("Stop", []).append({"hooks": [{"type": "command", "command": cmd("stop"), "timeout": 15}]})
if not hooks: data.pop("hooks", None)
path.write_text(json.dumps(data, indent=2) + "\n")
PY

  if [[ $WITH_CLAUDE_MD == 1 ]]; then
    say "Adding the Cortex memory instructions to ~/.claude/CLAUDE.md"
    "$vpy" - "$HOME/.claude/CLAUDE.md" "$REPO_DIR/docs/claude-memory-block.md" <<'PY'
import re, sys, pathlib
path, block_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
block = block_path.read_text().strip()
text = path.read_text() if path.exists() else ""
pattern = re.compile(r"<!-- cortex:begin -->.*?<!-- cortex:end -->", re.S)
text = pattern.sub(block, text) if pattern.search(text) else (text.rstrip() + "\n\n" + block + "\n").lstrip()
path.write_text(text)
PY
  fi

  if [[ -n "$JEV" || -n "$AUTO_JUDGE" ]]; then
    warn "Jev and the auto-judge only review Hermes's staged memory proposals. The Claude Code MCP path writes memories directly, so there is nothing for them to do; the flag was recorded but changes nothing here."
  fi

  if [[ $WITH_VERIFY == 1 ]]; then
    say "Verifying: JSON-RPC smoke test of every MCP tool (temporary database)"
    CORTEX_HOME="$home" "$vpy" "$REPO_DIR/scripts/mcp_smoke_test.py" --python "$vpy"
    say "Verifying: health of the real store"
    CORTEX_HOME="$home" CORTEX_MCP_QUIET=1 "$vpy" -m cortex.mcp_server doctor
    if command -v claude >/dev/null 2>&1; then
      claude mcp get cortex 2>/dev/null | sed -n '1,12p' || true
    fi
  fi
  cat <<DONE

Cortex is installed for Claude Code.
  data:     $home/cortex.db   (back up with: $vpy -m cortex.mcp_server backup)
  server:   $vpy -m cortex.mcp_server   (registered as MCP server "cortex", user scope)
  hooks:    $([[ $WITH_HOOKS == 1 ]] && echo "SessionStart brief, UserPromptSubmit recall, Stop episode log" || echo "not installed")
Start a NEW Claude Code session (or run /mcp) to load it. Re-run this script any time to upgrade.
DONE
}

# ------------------------------------------------------------------ Hermes
hermes_bin() {
  local candidate
  for candidate in "$(command -v hermes 2>/dev/null || true)" "$HOME/.local/bin/hermes" \
                   "$HERMES_HOME/hermes-agent/.hermes/bin/hermes" "$HERMES_HOME/hermes-agent/venv/bin/hermes"; do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

# shellcheck source=scripts/hermes_python.sh
source "$REPO_DIR/scripts/hermes_python.sh"
find_hermes_python() { cortex_find_hermes_python "${HERMES_PYTHON:-}"; }

find_uv() {
  local candidate
  for candidate in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HERMES_HOME"/tools/uv-*/uv; do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

pip_into() {  # $1 python, rest: packages
  local py="$1" uv; shift
  if "$py" -m pip --version >/dev/null 2>&1; then
    "$py" -m pip install --quiet "$@"
  elif uv="$(find_uv)"; then
    "$uv" pip install --quiet --python "$py" "$@"
  else
    "$py" -m ensurepip --upgrade >/dev/null 2>&1 && "$py" -m pip install --quiet "$@"
  fi
}

set_env_line() {  # $1 file, $2 key, $3 value
  local file="$1" key="$2" value="$3"
  mkdir -p "$(dirname "$file")"; touch "$file"; chmod 600 "$file"
  if grep -q "^$key=" "$file"; then sed -i "s|^$key=.*|$key=$value|" "$file"; else echo "$key=$value" >> "$file"; fi
}

install_hermes() {
  export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
  [[ -d "$HERMES_HOME" ]] || die "HERMES_HOME=$HERMES_HOME does not exist. Install Hermes first (https://github.com/NousResearch/hermes-agent)."
  local hpy
  hpy="$(find_hermes_python)" || die "could not find Hermes's Python. Set HERMES_PYTHON=/path/to/hermes/venv/bin/python and re-run."
  "$hpy" -c 'import importlib.util as u, sys; sys.exit(not (u.find_spec("yaml") or u.find_spec("ruamel")))' 2>/dev/null || die "$hpy is not Hermes's Python (no PyYAML). Set HERMES_PYTHON=/path/to/hermes/python."
  say "Hermes home: $HERMES_HOME"
  say "Hermes Python: $hpy"

  if [[ $WITH_MODEL == 1 ]]; then
    # Sidecar dir, not Hermes's managed env: 'hermes update' / 'hermes pm repair'
    # rebuild that env and would silently drop these (see runtime_path.py).
    local pyver pydeps
    pyver="$("$hpy" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    pydeps="${CORTEX_PYDEPS:-$HERMES_HOME/cortex/pydeps/py$pyver}"
    if PYTHONPATH="$pydeps" "$hpy" -c 'import numpy, onnxruntime, tokenizers' >/dev/null 2>&1; then
      say "Local embedding runtime already present ($pydeps)"
    else
      say "Installing the local embedding runtime (numpy, onnxruntime, tokenizers) -> $pydeps"
      mkdir -p "$pydeps"
      pip_into "$hpy" --target "$pydeps" "numpy>=1.24,<3" "onnxruntime>=1.17,<2" "tokenizers>=0.15,<1" \
        || die "could not install the embedding runtime for $hpy"
    fi
  fi

  say "Copying the Cortex plugin into $HERMES_HOME/plugins/cortex"
  HERMES_HOME="$HERMES_HOME" bash "$REPO_DIR/scripts/install_local.sh" | sed 's/^/    /'
  download_model "$hpy" "$HERMES_HOME/models/bge-small-en-v1.5"

  say "Selecting Cortex as the Hermes memory provider (fast recall on)"
  local semantic=10; [[ $WITH_MODEL == 1 ]] || semantic=0
  PYTHONPATH="$HERMES_HOME/plugins" "$hpy" - "$HERMES_HOME/config.yaml" "$semantic" <<'PY'
import shutil, sys, time, pathlib
from cortex.hermes_config import _load_yaml, write_cortex_config
path, semantic = pathlib.Path(sys.argv[1]), float(sys.argv[2])
before = (_load_yaml(path.read_text(encoding="utf-8-sig")) or {}) if path.exists() else {}
if path.exists():
    shutil.copy2(path, path.with_name(f"config.yaml.bak-cortex-{time.strftime('%Y%m%d%H%M%S')}"))
previous = (before.get("memory") or {}).get("provider")
# setdefault semantics: an explicit operator choice is never overwritten.
section = write_cortex_config(path, {"background_preload": True, "query_cache_ttl_seconds": 45,
                                     "semantic_fusion_weight": semantic}, provider="cortex", only_missing=True)
print(f"    memory.provider: {previous!r} -> 'cortex'; plugins.cortex = {section}")
PY

  local hb_enable
  if hb_enable="$(hermes_bin)"; then
    # Hermes lists newly found user plugins under plugins.disabled until enabled.
    say "Enabling the cortex plugin in Hermes"
    HERMES_HOME="$HERMES_HOME" timeout 180 "$hb_enable" plugins enable cortex 2>&1 | tail -n 2 | sed 's/^/    /' \
      || warn "'hermes plugins enable cortex' failed; run it manually"
  fi

  # Optional remote components: strictly opt-in, never inferred from credentials.
  local env_file="$HERMES_HOME/cortex/auto-judge.env"
  if [[ "$JEV" == on ]]; then AUTO_JUDGE=on; fi
  if [[ "$AUTO_JUDGE" == on ]]; then
    say "Enabling the remote auto-judge timer (explicit --enable flag given)"
    if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
      CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT=1 CORTEX_PYTHON_BIN="$hpy" HERMES_HOME="$HERMES_HOME" \
        CORTEX_PLUGIN_DIR="$HERMES_HOME/plugins/cortex" bash "$HERMES_HOME/plugins/cortex/scripts/install_auto_judge_timer.sh"
    else
      set_env_line "$env_file" CORTEX_AUTO_JUDGE_ENABLED 1
      warn "no user systemd session, so no timer was installed. Settings were written to $env_file;"
      warn "schedule it yourself, e.g. cron: */5 * * * * set -a; . $env_file; PYTHONPATH=$HERMES_HOME/plugins $hpy -m cortex auto-judge --quiet"
    fi
  elif [[ "$AUTO_JUDGE" == off ]]; then
    if command -v systemctl >/dev/null 2>&1; then systemctl --user disable --now cortex-auto-judge.timer >/dev/null 2>&1 || true; fi
    [[ -f "$env_file" ]] && set_env_line "$env_file" CORTEX_AUTO_JUDGE_ENABLED 0
    say "Auto-judge disabled"
  fi
  if [[ "$JEV" == on ]]; then
    set_env_line "$env_file" CORTEX_JEV_ENABLED 1
    say "Jev enabled (CORTEX_JEV_ENABLED=1 in $env_file). Put TYPESAFE_API_KEY in $HERMES_HOME/.env."
  elif [[ "$JEV" == off ]]; then
    [[ -f "$env_file" ]] && set_env_line "$env_file" CORTEX_JEV_ENABLED 0
    say "Jev disabled"
  fi

  if [[ $WITH_MODEL == 1 && -f "$HERMES_HOME/cortex/cortex.db" ]]; then
    say "Backfilling embeddings for existing memories (safe to re-run)"
    PYTHONPATH="$HERMES_HOME/plugins" "$hpy" "$REPO_DIR/scripts/backfill_embeddings.py" \
      --db "$HERMES_HOME/cortex/cortex.db" --model-dir "$HERMES_HOME/models/bge-small-en-v1.5" 2>&1 | tail -n 3 | sed 's/^/    /' || warn "backfill skipped"
  fi

  if [[ $WITH_VERIFY == 1 ]]; then
    say "Verifying through Hermes's own plugin loader"
    local hermes_src=""
    if [[ -d "$HERMES_HOME/hermes-agent/hermes_cli" ]]; then hermes_src="$HERMES_HOME/hermes-agent"
    else hermes_src="$("$hpy" -c 'import hermes_cli, pathlib; print(pathlib.Path(hermes_cli.__file__).resolve().parents[1])' 2>/dev/null || true)"; fi
    (cd "${hermes_src:-$HERMES_HOME}" && HERMES_HOME="$HERMES_HOME" PYTHONPATH="${hermes_src:+$hermes_src:}$HERMES_HOME/plugins" \
       "$hpy" "$REPO_DIR/scripts/verify_hermes_install.py" --write-test)
    local hb
    if hb="$(hermes_bin)"; then
      say "hermes memory status"
      HERMES_HOME="$HERMES_HOME" timeout 90 "$hb" memory status 2>&1 | sed -n '1,20p' | sed 's/^/    /' || true
    fi
  fi
  cat <<DONE

Cortex is installed for Hermes.
  data:    $HERMES_HOME/cortex/cortex.db
  config:  memory.provider=cortex, plugins.cortex.background_preload=true$([[ $WITH_MODEL == 1 ]] && echo ", semantic_fusion_weight=10")
  remote:  auto-judge $([[ "$AUTO_JUDGE" == on ]] && echo ON || echo off), Jev $([[ "$JEV" == on ]] && echo ON || echo off)
Restart Hermes (start a new session, or 'hermes gateway restart' if you run the gateway) to load it.
Re-run this script any time to upgrade.
DONE
}

case "$AGENT" in
  claude) install_claude ;;
  hermes) install_hermes ;;
esac
