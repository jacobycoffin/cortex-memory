#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DIR="${CORTEX_PLUGIN_DIR:-$HERMES_HOME/plugins/cortex}"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
ENV_FILE="${CORTEX_AUTO_JUDGE_ENV_FILE:-$HERMES_HOME/cortex/auto-judge.env}"
SERVICE_NAME="cortex-auto-judge.service"
TIMER_NAME="cortex-auto-judge.timer"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_TEMPLATE="$SCRIPT_DIR/cortex-auto-judge.service.in"
TIMER_TEMPLATE="$SCRIPT_DIR/cortex-auto-judge.timer"

if [[ ! -f "$PLUGIN_DIR/__main__.py" ]]; then
  echo "Cortex plugin not found at $PLUGIN_DIR" >&2
  exit 1
fi
if [[ ! -f "$SERVICE_TEMPLATE" || ! -f "$TIMER_TEMPLATE" ]]; then
  echo "Cortex auto-judge systemd templates are missing" >&2
  exit 1
fi

if [[ -n "${CORTEX_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$CORTEX_PYTHON_BIN"
elif [[ -x "$HERMES_HOME/venv/bin/python" ]]; then
  PYTHON_BIN="$HERMES_HOME/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "python3 was not found" >&2
  exit 1
fi

ENDPOINT="${CORTEX_AUTO_JUDGE_ENDPOINT:-https://openrouter.ai/api/v1/chat/completions}"
API_KEY_ENV="${CORTEX_AUTO_JUDGE_API_KEY_ENV:-OPENROUTER_API_KEY}"
if [[ ! "$API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "CORTEX_AUTO_JUDGE_API_KEY_ENV must be an environment-variable name" >&2
  exit 1
fi
for value in \
  "$ENDPOINT" \
  "${CORTEX_AUTO_JUDGE_MODEL:-openai/gpt-4o-mini}" \
  "${CORTEX_AUTO_JUDGE_CREDENTIAL_FILE:-$HERMES_HOME/.env}"; do
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* || "$value" == *'"'* ]]; then
    echo "Auto-judge configuration values may not contain quotes or newlines" >&2
    exit 1
  fi
done

EFFECTIVE_ENDPOINT="$ENDPOINT"
if [[ -f "$ENV_FILE" ]]; then
  EXISTING_ENDPOINT="$("$PYTHON_BIN" -c 'import pathlib,sys
endpoint=""
for raw in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line=raw.strip()
    if line.startswith("export "): line=line[7:].lstrip()
    name,sep,value=line.partition("=")
    if sep and name.strip()=="CORTEX_AUTO_JUDGE_ENDPOINT":
        value=value.strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in {"\"", "'"'"'"}: value=value[1:-1]
        endpoint=value
print(endpoint)' "$ENV_FILE")"
  if [[ -n "$EXISTING_ENDPOINT" ]]; then
    EFFECTIVE_ENDPOINT="$EXISTING_ENDPOINT"
  fi
fi

HAS_URL_USERINFO="$("$PYTHON_BIN" -c 'from urllib.parse import urlparse; import sys; p=urlparse(sys.argv[1]); print("1" if p.username is not None or p.password is not None else "0")' "$EFFECTIVE_ENDPOINT")"
if [[ "$HAS_URL_USERINFO" == "1" ]]; then
  echo "CORTEX_AUTO_JUDGE_ENDPOINT must not contain URL userinfo" >&2
  exit 1
fi
IS_LOOPBACK="$("$PYTHON_BIN" -c 'from urllib.parse import urlparse; import sys; print("1" if urlparse(sys.argv[1]).hostname in {"127.0.0.1", "localhost", "::1"} else "0")' "$EFFECTIVE_ENDPOINT")"
if [[ "$IS_LOOPBACK" != "1" ]] \
  && [[ "${CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT:-0}" != "1" ]]; then
  echo "Refusing to enable remote auto-judge without explicit data-egress consent." >&2
  echo "Set CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT=1 after approving the provider and staged-memory transmission." >&2
  exit 1
fi

mkdir -p "$SYSTEMD_USER_DIR" "$(dirname -- "$ENV_FILE")"
if [[ ! -f "$ENV_FILE" ]]; then
  umask 077
  {
    echo 'CORTEX_AUTO_JUDGE_ENABLED=1'
    echo "CORTEX_AUTO_JUDGE_ENDPOINT=\"$ENDPOINT\""
    echo "CORTEX_AUTO_JUDGE_MODEL=\"${CORTEX_AUTO_JUDGE_MODEL:-openai/gpt-4o-mini}\""
    echo "CORTEX_AUTO_JUDGE_API_KEY_ENV=\"$API_KEY_ENV\""
    echo "CORTEX_AUTO_JUDGE_CREDENTIAL_FILE=\"${CORTEX_AUTO_JUDGE_CREDENTIAL_FILE:-$HERMES_HOME/.env}\""
    echo 'CORTEX_AUTO_JUDGE_MINIMUM_AGE_SECONDS=120'
    echo 'CORTEX_AUTO_JUDGE_MAX_PROPOSALS=12'
    echo 'CORTEX_AUTO_JUDGE_KEEP_THRESHOLD=0.80'
    echo 'CORTEX_AUTO_JUDGE_DECISION_THRESHOLD=0.72'
    echo 'CORTEX_AUTO_JUDGE_STRONG_FEEDBACK_BOOST=0.08'
    echo 'CORTEX_AUTO_JUDGE_POSITIVE_FEEDBACK_BOOST=0.02'
  } > "$ENV_FILE"
fi
chmod 600 "$ENV_FILE"

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

sed \
  -e "s|@@HERMES_HOME@@|$(escape_sed "$HERMES_HOME")|g" \
  -e "s|@@PYTHON_BIN@@|$(escape_sed "$PYTHON_BIN")|g" \
  -e "s|@@AUTO_JUDGE_ENV@@|$(escape_sed "$ENV_FILE")|g" \
  "$SERVICE_TEMPLATE" > "$SYSTEMD_USER_DIR/$SERVICE_NAME"
install -m 0644 "$TIMER_TEMPLATE" "$SYSTEMD_USER_DIR/$TIMER_NAME"

systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "Installed $TIMER_NAME (clock-aligned every five minutes)."
echo "Configuration: $ENV_FILE"
echo "The service is quiet; inspect failures with: journalctl --user -u $SERVICE_NAME"
