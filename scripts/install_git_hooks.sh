#!/usr/bin/env bash
# Install the local pre-push gate for the frozen eval set.
#
# Why local and not GitHub Actions: the frozen cases are real private queries and
# memory ids. They must never sit in the repository or a CI log, so the gate runs
# where the set already lives — on the machine that owns the database.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git config core.hooksPath .githooks
chmod +x .githooks/pre-push 2>/dev/null || true
echo "core.hooksPath -> .githooks"
echo "A push now runs the frozen eval gate when ~/cortex-evaluations/frozen exists."
echo "Override for one push with: SKIP_FROZEN_GATE=1 git push"
