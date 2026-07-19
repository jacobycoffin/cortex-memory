# Automatic memory judge

Cortex can run a silent, bounded LLM review of staged memory-creation proposals every five minutes. This is separate from the foreground Hermes turn: capture remains deterministic and non-recallable, while the timer makes an audited decision later.

## Lifecycle

1. Foreground capture sanitizes text and creates a `memory_creation_proposals` row.
2. The systemd timer is clock-aligned at five-minute boundaries.
3. Candidates wait at least 120 seconds so a following user turn can provide feedback.
4. One provider request evaluates at most 12 pending proposals.
   The oldest eligible proposals are selected first so a large backlog cannot starve them.
5. Strict validation accepts only `remember`, `evidence_only`, `reject`, `needs_context`, or `defer` for IDs in that batch.
6. Applied decisions go through `review_memory_creation`, the same reversible review ledger used by the dashboard.
   The proposal revision is compared transactionally before applying the result;
   operator decisions and changed contradiction/feedback metadata win any race.

`remember` creates a recallable memory labeled `AUTOMATIC_APPROVED` / `automatic_approved`. That label means only that an automatic judge admitted the record; it is not independent verification of the claim. The original source category remains in `origin_source_category`.

`reject` dismisses the candidate but does not hard-delete it. `needs_context` keeps the candidate outside recall. `defer`, omitted IDs, low-confidence output, malformed JSON, provider failures, and missing credentials all fail closed without changing the proposal.

## Safety boundaries

The automatic judge:

- cannot edit, merge, or invent memory text;
- cannot admit quarantined, redacted, contradiction-bearing, or blocking-quality-flag candidates;
- receives no session ID, source reference, or `source_context`; bounded entities,
  scope, preconditions, systems, and versions carry applicability instead;
- processes at most 12 proposals and bounds the complete serialized request, each
  candidate, applicability map/list entry counts, keys, and values; candidates with
  incomplete transport metadata are deferred without provider egress, and a truncated
  content excerpt cannot be admitted automatically;
- rejects oversized provider responses, embedded URL userinfo, and provider redirects;
- validates every declared runtime configuration field by exact type and safe range;
- records model, confidence, feedback adjustment, actor, reason, and review ID,
  including that review provenance on newly created recall-set membership;
- commits memory admission and its review ledger atomically, and can be reversed with the existing creation-review undo path;
- never hard-deletes memory, proposals, or review evidence;
- does not send a chat, notification, or cron delivery to the user.

The service has `StandardOutput=null`; failures remain available only in the local systemd journal.

## Positive-feedback reinforcement

When the next user turn contains unambiguous positive outcome feedback, Cortex reinforces the preceding assistant turn's still-staged proposals. Strong phrases such as “that worked perfectly” or “exactly what I wanted” receive a stronger signal than narrower outcome confirmations such as “that worked.” Generic acknowledgements such as “thanks” do not reinforce creation proposals, and user-authored candidates from the preceding turn are not credited merely because the assistant answer succeeded.

Only compact counters and an audit row are stored. Raw feedback text is not copied into the creation-feedback ledger. A strong signal raises a later `remember`/`evidence_only` confidence calculation and lowers `reject` confidence by the same amount; it never creates a memory by itself and cannot override quarantine or contradiction guards.

Defaults:

- ordinary positive feedback boost: `0.02`
- strong positive feedback boost: `0.08`
- keep threshold: `0.80`
- other-decision threshold: `0.72`

## Installation

The normal local installer copies the timer files but leaves remote automatic judging disabled. This prevents an upgrade from unexpectedly sending staged memory to a billable provider:

```bash
HERMES_HOME="$HOME/.hermes" ./scripts/install_local.sh
```

To explicitly approve OpenRouter data egress and enable the timer during installation:

```bash
CORTEX_INSTALL_AUTO_JUDGE_TIMER=1 \
CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT=1 \
HERMES_HOME="$HOME/.hermes" ./scripts/install_local.sh
```

To install or repair only the remote timer, the same consent is required:

```bash
CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT=1 \
HERMES_HOME="$HOME/.hermes" ./scripts/install_auto_judge_timer.sh
```

Loopback endpoints do not require the remote-egress consent flag. When a protected environment file already exists, consent is checked against that preserved effective endpoint rather than only the invocation's proposed endpoint. No Hermes/agent restart is performed by either installer.

The timer installer creates `~/.hermes/cortex/auto-judge.env` with mode `0600`. It stores a provider endpoint, model name, credential variable name, and credential-file path—not the credential value. The service does not import the complete Hermes `.env`; Cortex reads only the named key at runtime.

Default provider configuration:

```dotenv
CORTEX_AUTO_JUDGE_ENABLED=1
CORTEX_AUTO_JUDGE_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
CORTEX_AUTO_JUDGE_MODEL="openai/gpt-4o-mini"
CORTEX_AUTO_JUDGE_API_KEY_ENV="OPENROUTER_API_KEY"
CORTEX_AUTO_JUDGE_CREDENTIAL_FILE="/home/you/.hermes/.env"
```

Sending candidate text to a remote model is privacy-sensitive and normally billable. The consent flag is an installation-time acknowledgement, not a stored credential. Change the endpoint/model or disable the judge in `auto-judge.env` when that trade-off is not acceptable. Plain HTTP endpoints are accepted only on loopback.

## Operations

```bash
systemctl --user status cortex-auto-judge.timer
systemctl --user list-timers cortex-auto-judge.timer
journalctl --user -u cortex-auto-judge.service
```

Manual bounded run with a summary, using the installed non-secret configuration:

```bash
set -a
source "$HOME/.hermes/cortex/auto-judge.env"
set +a
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex auto-judge
```

Timer/service runs add `--quiet`. To stop automatic judgment without removing Cortex:

```bash
systemctl --user disable --now cortex-auto-judge.timer
systemctl --user stop cortex-auto-judge.service
```

The Cortex uninstaller disables the timer before stopping an active one-shot service. A failed service stop aborts removal, so the judge cannot reactivate or continue against deleted code.

## Verification

The implementation is covered by tests for admission and recall-membership provenance across trained-set transitions, exact promotion-membership undo, atomic review rollback, exact-duplicate stability across stale assessment, archive, and reopen, contradiction/truncation guards, revision-checked operator and assessment races, minimized and fully bounded nested applicability context, strict provider-output and configuration types, bounded complete requests and responses, redirect refusal, numeric configuration bounds, oldest-first backlog handling, direct schema-23/24 origin migration, negated/ambiguous feedback exclusion, clock-aligned timer artifacts, effective-endpoint data-egress consent, fail-closed uninstall ordering, quiet service behavior, install/upgrade copying, and wheel-content hygiene.
