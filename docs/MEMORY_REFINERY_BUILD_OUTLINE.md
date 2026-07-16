# Memory Refinery build outline

## Handoff

Build this as one integrated, testable Cortex feature on the `testing` branch. The first deployed checkpoint must make the memory system understandable without changing live retrieval behavior. Retrieval changes come later, behind measurement and promotion gates.

The user should never need to interpret raw code, tables, logs, or imported document chunks as if they were normal human-readable memories. Preserve those records as evidence, but present and govern them differently from facts, preferences, procedures, decisions, and commitments.

## Problem

The current dashboard treats several fundamentally different record types as one flat list of memories:

- durable facts, preferences, procedures, decisions, and commitments;
- imported document and vault sections;
- code, configuration, tables, diagrams, and command-heavy reference material;
- episodic observations and status-like records;
- unsupported inferred claims awaiting review.

A private live audit showed that raw document sections dominate the recallable index and that many are long or structurally dense. The raw records can still be useful evidence, but displaying them as ordinary semantic memories makes Cortex look incoherent and makes operator review unnecessarily difficult.

Do not copy private memory text, source paths, identifiers, or live-data examples into repository fixtures, documentation, screenshots, commits, or public reports.

## Product outcome

The default Cortex experience should feel like a readable knowledge base:

1. The Index opens to **Readable memories**.
2. Each card states what Kaya should remember, when it applies, why it was retained, and where it came from.
3. Raw evidence is collapsed but always available.
4. Code, tables, logs, and imported sections live under **Reference evidence** instead of masquerading as normal memories.
5. Unclear records enter a **Clarity review** workflow with reversible actions.
6. The operator can improve one record without accidentally training a general policy.
7. No migration deletes content, rewrites provenance, or changes live retrieval silently.

## Core concepts

### Record roles

Add an explicit role that is separate from memory `kind` and lifecycle `state`:

- `canonical`: a concise, independently understandable fact, preference, procedure, decision, identity item, or commitment intended for normal memory use;
- `reference`: raw or document-like evidence that may be searched when relevant but should not be presented as a normal memory;
- `event`: an episodic observation with bounded temporal value;
- `claim`: an inferred statement that requires evidence or operator confirmation before being treated as canonical.

Do not overload existing `kind`, `state`, or `source_type` to carry this meaning. A procedure can be canonical or reference evidence; an active record can still be reference evidence.

### Raw evidence versus presentation

Never replace the stored raw `content` merely to make the dashboard prettier. Keep it as the evidence and provenance record.

Introduce a derived presentation layer containing, at minimum:

- `memory_id`;
- `display_title`;
- `display_summary`;
- `applies_when`;
- `retention_reason`;
- `presentation_method` and version;
- a digest of the raw source used to produce the presentation;
- readability flags;
- timestamps.

The presentation is not additional factual evidence and must not be used to raise confidence. It should be invalidated and rebuilt when the underlying content changes.

### Source-backed promotion

Turning reference evidence into a canonical memory is a factual operation, not a display operation. It must create a preview containing the proposed concise memory and its source dependency.

- User-authored rewrites may be saved as explicit operator evidence after confirmation.
- Deterministic extraction may create a proposal, but not an automatic canonical memory.
- Provider-generated rewrites are optional, disabled by default, source-cited, and proposal-only.
- Approved canonical memories keep dependency links to the reference records that support them.
- Rejecting a proposal leaves the raw evidence unchanged.

## Data model

Implement the next additive schema migration; do not renumber an already-landed schema.

Recommended additions:

### `memories`

Add:

- `record_role TEXT NOT NULL DEFAULT 'canonical'` with a constrained set of supported values;
- `role_method TEXT NOT NULL DEFAULT 'legacy_default'`;
- `role_version TEXT`;
- `role_reviewed_at TEXT` when explicitly reviewed.

Legacy rows must initially remain behavior-compatible. The migration itself must not change retrieval ranking.

### `memory_presentations`

One current presentation per memory:

- `memory_id` primary key and foreign key;
- `display_title`;
- `display_summary`;
- `applies_when`;
- `retention_reason`;
- `readability_flags_json`;
- `presentation_method`;
- `presentation_version`;
- `source_digest`;
- `created_at` and `updated_at`.

### `memory_refinery_proposals`

Use this only for changes that affect factual memory content or role:

- proposal ID and source memory IDs;
- proposal kind: `promote`, `rewrite`, `split`, `role_change`, or `merge`;
- proposed records and dependencies as bounded JSON;
- deterministic rationale and readability evidence;
- status, actor, timestamps, and linked operator-review decision;
- before-state and undo information where an action changes an existing record.

Reuse the existing operator review ledger and lifecycle/version mechanisms wherever possible. Do not create a second unaudited mutation path.

## Deterministic classification

Create a versioned, inspectable classifier. It should output a proposed role, readability flags, and reasons. It must not call a model.

Classify as `reference` by default when the source is document extraction, including vault Markdown. Additional reference signals include:

- fenced code or code-like punctuation density;
- Markdown tables;
- configuration blocks;
- diagrams or tree layouts;
- long command sequences;
- source mirrors or verbatim sections;
- unusually long sections that contain multiple independent claims.

Classify dedicated tool execution and aggregate telemetry into the existing tool/event ledgers rather than canonical memory.

Potential canonical candidates must be independently understandable and should normally contain one durable idea. Flag, but do not silently discard:

- unresolved pronouns or missing subjects;
- missing source context;
- placeholder text;
- transient completion/status language;
- mixed unrelated claims;
- raw JSON, stack traces, or command output;
- duplicate content;
- likely contradictions;
- ambiguous temporal validity.

Explicit user memories such as identity, preference, correction, decision, and prospective commitments can remain canonical when they pass the existing safety and context checks.

## Presentation generation

Version the generator so presentations can be rebuilt safely.

### Canonical records

- Prefer a concise normalized form of the original statement.
- Preserve names, dates, version boundaries, and other anchors.
- Do not omit negation or convert uncertain language into certainty.
- Show scope and applicability separately rather than burying them in prose.

### Reference evidence

- Use the document title and section heading for `display_title`.
- Describe the evidence type without pretending to summarize facts, for example: “Configuration reference,” “Code reference,” “Operational table,” or “Document section.”
- Keep the raw body collapsed under **View raw evidence**.
- Show source path only in the authenticated detail surface.

### Events and claims

- Include the observed time or validity window.
- Clearly label claims as unconfirmed when evidence is missing.
- Do not phrase an inference as an established user fact.

The deterministic presentation generator may use headings and safe sentence boundaries. It must never invent a summary when it cannot produce one faithfully; use a transparent evidence-type description instead.

## Dashboard experience

### Index

Replace the flat default with four views:

1. **Readable memories** — canonical records, default view;
2. **Reference evidence** — documents, code, tables, configuration, and raw technical material;
3. **Needs clarity** — unresolved, mixed, contextless, unsupported, or status-like records;
4. **All records** — the complete audited index.

Show counts from full aggregates while keeping interactive rows bounded for performance.

Each readable-memory card should show:

- a clear title;
- the concise memory statement;
- type and lifecycle state;
- scope or “applies everywhere”;
- source category and observed date;
- why Cortex kept it;
- recall/use/outcome evidence;
- an affordance to open provenance and raw evidence.

Reference cards should emphasize what the source contains, not “what Kaya believes.” Raw content must be collapsed by default.

### Memory detail

Organize detail into plain-language sections:

- **What Kaya remembers**;
- **When this applies**;
- **Why it is retained**;
- **Source and evidence**;
- **Raw record**;
- **History and connections**.

Every connection must continue to display its evidence or explicitly say that preserved evidence is missing.

### Clarity review

Add a `Clarity` category to Train Kaya or a directly connected refinery inbox. Reuse the simplified progressive decision pattern:

1. choose what should happen;
2. choose where it applies;
3. optionally add a reason;
4. read a plain-language final summary;
5. confirm.

Actions should be phrased as outcomes:

- **Keep as readable memory**;
- **Keep only as reference**;
- **Rewrite clearly**;
- **Split into separate memories**;
- **Archive it**;
- **Move to trash**.

`Rewrite clearly` and `Split into separate memories` must show an editable preview and source dependencies before confirmation. Default reach remains one item. Exact-copy handling and **Teach Kaya from this** must retain their current semantics.

After confirmation, show exactly what changed and provide Undo when the underlying action supports it.

### Memory map and summaries

- Default the map to canonical memories.
- Add a visible toggle for reference evidence rather than silently removing it.
- Keep role counts separate from kind and lifecycle counts.
- Do not let presentation-only records create graph links.
- Make unexplained legacy connections visibly distinct and route them to connection review.

## Retrieval rollout

Do not combine the presentation release with an unmeasured ranking change.

### Stage 1: presentation only

- Classify and display roles.
- Preserve current retrieval eligibility and ranking.
- Log how often each role is selected, injected, used, helpful, harmful, or ignored.

### Stage 2: shadow comparison

Compare current retrieval with a proposed tiered policy:

- canonical records remain normally eligible;
- reference evidence requires stronger direct lexical, scope, entity, system, or version support;
- broad graph expansion should not pull in reference evidence without a supported path;
- event records respect temporal relevance;
- unsupported claims remain gated.

Show which results would differ without changing Kaya’s context.

### Stage 3: controlled activation

Only permit activation after representative labeled cases exist and the paired evaluation shows no unacceptable loss in answer availability, technical lookup coverage, or relevant recall. Activation must be a versioned, reversible policy with a visible rollback path.

TurboVec or another embedding backend is outside this feature. Evaluate alternate retrieval backends only after the refinery produces trustworthy labels and role-aware test cases.

## Existing-record migration

The migration must be reversible and idempotent.

1. Take a verified database and plugin backup.
2. Run a dry classification report without mutations.
3. Report aggregate proposed roles, readability flags, source categories, and bounded redacted examples privately.
4. Add schema structures without changing states, content, versions, IDs, or retrieval behavior.
5. Backfill roles and deterministic presentations.
6. Backfill defensible source context for imported document chunks from existing document metadata.
7. Preserve stable vault memory IDs and document-chunk mappings.
8. Re-run the vault indexer and prove idempotency.
9. Run the audit and compare before/after counts.
10. Keep a one-command rollback path to the prior plugin and readable database.

Do not automatically archive, tombstone, merge, split, or rewrite an existing record during backfill.

## APIs and server behavior

Prefer bounded endpoints over sending every raw record in the dashboard snapshot.

Suggested authenticated routes:

- `GET /api/refinery/summary` — role/readability aggregates and migration status;
- `GET /api/refinery/items` — bounded, filtered rows;
- `POST /api/refinery/preview` — deterministic preview for rewrite, split, or role change;
- `POST /api/refinery/action` — confirmed, audited action;
- `POST /api/refinery/undo` — reversible action rollback;
- `POST /api/refinery/rebuild-presentations` — bounded administrative rebuild with progress.

Follow existing dashboard security boundaries:

- signed authenticated session;
- same-origin request verification;
- `CORTEX_DASHBOARD_REVIEWS=1` for mutations;
- strict request-size and item-count bounds;
- one active administrative job at a time;
- no secret or raw private content in logs;
- read endpoints remain private when they expose memory data;
- no hard-delete endpoint.

## Implementation order

### Phase 1 — schema, classifier, and presentation

- additive schema migration;
- deterministic role classifier;
- deterministic presentation generator;
- backfill and dry-run reporting;
- aggregate role/readability snapshot;
- store, migration, privacy, and idempotency tests.

Checkpoint: records are classified and readable through APIs, but retrieval is unchanged.

### Phase 2 — readable dashboard and Clarity review

- Index role views and filters;
- redesigned cards and detail drawer;
- collapsed raw evidence;
- Clarity review actions, previews, confirmation, and Undo;
- map toggle and role-aware summary counts;
- tablet/mobile layouts and accessibility.

Checkpoint: the user can inspect and correct the system comfortably on the live dashboard.

### Phase 3 — future-ingestion enforcement

- vault imports default to reference evidence;
- automatic capture requires canonical clarity checks;
- telemetry stays in dedicated ledgers;
- inferred claims remain reviewable;
- stable identity and reindex behavior remain intact.

Checkpoint: new gobbledygook no longer accumulates as canonical memory.

### Phase 4 — shadow tiered retrieval

- role-aware shadow ranking;
- before/after result inspection;
- role-level attribution and outcome metrics;
- paired private evaluation;
- versioned promotion and rollback only after evidence gates pass.

Checkpoint: retrieval changes are measurable and optional, not bundled into the visual migration.

## Primary code touchpoints

- `store.py` — schema, role/presentation data, classifier output, migration, snapshots, audit, proposals, actions, and Undo;
- `vault.py` — imported-document roles, source-context backfill, stable chunk identity, and reindex idempotency;
- `retrieval.py` — shadow role-aware comparison and later versioned policy application;
- `dashboard.py` — bounded refinery routes, authentication, job progress, and error handling;
- `dashboard.html` — role views, readable cards, detail hierarchy, Clarity review, progress, and map toggle;
- `client.py` and provider integration only if role metadata must cross the harness-neutral API;
- `tests/test_store.py`, `tests/test_vault.py`, `tests/test_dashboard.py`, `tests/test_provider.py`, `tests/test_adversarial.py`, and evaluation tests;
- `README.md`, `CHANGELOG.md`, `docs/ARCHITECTURE.md`, `docs/PRIVACY.md`, `docs/ROADMAP.md`, and `docs/TESTING.md`.

## Required tests

### Classification and presentation

- code, tables, diagrams, configuration, and verbatim document sections become reference evidence;
- explicit durable user preferences and identity records remain canonical;
- transient automation status does not become canonical;
- unresolved fragments enter Needs clarity;
- deterministic presentations preserve negation, anchors, and uncertainty;
- presentation rebuilds are idempotent and invalidate on source change;
- private paths and raw content do not appear in aggregate responses or logs.

### Migration and vault behavior

- upgrading an older database preserves every memory ID, content value, state, version, dependency, and source mapping;
- rollback restores the prior readable database and plugin;
- repeated vault indexing does not create duplicate memory or presentation rows;
- changed and restored vault sections reuse stable IDs;
- removed sections retain existing reversible archive behavior;
- backfill performs no automatic archive, trash, merge, rewrite, or split.

### Review actions

- every action defaults to item-only;
- role change, rewrite, and split show a correct preview;
- reason is optional;
- exact-copy and Teach Kaya scopes remain distinct;
- source dependencies are preserved;
- Undo restores the previous state without overwriting later changes;
- unauthenticated and cross-origin mutations fail.

### Retrieval safety

- Stage 1 produces byte-for-byte equivalent selected IDs and ordering for frozen queries;
- reference classification alone does not change live retrieval;
- shadow results never affect injected context;
- technical exact-match lookups remain represented in shadow comparisons;
- a promoted role policy remains bounded, versioned, and reversible.

### Dashboard quality

- Readable memories is the default;
- raw evidence is collapsed by default but accessible;
- cards remain readable on desktop, tablet, and phone;
- full aggregate counts do not use the bounded detail row list;
- keyboard navigation, focus state, labels, and confirmation work;
- no horizontal overflow or console errors;
- empty, loading, failure, read-only, migration, and job-progress states are understandable.

## Acceptance criteria

The feature is ready for user testing when all of the following are true:

1. A normal Index visit no longer presents raw code or tables as ordinary readable memories.
2. Every active/cold record has an explicit role and an explainable classification reason.
3. Every visible card has a readable title and summary or an honest reference-evidence description.
4. Raw content and provenance remain available and unchanged.
5. The user can keep, reclassify, rewrite, split, archive, trash, and undo through clear confirmed workflows.
6. The migration changes no live retrieval result in Stage 1.
7. Future vault imports no longer default to canonical semantic memory.
8. No automatic rewrite, merge, lifecycle mutation, hard deletion, or opaque provider summary occurs.
9. Full unit and adversarial tests pass, along with clean-install, upgrade, rollback, and vault-idempotency smoke tests.
10. Authenticated browser QA passes on desktop and tablet/mobile layouts.
11. The deployed dashboard serves the exact tested artifact, guided-review writes remain explicitly enabled, and unauthenticated private APIs return `401`.
12. A private before/after report records aggregate role counts and readability flags without publishing memory content or paths.

## Non-goals

- replacing the raw evidence with generated prose;
- automatically deciding which private facts are important;
- deleting or archiving existing records during migration;
- changing graph links as part of the presentation pass;
- turning every document section into an LLM-generated summary;
- enabling provider calls by default;
- deploying role-aware retrieval before paired evaluation;
- integrating TurboVec or another vector engine in this slice;
- claiming that a cleaner display proves better memory accuracy.

## Fable completion report

When handing the work back, report:

- schema and migration result;
- number of records classified by role and readability flag, without private content;
- exact retrieval-equivalence result for Stage 1;
- review and Undo flows exercised;
- test, smoke-test, and browser-QA results;
- privacy/security checks;
- deployed commit and backup path if deployment was authorized;
- a short user test path beginning with **Index → Readable memories** and **Train Kaya → Clarity**;
- any remaining blocked or intentionally deferred work, especially role-aware retrieval activation.
