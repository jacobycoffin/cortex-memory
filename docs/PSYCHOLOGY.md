# The psychology behind Cortex—in plain language

Cortex is inspired by memory research, but it is not a digital brain. The research gives us design questions. Tests on Hermes tell us whether the answers are useful.

## Six ideas

1. **Attention is selective.** Human working access is limited, so Cortex does not inject the entire memory database. It can inject nothing for a greeting, a small evidence set for routine work, or a larger set for personal and multi-step questions.
2. **Accessibility changes with need.** Recent and meaningfully reused memories are easier to retrieve. Cortex also considers type, source confidence, volatility, and harmful outcomes so repetition cannot automatically turn a mistake into truth.
3. **Cues activate related knowledge.** A question can match exact words, transparent concept features, or a bounded neighborhood of explicit connections.
4. **Episodes and knowledge are different.** Cortex keeps raw observations while gradually reinforcing stable facts, decisions, and procedures. It can learn a tool workflow only after it works across distinct tasks.
5. **Remembering can involve revision.** A correction creates a new version and keeps the old one. Knowledge derived from changed evidence becomes reviewable instead of silently remaining confident.
6. **Forgetting can be useful.** Low-value stale memories cool and later archive to reduce interference. They are retained, auditable, restorable, and monitored for pruning regret.

## What “self-healing” means here

It means bounded, inspectable repair:

- harmful evidence loses ranking weight;
- corrections preserve their history;
- contradictions stay visible;
- derived beliefs react to changed support;
- suspicious text is quarantined;
- duplicates can consolidate reversibly;
- stale evidence cools without being erased;
- an archived match can trigger regret detection or restoration.

It does not mean Cortex rewrites its own code or autonomously invents truths.

## The honest claim

Psychology motivates attention gating, activation, association, outcome learning, consolidation, revision, and adaptive forgetting. It does not prove those mechanisms improve an AI agent. Cortex publishes recall, token, latency, tool, lifecycle, and regret evidence so each mechanism can be compared with a simpler baseline.

For the annotated primary-source index, anatomy cautions, and research-to-feature test ledger, read [Brain and memory foundations](BRAIN_FOUNDATIONS.md).
