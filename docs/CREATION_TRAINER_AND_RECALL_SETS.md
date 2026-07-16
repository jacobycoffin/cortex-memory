# Creation Trainer and recall sets

This document describes the boundary Cortex is moving toward. It is both a
product explanation and an implementation contract.

## The simple model

```text
Something happened
        |
        v
Episode or source evidence
        |
        v
Creation candidate  -- not recallable
        |
        | operator review
        v
Approved memory     -- eligible for a recall set
        |
        | retrieval test and outcome feedback
        v
Trusted behavior
        |
        | evidence-backed relationship proposal
        v
Approved connection
```

Cortex must not treat these objects as interchangeable:

- An **episode** records that something happened. It is history, not
  automatically a durable fact.
- A **source** preserves where a statement came from. It is evidence, not
  automatically something Kaya should inject into an unrelated answer.
- A **creation candidate** is Cortex asking whether a statement is worth
  remembering. Candidates live outside the retrieval index.
- An **approved memory** is a concise, reusable statement the operator has
  allowed into a recall set.
- A **connection** is an explained relationship between two approved records.
  It must say what the relationship is and cite the evidence for it.

## Provenance words

The system should use provenance labels literally:

- `USER_STATED`: the user happened to say it in conversation.
- `USER_EXPLICIT`: the user explicitly asked Cortex to remember it.
- `AGENT_PROPOSED`: an agent or extractor believes it may be reusable.
- `OPERATOR_APPROVED`: the dashboard operator confirmed it through review.

`USER_STATED` is not an alias for `USER_EXPLICIT`. Automatic capture must not
upgrade either a user sentence or an agent summary into approved knowledge.

## Creation Trainer decisions

Every card should answer four questions before asking for a decision:

1. What happened and where did this come from?
2. What exact sentence does Cortex want to remember?
3. Why did Cortex notice it?
4. Where would it be allowed to appear during recall?

The primary decisions are deliberately small:

- **Remember it** creates one approved memory in the active recall set.
- **Edit before remembering** creates the edited statement and retains the
  candidate as its source evidence.
- **Keep as lookup evidence** makes it available only for an explicit source,
  document, or technical lookup.
- **Don't make this a memory** rejects the candidate without deleting its
  auditable source.
- **Need more context** leaves it non-recallable and waiting.

A reason is never mandatory. Optional reason chips help Cortex learn patterns,
but the operator may always decide without finding a perfect label.

After the decision, the operator chooses its reach:

- **Just this candidate** changes only this item.
- **Teach Cortex from this example** also records typed training evidence. One
  example never changes the global admission policy immediately. Repeated
  consistent examples produce a proposed standard that must be replayed,
  observed in shadow mode, and explicitly promoted.

## Recall sets and a reversible fresh start

Lifecycle state, record role, and recall permission are different things.
Archiving changes lifecycle. Calling something a reference changes how it is
described. Neither is a dependable replacement for an explicit recall boundary.

A recall set is that boundary:

- `primary` members may participate in ordinary personal recall.
- `evidence_only` members require an explicit source or technical lookup.
- `pending` candidates never enter retrieval.
- excluded or legacy-only records stay preserved but cannot enter the active
  set through text search, semantic search, context search, graph expansion, or
  archived fallback.

Starting a trained set must not edit or delete the legacy corpus. It creates a
new set, changes one active-set pointer, and places automatic candidates in the
Creation Inbox. Promoting a useful legacy memory adds membership to the trained
set; it does not rewrite the original. Switching the active pointer back is the
rollback.

## Order of training

Train these layers in order:

1. **Creation:** should this candidate become a memory at all?
2. **Readability:** is the approved memory understandable on its own?
3. **Retrieval:** did Kaya select it for the right question?
4. **Outcome:** did using it help, mislead, or make no difference?
5. **Connections:** does a typed, evidence-backed relationship improve recall
   or explanation?

Connection training before creation training spends operator time explaining a
graph whose nodes may never have deserved to exist.

## Semantic neighborhoods

Memories are not forced into one folder. They may belong to several overlapping
neighborhoods, which Cortex can use for map layout and narrowly triggered search
expansion. A deployment procedure might belong to `Project: Cortex`,
`Service: Hermes`, `Procedures`, and `Tool use` simultaneously.

`Credential references` is a protected neighborhood beneath `Tool use`. It may
contain statements such as "The Hermes deploy credential is stored in 1Password
under Hermes VPS." It must never contain the password, token, API key, private
key, or other secret value itself. When a secret value appears in a candidate,
Cortex redacts it before proposal storage and asks the operator to edit the
candidate into a secret-manager reference. Secret values therefore do not enter
memory text, search indexes, embeddings, model context, logs, or the memory map.

Neighborhood expansion is not an unexplained graph hop. It activates only when
the request directly names the neighborhood or one of its scoped services or
projects, and active recall-set eligibility still applies.

## Non-negotiable safety rules

- Automatic capture proposes; it does not silently approve.
- Pending candidates are stored outside the retrieval index.
- Approval is one audited transaction and can be undone.
- A legacy duplicate cannot silently update or promote itself.
- Graph traversal cannot bypass recall-set eligibility.
- Archived fallback cannot bypass recall-set eligibility.
- Provider or copilot failure must leave manual review usable.
- Every receipt says whether ordinary recall changed and how to reverse it.
