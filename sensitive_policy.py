"""What may be done with a memory in a sensitive domain, in one table (#184)

`consent_gate` answers the question before the row exists: may this be written at all. This module
answers the next one, and it is asked far more often: **now that it is here, what may be done with
it.** Those are different questions with different answers. Consent given to *remember* a diagnosis
is not consent to put it in a shared bundle, and consent given to recall it in this conversation is
not consent to hand it to another model. Before this module the answer was scattered across the
callers - a refusal here, a redaction there, an export path that simply did not check - and a rule
that lives in five places is a rule that holds in four.

So the policy is **data**: `BASE_POLICY` is domain → action → `Rule`, with the rest of the module
only reading it. Five decisions follow, each of which is a failure this file exists to prevent:

- **the policy is a table, not a pile of ifs.** A rule can be read, diffed and counted without
  tracing a call graph, and a domain that has no entry is a visible hole rather than a missing
  branch somebody forgot.
- **an unrecognised domain is the strictest known class, never ordinary.** Classification is
  approximate by construction (it is a keyword screen, see `consent_gate.classify`), so the failure
  mode of a *new* domain name arriving from a caller is exactly the case where being generous is
  unrecoverable. `strictest_domain()` is computed from the table, not written down twice, so it
  cannot drift from the rules it is supposed to be the strictest of.
- **a refusal names the rule, never the value.** A `Decision` holds a content hash and a class name.
  There is no field for the text, so there is no code path that can accidentally explain a refusal
  by quoting the thing it refused.
- **a decision is checkable.** `explain()` prints the class it matched, the rule id, the action, the
  version and the reason it rests on, so "why was that refused" is answerable without re-running
  anything.
- **the policy is versioned.** Rules change when we learn something, and a log of decisions made
  under yesterday's rules is only useful if it says *which* rules. Every decision carries the
  version it was made under, `policy_at()` rebuilds any past table, and `replay()` reports whether
  today's policy would still make the same call.

Actions are deliberately few, because an action name is a permission:

- `recall` — the memory is put into a prompt inside this harness.
- `derive` — a new row is made from it (summary, consolidation, lesson), i.e. a second copy exists.
- `export` — it leaves the store: a share bundle, another agent's context, a remote model.

The classes are the ones the owner named: `finance`, `health`, `credential`, `legal`, and
`third_party_personal` — data about someone who is not the person whose store this is. Work systems
are absent on purpose: `consent_gate` refuses them before the write, so a store that contains one is
already a bug this table cannot fix.

Honest limit: this is a **decision layer, not an enforcement layer**. It says what may be done and
records what was decided; a caller that never asks is not constrained by it, and nothing here can
prove a caller asked. `gate()` is the asking, and its record is the evidence.

The other limit is coverage, and it is visible by design: detection is the write path's keyword
screen, so most memories match no sensitive class at all and land in the fold, where the strictest
class refuses them. That is the correct fail-closed answer to "we cannot tell", and it is also a
measurement - `tally()` reports `folded` so the size of the gap is a number rather than an argument.
A store where the fold dominates is not ready for enforcement; it is ready for a coverage class.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from . import consent_gate

ALLOW = "allow"
ALLOW_WITH_CONSENT = "allow_with_consent"
REFUSE = "refuse"

# Ordered harshest first: used for the "unrecognised is the strictest" fold and for scoring a
# domain's overall restrictiveness. The order is the only place that ranking is written down.
RESTRICTIVENESS = {REFUSE: 2, ALLOW_WITH_CONSENT: 1, ALLOW: 0}

ACTIONS: tuple[str, ...] = ("recall", "derive", "export")

# Ordered. A credential goes first because a sentence that contains one is about the credential
# whatever else it mentions. Relationship words come last because they are usually **modifiers**,
# not topics: "my wife's diagnosis" is a health memory, and letting the relationship pattern win
# would relabel most of the store as third-party data.
DOMAINS: tuple[str, ...] = ("credential", "legal", "finance", "health", "third_party_personal")


def _rule(rule_id: str, decision: str, reason: str, basis: str) -> "Rule":
    return Rule(rule_id=rule_id, decision=decision, reason=reason, basis=basis)


@dataclass(frozen=True)
class Rule:
    """One cell of the policy: what this domain may do, by what rule, resting on what claim.

    `basis` is separate from `reason` on purpose. `reason` is what a person reads; `basis` is the
    thing the rule rests on - a stated boundary, a numbered item, a measured fact - so a rule that
    loses its basis can be found by reading the table rather than by remembering an argument.
    """

    rule_id: str
    decision: str
    reason: str
    basis: str


# --- the policy, as data --------------------------------------------------------------------------
# Version 1, the table as first written. Cells marked in OVERRIDES below were changed later; the
# base is never edited, because editing it would rewrite history for every decision already logged.
BASE_POLICY: dict[str, dict[str, Rule]] = {
    "credential": {
        "recall": _rule(
            "credential.recall.refuse.v1", REFUSE,
            "a credential is never put into a prompt, not even a part of one",
            "no credential in context (#43): a model that can read a secret can repeat it",
        ),
        "derive": _rule(
            "credential.derive.refuse.v1", REFUSE,
            "a credential is never summarised or consolidated, because the copy outlives the source",
            "#43: a redacted credential still records where the credential lives",
        ),
        "export": _rule(
            "credential.export.refuse.v1", REFUSE,
            "a credential never leaves the store",
            "#43, and the store's own rule that a credential must not be written at all",
        ),
    },
    "legal": {
        "recall": _rule(
            "legal.recall.allow.v1", ALLOW,
            "a legal note is recalled normally; the boundary is the copy, not the reading",
            "he kept the note in his own store to be told it back to him; recall into this "
            "conversation is the reason it exists",
        ),
        "derive": _rule(
            "legal.derive.allow.v1", ALLOW,
            "a legal note may be summarised like any other note",
            "v1: a summary of a note is not new legal content",
        ),
        "export": _rule(
            "legal.export.refuse.v1", REFUSE,
            "legal material does not leave the store",
            "a copy cannot be corrected; privileges do not travel with a bundle",
        ),
    },
    "finance": {
        "recall": _rule(
            "finance.recall.consent.v1", ALLOW_WITH_CONSENT,
            "finance content is recalled only with consent for this class",
            "consent is per class: authorising technical facts never authorises finance (#182)",
        ),
        "derive": _rule(
            "finance.derive.consent.v1", ALLOW_WITH_CONSENT,
            "a finance memory may be summarised only with consent, because the summary is a second copy",
            "a derived row is recalled without the consent check that produced it",
        ),
        "export": _rule(
            "finance.export.consent.v1", ALLOW_WITH_CONSENT,
            "finance content may be exported with explicit consent",
            "v1: explicit consent covers every use of the class",
        ),
    },
    "health": {
        "recall": _rule(
            "health.recall.consent.v1", ALLOW_WITH_CONSENT,
            "health content is recalled only with consent for this class",
            "consent is per class (#182); health is the class most likely to be read by someone else",
        ),
        "derive": _rule(
            "health.derive.consent.v1", ALLOW_WITH_CONSENT,
            "a health memory may be summarised only with consent",
            "a summary drops the caveats that made the original safe to read",
        ),
        "export": _rule(
            "health.export.refuse.v1", REFUSE,
            "health content does not leave the store",
            "a diagnosis in someone else's context cannot be withdrawn",
        ),
    },
    "third_party_personal": {
        "recall": _rule(
            "third_party.recall.consent.v1", ALLOW_WITH_CONSENT,
            "data about another person is recalled only with consent",
            "the person it is about is not the person who can consent",
        ),
        "derive": _rule(
            "third_party.derive.allow.v1", ALLOW,
            "a note about a third party may be summarised",
            "v1: summarising one memory does not widen who it is about",
        ),
        "export": _rule(
            "third_party.export.consent.v1", ALLOW_WITH_CONSENT,
            "data about another person may be exported with consent",
            "v1: consent covers the class",
        ),
    },
}

# Version history as deltas, not as five more copies of the table: a delta list can be read in one
# screen, and `policy_at()` replays it in order, so a past decision is re-decided under exactly the
# table that was in force. (version, domain, action, Rule).
OVERRIDES: tuple[tuple[int, str, str, Rule], ...] = (
    (
        2, "finance", "export", _rule(
            "finance.export.refuse.v2", REFUSE,
            "finance content does not leave the store",
            "found in review: consent to remember is not consent to a copy that no correction reaches",
        ),
    ),
    (
        2, "third_party_personal", "export", _rule(
            "third_party.export.refuse.v2", REFUSE,
            "data about another person does not leave the store",
            "the person it is about never consented to the copy, and cannot withdraw it",
        ),
    ),
    (
        3, "legal", "derive", _rule(
            "legal.derive.consent.v3", ALLOW_WITH_CONSENT,
            "a legal note may be summarised only with consent",
            "found in review: a summary of privileged material is privileged material",
        ),
    ),
    (
        3, "third_party_personal", "derive", _rule(
            "third_party.derive.consent.v3", ALLOW_WITH_CONSENT,
            "a note about another person may be summarised only with consent",
            "summaries are recalled as their own rows, so a second copy escapes the first one's rules",
        ),
    ),
)

CURRENT_VERSION = 3
POLICY_VERSIONS: tuple[int, ...] = tuple(range(1, CURRENT_VERSION + 1))

# Unknown *actions* are refused rather than guessed: an action the table does not know is an action
# nobody decided the rules for, and defaulting it to `recall` would silently widen a permission.
UNKNOWN_ACTION_RULE = _rule(
    "action.unknown.refuse.v1", REFUSE,
    "the requested action is not one the policy describes",
    "a permission nobody wrote down is not a permission",
)


def policy_at(version: int | None = None) -> dict[str, dict[str, Rule]]:
    """Rebuild the table as it stood at `version`. Unknown versions raise: a replay against a
    guessed policy is not a replay."""
    target = CURRENT_VERSION if version is None else int(version)
    if target not in POLICY_VERSIONS:
        raise ValueError(f"unknown policy version {target}; known: {list(POLICY_VERSIONS)}")
    table = {domain: dict(rules) for domain, rules in BASE_POLICY.items()}
    for changed_in, domain, action, rule in OVERRIDES:
        if changed_in <= target:
            table.setdefault(domain, {})[action] = rule
    return table


def _domain_key(domain: str) -> str:
    return re.sub(r"[\s-]+", "_", str(domain or "").strip().lower())


def strictest_domain(table: Mapping[str, Mapping[str, Rule]] | None = None) -> str:
    """The class every unrecognised domain is folded into.

    Computed from the table, never written down: if a later version makes some other class harsher,
    the fold follows it instead of quietly retaining today's answer. Ties break alphabetically so the
    fold is deterministic across processes (a class that changed with `hash()` would make replays
    disagree).
    """
    policy = table or policy_at()

    def score(domain: str) -> tuple[tuple[int, ...], str]:
        # Sorted decision ranks, harshest first, so a domain with more refusals outranks one with
        # more consents. The domain name is the tie-break, ascending, so the answer is stable.
        ranks = sorted((RESTRICTIVENESS.get(rule.decision, 2) for rule in policy[domain].values()),
                       reverse=True)
        return ranks, domain

    return max(sorted(policy), key=score)


# --- what a memory is about ----------------------------------------------------------------------
# One definition of a credential, shared with the gate that already screens writes. Copying the
# pattern here would give the store two opinions about what a secret looks like, and the one that
# matters is the one the write path used.
_SHARED = consent_gate.patterns()

DOMAIN_PATTERNS: dict[str, re.Pattern[str]] = {
    "credential": _SHARED["credential"],
    "legal": re.compile(
        r"\b(?:attorney|lawyer|litigation|lawsuit|subpoena|court date|custody|settlement|"
        r"non-disclosure|nda|power of attorney|will and testament|legal advice|retainer|"
        r"lease|tenancy|notaris\w*|deposition)\b",
        re.I,
    ),
    "finance": _SHARED["finance"],
    "health": _SHARED["health"],
    "third_party_personal": re.compile(
        r"\b(?:my (?:wife|husband|partner|mother|father|brother|sister|son|daughter|friend|"
        r"neighbou?r|coworker|colleague)|someone else's|their (?:address|phone|birthday|salary)|"
        r"date of birth|social security)\b",
        re.I,
    ),
}


def detect_domain(text: str) -> str:
    """The most protective domain the text matches, or `unrecognised`.

    `unrecognised` rather than `ordinary`: the caller that supplies no domain and the text that
    matches nothing are the same case, and `consent_gate`'s `technical` fallback is a statement about
    a *write* that already happened, not a licence for what to do with it afterwards. This is a
    keyword screen and therefore an upper bound on sensitivity, exactly like its source.
    """
    body = str(text or "")
    for domain in DOMAINS:
        pattern = DOMAIN_PATTERNS.get(domain)
        if pattern is not None and pattern.search(body):
            return domain
    return "unrecognised"


# --- decisions -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A decision, with everything needed to check it and nothing needed to leak it.

    There is deliberately no `text` field: a refusal is explained by its rule, and the content is
    represented by a short hash so the same memory can be recognised later without being copied.
    """

    decision: str
    domain: str  # the class the rules were read from (after folding an unknown one)
    requested_domain: str  # what the caller actually said, so a fold is visible
    action: str
    rule_id: str
    reason: str
    basis: str
    policy_version: int
    content_hash: str
    folded: bool = False
    consented: bool = False

    @property
    def refused(self) -> bool:
        return self.decision == REFUSE

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "domain": self.domain,
            "requested_domain": self.requested_domain,
            "action": self.action,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "basis": self.basis,
            "policy_version": self.policy_version,
            "content_hash": self.content_hash,
            "folded": self.folded,
            "consented": self.consented,
        }

    def explain(self) -> str:
        """One line: the verdict, the class, the action, the rule and the version. Never the value."""
        where = f"{self.domain}/{self.action}"
        if self.folded:
            where += f" (requested {self.requested_domain!r}: unrecognised, treated as the strictest class)"
        return (
            f"{self.decision} {where} by rule {self.rule_id} under policy v{self.policy_version}: "
            f"{self.reason} — rests on: {self.basis} · content sha256:{self.content_hash}"
        )


def _hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def decide(
    text: str = "",
    *,
    domain: str | None = None,
    action: str = "recall",
    consent: Mapping[str, bool] | None = None,
    version: int | None = None,
    content_hash: str | None = None,
) -> Decision:
    """Decide one (memory, action) pair.

    Order matters and is part of the contract: classify, fold, look up the rule, then - only if the
    rule is `allow_with_consent` - consult consent. A `refuse` is never upgradable by consent, so no
    caller can turn a credential into an allowed action by passing `{"credential": True}`.
    """
    table = policy_at(version)
    requested = _domain_key(domain) if domain else detect_domain(text)
    folded = requested not in table
    primary = strictest_domain(table) if folded else requested
    action_name = str(action or "").strip().lower()
    rule = table.get(primary, {}).get(action_name, UNKNOWN_ACTION_RULE)

    granted = bool((consent or {}).get(primary)) or bool((consent or {}).get(requested))
    decision = rule.decision
    reason, rule_id = rule.reason, rule.rule_id
    if decision == ALLOW_WITH_CONSENT and granted:
        # Consent upgrades the verdict but keeps the rule: the log has to show *which* rule was
        # satisfied, or "consented" and "no rule applied" look identical afterwards.
        decision = ALLOW
        reason = f"{rule.reason} (consent granted for {primary})"

    return Decision(
        decision=decision,
        domain=primary,
        requested_domain=requested,
        action=action_name,
        rule_id=rule_id,
        reason=reason,
        basis=rule.basis,
        policy_version=CURRENT_VERSION if version is None else int(version),
        content_hash=content_hash or (_hash(text) if text else ""),
        folded=folded,
        consented=granted and rule.decision == ALLOW_WITH_CONSENT,
    )


@dataclass(frozen=True)
class Replay:
    """An old decision re-decided under another version of the policy."""

    original: Decision
    replayed: Decision

    @property
    def changed(self) -> bool:
        return (self.original.decision, self.original.rule_id) != (
            self.replayed.decision, self.replayed.rule_id)

    def note(self) -> str:
        before, after = self.original, self.replayed
        if not self.changed:
            return (f"unchanged: {before.domain}/{before.action} is still {before.decision} "
                    f"under {before.rule_id} at v{after.policy_version}")
        return (f"changed: {before.domain}/{before.action} was {before.decision} under "
                f"{before.rule_id} at v{before.policy_version}, now {after.decision} under "
                f"{after.rule_id} at v{after.policy_version}")


def replay(decision: Decision, *, version: int | None = None,
           consent: Mapping[str, bool] | None = None) -> Replay:
    """Re-decide a logged decision under another policy version.

    Only the class, the action and the consent are replayed - the text was never stored, so a replay
    cannot accidentally depend on it. That is a limitation and it is the right one: a policy that
    could only be checked by keeping the secrets it refused would be a worse policy.
    """
    again = decide(
        domain=decision.domain,
        action=decision.action,
        consent=consent,
        version=version,
        content_hash=decision.content_hash,
    )
    return Replay(original=decision, replayed=again)


# --- the log -------------------------------------------------------------------------------------
TABLE = "sensitive_policy_decisions"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_schema(store: Any) -> bool:
    """The log holds the hash, the class, the rule and the version - never the content."""
    try:
        with store.transaction() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {TABLE} ("
                "content_hash TEXT NOT NULL, action TEXT NOT NULL, domain TEXT NOT NULL, "
                "decision TEXT NOT NULL, rule_id TEXT NOT NULL, policy_version INTEGER NOT NULL, "
                "decided_at TEXT NOT NULL, memory_id TEXT, "
                "PRIMARY KEY(content_hash, action))"
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def record_decision(store: Any, decision: Decision, *, memory_id: str | None = None,
                    now: datetime | None = None) -> bool:
    ensure_schema(store)
    try:
        with store.transaction() as conn:
            conn.execute(
                f"INSERT INTO {TABLE}(content_hash, action, domain, decision, rule_id, "
                "policy_version, decided_at, memory_id) VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(content_hash, action) DO UPDATE SET decision = excluded.decision, "
                "domain = excluded.domain, rule_id = excluded.rule_id, "
                "policy_version = excluded.policy_version, decided_at = excluded.decided_at, "
                "memory_id = COALESCE(excluded.memory_id, " + TABLE + ".memory_id)",
                (decision.content_hash, decision.action, decision.domain, decision.decision,
                 decision.rule_id, decision.policy_version, (now or _now()).isoformat(), memory_id),
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def gate(store: Any, text: str, *, domain: str | None = None, action: str = "recall",
         consent: Mapping[str, bool] | None = None, version: int | None = None,
         memory_id: str | None = None, now: datetime | None = None) -> Decision:
    """Decide, then log. A broken store does not change the verdict - only whether it is remembered.

    Fail-closed on the decision, open on the logging: refusing to answer because the log is
    unavailable would make a store problem look like a policy problem, and the caller would learn
    nothing. `record_decision` returns False and the decision stands.
    """
    decision = decide(text, domain=domain, action=action, consent=consent, version=version)
    record_decision(store, decision, memory_id=memory_id, now=now)
    return decision


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def history(store: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    return _rows(
        store,
        f"SELECT content_hash, action, domain, decision, rule_id, policy_version, decided_at, "
        f"memory_id FROM {TABLE} ORDER BY decided_at DESC LIMIT ?",
        (max(1, int(limit)),),
    )


def summary(store: Any) -> dict[str, Any]:
    rows = _rows(store, f"SELECT decision, COUNT(*) AS count FROM {TABLE} GROUP BY decision")
    counts = {str(row["decision"]): int(row["count"] or 0) for row in rows}
    total = sum(counts.values())
    versions = _rows(store, f"SELECT policy_version, COUNT(*) AS count FROM {TABLE} "
                            "GROUP BY policy_version ORDER BY policy_version")
    note = "the log holds content hashes, not content"
    if total == 0:
        note = ("no decisions recorded: the policy exists but nothing asks it yet, so the store's "
                "existing contents were never screened for what may be done with them")
    return {
        "decisions": total,
        "by_decision": dict(sorted(counts.items())),
        "by_version": {int(row["policy_version"]): int(row["count"] or 0) for row in versions},
        "note": note,
    }


def render(store: Any, *, limit: int = 10) -> list[str]:
    described = summary(store)
    head = f"sensitive-domain decisions: {described['decisions']} recorded"
    if described["by_decision"]:
        head += " (" + ", ".join(f"{name} {count}"
                                 for name, count in described["by_decision"].items()) + ")"
    lines = [head]
    for row in history(store, limit=limit):
        lines.append(f"  · {row['decision']} {row['domain']}/{row['action']} {row['content_hash']}"
                     f" — {row['rule_id']} v{row['policy_version']}")
    lines.append(f"  {described['note']}")
    return lines


# --- counting (used by the live measurement and by the CLI later) ---------------------------------


def tally(texts: Iterable[str], *, action: str = "recall",
          consent: Mapping[str, bool] | None = None, version: int | None = None) -> dict[str, Any]:
    """Count decisions over a corpus, without keeping a single one of its rows.

    Returns counts only: by decision, by domain, and the fold count. The docstring says so because
    the caller that measures the live store must not print what it measured - a measurement of
    sensitive content that quotes the content is the leak it was measuring.
    """
    by_decision: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    folds = 0
    total = 0
    for text in texts:
        decision = decide(text, action=action, consent=consent, version=version)
        total += 1
        by_decision[decision.decision] = by_decision.get(decision.decision, 0) + 1
        by_domain[decision.domain] = by_domain.get(decision.domain, 0) + 1
        folds += int(decision.folded)
    return {
        "scanned": total,
        "action": action,
        "policy_version": CURRENT_VERSION if version is None else int(version),
        "by_decision": dict(sorted(by_decision.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "folded": folds,
    }
