"""Fast, deterministic recall planning for Cortex.

The planner is deliberately conservative: it avoids an LLM call before the
real LLM call, but still varies memory depth and context budget by task.  It is
an engineering analogue of attentional gating, not a model of a brain region.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


_GREETING_ONLY = re.compile(
    r"^\s*(?:hi|hey|hello|thanks|thank you|good (?:morning|afternoon|evening|night)|"
    r"ok(?:ay)?|sounds good|got it|understood|cool|nice)[.! ]*\s*$",
    re.I,
)
_STATELESS = re.compile(
    r"^\s*(?:calculate|compute|convert|translate)\b|^\s*(?:summarize|rewrite|proofread)\s+(?:this|the following)\b",
    re.I,
)
_PERSONAL = re.compile(
    r"\b(?:my|our|i\s+(?:prefer|said|asked|decided)|remember|memory|previous|before|again|usual|favorite|"
    r"preference|profile|vault|assistant|agent|hermes|cortex)\b",
    re.I,
)
_DECISION = re.compile(
    r"\b(?:decide|decision|approved|chosen|chose|choice|should\s+(?:we|i)|which\s+.+\s+(?:use|used)|backend|"
    r"setting|configuration|procedure|workflow)\b",
    re.I,
)
_TOOL = re.compile(
    r"\b(?:tool|call|command|shell|terminal|build|test|deploy|install|research|search|browser|filesystem|"
    r"github|calendar|email|server|ssh|api|mcp)\b",
    re.I,
)
_MULTI_HOP = re.compile(
    r"\b(?:why|compare|relationship|related|connect|combine|both|across|root cause|what changed|how did|timeline)\b",
    re.I,
)
_CURRENT = re.compile(r"\b(?:current|currently|now|today|latest|present|still|active)\b", re.I)
_HISTORICAL = re.compile(
    r"\b(?:previously|formerly|historical|history|back then|last (?:week|month|year)|used to|during)\b", re.I
)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")
_ARITHMETIC_ONLY = re.compile(
    r"^\s*(?:(?:what(?:'s| is)|solve|evaluate)\s+)?(?:\(?\s*-?\d+(?:\.\d+)?\s*\)?\s*)"
    r"(?:[+\-*/%^]|\*\*)\s*(?:\(?\s*-?\d+(?:\.\d+)?\s*\)?\s*)"
    r"(?:(?:[+\-*/%^]|\*\*)\s*(?:\(?\s*-?\d+(?:\.\d+)?\s*\)?\s*))*[?=.! ]*$",
    re.I,
)
_EXPLICIT_ATTENTION = re.compile(
    r"\b(?:pay close attention to|focus closely on|use detailed memory (?:for|about))\b",
    re.I,
)
# Query classes (#5). The planner already varied depth by *cue*; this names the kind of
# question, because a lookup and a planning turn want opposite things from memory: a lookup
# wants one precise fact and tolerates a narrow threshold, planning wants principles and
# tolerates almost nothing irrelevant.
_LOOKUP = re.compile(
    r"^\s*(?:what(?:'s| is| are| was)|where(?:'s| is| are)|who(?:'s| is| are)|when (?:is|was|did)|"
    r"which (?:port|path|file|ip|host|token|version|name|url|command)|how (?:many|much|long|far)|"
    r"(?:give me|show me|find|look up|what value))\b",
    re.I,
)
_PLANNING = re.compile(
    r"\b(?:should (?:we|i)|next step|what'?s next|plan|roadmap|priorit\w*|strategy|approach|"
    r"how (?:do|should) (?:we|i)|what would you|options|trade-?offs?|design)\b",
    re.I,
)
_CHITCHAT = re.compile(
    r"\b(?:thanks|thank you|nice|cool|great|lol|haha|good (?:morning|night)|how are you|"
    r"you around|just checking)\b",
    re.I,
)
CLASS_STRATEGIES: dict[str, dict[str, Any]] = {
    # A lookup wants the fact, so widen the net slightly and lower the bar a little.
    "lookup": {"limit": 6, "token_budget": 560, "threshold_delta": -0.02, "graph_depth": 1},
    # Planning wants principles and prior decisions, compact and high-precision.
    "planning": {"limit": 4, "token_budget": 520, "threshold_delta": 0.02, "graph_depth": 1},
    "recall": {"limit": 6, "token_budget": 620, "threshold_delta": 0.0, "graph_depth": 1},
    "chitchat": {"limit": 0, "token_budget": 0, "threshold_delta": 0.0, "graph_depth": 0},
}

_TOPIC_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9._-]{2,31}")
_TOPIC_STOP = {
    "about",
    "again",
    "and",
    "are",
    "assistant",
    "attention",
    "before",
    "close",
    "closely",
    "could",
    "current",
    "detailed",
    "does",
    "during",
    "focus",
    "for",
    "from",
    "have",
    "help",
    "how",
    "into",
    "latest",
    "memory",
    "need",
    "pay",
    "please",
    "remember",
    "should",
    "that",
    "the",
    "this",
    "use",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}

ATTENTION_POLICY_VERSION = "topic_salience_shadow_v1"


@dataclass(frozen=True)
class RecallPlan:
    mode: str
    needs_memory: bool
    limit: int
    token_budget: int
    threshold: float
    reason: str
    temporal_mode: str = "current"
    as_of: str | None = None
    graph_depth: int = 1
    tool_limit: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def query_class(query: str) -> str:
    """Name the kind of turn: lookup, planning, chitchat or recall.

    Deterministic and cheap — same spirit as the rest of the planner, no model call.
    """
    text = " ".join((query or "").split())
    if not text:
        return "recall"
    if _GREETING_ONLY.match(text):
        return "chitchat"
    # Planning is checked before lookup: "what should we do next" starts with "what" but
    # wants principles, not a value.
    if _PLANNING.search(text):
        return "planning"
    if _LOOKUP.search(text):
        return "lookup"
    if _CHITCHAT.search(text) and len(text.split()) <= 6:
        return "chitchat"
    return "recall"


def plan_recall(
    query: str,
    *,
    max_limit: int = 6,
    max_token_budget: int = 700,
    base_threshold: float = 0.16,
) -> RecallPlan:
    """Choose the smallest safe recall plan for one query."""

    text = " ".join((query or "").split())
    if not text or _GREETING_ONLY.match(text):
        return RecallPlan("none", False, 0, 0, base_threshold, "social turn; durable recall not needed")

    personal = bool(_PERSONAL.search(text))
    decision = bool(_DECISION.search(text))
    tool = bool(_TOOL.search(text))
    multi_hop = bool(_MULTI_HOP.search(text))
    year = _YEAR.search(text)
    historical = bool(year or _HISTORICAL.search(text))
    current = bool(_CURRENT.search(text))

    if (_STATELESS.search(text) or _ARITHMETIC_ONLY.match(text)) and not (
        personal or decision or historical
    ):
        return RecallPlan("none", False, 0, 0, base_threshold, "self-contained transformation")

    temporal_mode = "historical" if historical else "current"
    as_of = _year_end_iso(int(year.group(1))) if year else None

    if _EXPLICIT_ATTENTION.search(text):
        return RecallPlan(
            "deep",
            True,
            min(max_limit, 6),
            min(max_token_budget, 680),
            max(0.08, base_threshold - 0.02),
            "explicit user request for close attention",
            temporal_mode,
            as_of,
            graph_depth=2,
            tool_limit=3 if tool else 0,
        )
    if multi_hop or (historical and decision):
        return RecallPlan(
            "deep",
            True,
            min(max_limit, 6),
            min(max_token_budget, 680),
            max(0.08, base_threshold - 0.02),
            "multi-memory or temporal reasoning",
            temporal_mode,
            as_of,
            graph_depth=2,
            tool_limit=3 if tool else 0,
        )
    if personal or decision or historical or current:
        return RecallPlan(
            "focused",
            True,
            min(max_limit, 6),
            min(max_token_budget, 620),
            base_threshold,
            "explicit durable-context cue",
            temporal_mode,
            as_of,
            graph_depth=1,
            tool_limit=2 if tool else 0,
        )
    if tool:
        return RecallPlan(
            "procedural",
            True,
            min(max_limit, 6),
            min(max_token_budget, 620),
            min(0.30, base_threshold + 0.02),
            "tool task; prefer compact procedural evidence",
            temporal_mode,
            as_of,
            graph_depth=1,
            tool_limit=3,
        )
    # No durable-context cue fired, so let the *kind* of question decide. A lookup still
    # wants its fact; a planning turn wants principles and should not pay for six memories.
    klass = query_class(text)
    if klass == "chitchat":
        return RecallPlan(
            "none",
            False,
            0,
            0,
            base_threshold,
            "chit-chat; durable recall not needed",
            temporal_mode,
            as_of,
        )
    strategy = CLASS_STRATEGIES[klass]
    return RecallPlan(
        "lean" if klass == "recall" else klass,
        True,
        min(max_limit, int(strategy["limit"])),
        min(max_token_budget, int(strategy["token_budget"])),
        min(0.30, max(0.08, base_threshold + float(strategy["threshold_delta"]))),
        f"{klass} turn; class-specific recall plan",
        temporal_mode,
        as_of,
        graph_depth=int(strategy["graph_depth"]),
    )


def attention_topics(query: str, *, limit: int = 6) -> tuple[str, ...]:
    """Extract a small, local-only set of deterministic topic cues.

    These keys are used only after a resolved retrieval outcome. Common prompt
    words are excluded so a generic instruction cannot become learned salience.
    """

    bounded = max(1, min(12, int(limit)))
    topics: list[str] = []
    for match in _TOPIC_TOKEN.finditer(query or ""):
        topic = match.group(0).casefold().strip("._-")
        if (
            len(topic) < 3
            or topic in _TOPIC_STOP
            or topic.isdigit()
            or topic in topics
        ):
            continue
        topics.append(topic)
        if len(topics) >= bounded:
            break
    return tuple(topics)


def _year_end_iso(year: int) -> str:
    now = datetime.now(timezone.utc)
    if year == now.year:
        return now.isoformat(timespec="seconds")
    return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).isoformat()
