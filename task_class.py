"""Scored task classification (backlog item #218).

The old classifier was a list of single-word regexes, first-match-wins:

    for name, pattern in _TASK_RULES:
        if pattern.search(combined):
            return name

Three things were wrong with it, and only the third is obvious:

1. **Vocabulary collides.** `read` counts for filesystem, `source` for web research,
   `message` for email, `event` for calendar, `build` for shell. A single common word
   decides the class.
2. **Order decides ties.** Rules were consulted top-down, so which rule happened to sit
   above another silently resolved every overlap. Reordering the tuple would have changed
   results with no test noticing.
3. **A weak word can beat a strong phrase.** "which systemd unit runs the cortex dashboard"
   classified as `data_analysis` off the word `dashboard`, though `systemd` and `unit` are
   unambiguous shell vocabulary.

This module scores instead of matching. Every term carries a weight that says how much
evidence it is. Phrases ("pull request", "systemd unit") outweigh single words ("file",
"dashboard"), and a term that appears in more than one class is demoted automatically by
being shared, so the same word can never be decisive in one class and ignored in another.

Two rules that matter more than the weights:

- **A tie returns `general`.** Guessing a class costs real behaviour: the class select the
  recall and token allocation (#13). `general` is the honest answer when the evidence is
  split, exactly as project detection returns nothing on a tie.
- **No match returns `general` too** — the classifier is not allowed to invent a class to
  look decisive.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

GENERAL = "general"

# Below this absolute score a class has not earned the turn: the class decides recall
# candidates and the token allocation (#13), so a single generic verb is not enough.
MIN_SCORE = 1.6

# (term, weight): weight is how much evidence one occurrence is. Phrases and unambiguous
# nouns sit at 2.0+; generic verbs low enough that they cannot outvote a phrase alone.
_TERMS: dict[str, tuple[tuple[str, float], ...]] = {
    "web_research": (
        ("web search", 2.5), ("search the web", 2.5), ("search online", 2.5),
        ("latest news", 2.5), ("research", 2.0), ("website", 2.0), ("online", 2.0),
        ("googled", 2.0), ("news", 1.8),
        ("search", 1.2), ("browse", 1.2), ("source", 0.6), ("article", 1.2),
    ),
    "filesystem": (
        ("file path", 2.5), ("read the file", 2.5), ("open the file", 2.5),
        ("directory", 2.0), ("folder", 2.0), ("patch", 2.0), ("workspace", 1.8),
        ("repo", 1.8), ("checkout", 1.5),
        ("file", 1.0), ("files", 1.0), ("read", 0.8), ("write", 0.8), ("edit", 0.8),
    ),
    "shell": (
        ("systemd unit", 3.0), ("systemd", 2.5), ("journalctl", 2.5), ("crontab", 2.5),
        ("command line", 2.5), ("shell script", 2.5), ("ssh", 2.0), ("bash", 2.0),
        ("terminal", 2.0), ("apt install", 2.5), ("docker", 2.0), ("ct116", 1.5),
        ("process", 1.0), ("server", 1.2), ("service", 1.2), ("install", 1.8),
        ("lxc", 2.0),
        ("build", 1.0), ("compile", 1.5), ("unit", 1.0), ("command", 1.2),
        ("container", 2.0), ("log", 1.8), ("disk", 2.0), ("uptime", 2.5),
        ("shell", 2.0),
        ("fail2ban", 2.5), ("test suite", 2.2), ("memory usage", 2.0),
        ("test", 1.6), ("build command", 2.5),
        ("r630", 1.8), ("grafana", 1.8), ("nginx", 1.8), ("port", 1.0),
    ),
    "github": (
        ("pull request", 3.0), ("merge conflict", 3.0), ("github actions", 3.0),
        ("github", 2.5), ("commit", 2.0), ("branch", 1.8), ("workflow run", 2.0),
        ("pr", 1.8), ("issue", 1.6), ("ci", 1.5),
    ),
    "calendar": (
        ("calendar", 2.5), ("meeting", 2.2), ("appointment", 2.2), ("availability", 2.0),
        ("schedule", 1.5), ("event", 0.8), ("book a", 1.5),
    ),
    "email": (
        ("email", 2.5), ("gmail", 2.5), ("inbox", 2.5), ("forward the", 2.0),
        ("mail", 1.5), ("reply", 1.2), ("message", 0.6),
    ),
    "memory": (
        ("remember that", 3.0), ("do you remember", 3.0), ("my preference", 2.5),
        ("memory", 2.0), ("recall", 1.8), ("forget", 2.5),
        ("preference", 1.8),
    ),
    "data_analysis": (
        ("csv", 2.5), ("spreadsheet", 2.5), ("data set", 2.5), ("dataset", 2.5),
        ("correlation", 2.5), ("regression", 2.2), ("percentile", 2.5),
        ("statistics", 2.0), ("average", 1.8), ("chart", 1.8), ("plot", 1.8),
        ("analyze", 1.0), ("analyse", 1.0), ("metric", 1.2), ("metrics", 1.2),
        ("data", 0.6), ("dashboard", 0.7),
    ),
}

_COMPILED: dict[str, tuple[tuple[str, re.Pattern[str], float], ...]] = {
    task: tuple(
        (
            term,
            re.compile(rf"(?<![\w-]){re.escape(term)}(?:s|es)?(?![\w-])", re.I),
            weight,
        )
        for term, weight in terms
    )
    for task, terms in _TERMS.items()
}

# A term appearing in several classes is weaker evidence in each of them: if "message" is
# both an email word and a general noun, it cannot be decisive for email.
_SHARED_PENALTY = 0.85
_TERM_CLASSES: dict[str, int] = {}
for _task, _terms in _TERMS.items():
    for _term, _weight in _terms:
        _TERM_CLASSES[_term] = _TERM_CLASSES.get(_term, 0) + 1


@dataclass(frozen=True)
class Classification:
    task_type: str
    scores: dict[str, float]
    reason: str

    @property
    def runner_up(self) -> str | None:
        if len(self.scores) < 2:
            return None
        return sorted(self.scores, key=lambda k: self.scores[k], reverse=True)[1]


def _score(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task, patterns in _COMPILED.items():
        total = 0.0
        hits = 0
        for term, pattern, weight in patterns:
            matches = len(pattern.findall(text))
            if not matches:
                continue
            hits += matches
            shared = _SHARED_PENALTY ** (_TERM_CLASSES.get(term, 1) - 1)
            # Repeated occurrences help, but sub-linearly: a query that says "file" three
            # times is not three times as much evidence of filesystem work.
            total += weight * shared * (1 + math.log2(matches))
        if hits:
            scores[task] = round(total, 4)
    return scores


def classify(query: str, tool_name: str = "", *, margin: float = 0.30) -> Classification:
    """Classify a turn by scoring every class instead of taking the first match.

    `margin` is how far ahead the winner must be. Below it the evidence is split and the
    answer is `general`, because the class allocates recall budget and token budget (#13)
    and a wrong class spends both silently.
    """
    text = f"{query} {tool_name.replace('_', ' ')}".strip()
    scores = _score(text)
    if not scores:
        return Classification(GENERAL, scores, "no class vocabulary matched")
    # One weak word is not a classification: "build" alone scored 1.0 and made every
    # roadmap question a shell task. The weak scores stay in the record (they are
    # evidence that was considered and rejected) but cannot win.
    considered = {task: score for task, score in scores.items() if score >= MIN_SCORE}
    if not considered:
        return Classification(GENERAL, scores, f"nothing above MIN_SCORE={MIN_SCORE}")
    ranked = sorted(considered.items(), key=lambda kv: kv[1], reverse=True)
    winner, top = ranked[0]
    if len(ranked) == 1:
        return Classification(winner, scores, f"only {winner} matched ({top:.2f})")
    second, runner = ranked[1]
    # Relative margin: 8.0 vs 7.7 is a tie in practice even though the gap is 0.3.
    if top - runner < margin * max(top, 1.0):
        return Classification(
            GENERAL, scores, f"tie between {winner} ({top:.2f}) and {second} ({runner:.2f})"
        )
    return Classification(winner, scores, f"{winner} ({top:.2f}) over {second} ({runner:.2f})")
