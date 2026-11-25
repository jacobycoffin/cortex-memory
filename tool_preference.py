"""Tool preference per task (#132): which tool actually works for this kind of job.

The store already records outcomes - `procedure_confidence` (#142) keeps them, `procedure_validation` (#115) gates them.
What was missing is the question a person actually asks: *for this kind of task, which tool should I reach for?*

This answers it without inventing a fourth ledger: a preference is a procedure whose id is `task::tool`, so the
outcome history, the consecutive-success gate and the failure evidence all come from the modules that already own them.

Six rules:

1. **A preference is earned here, not assumed.** No built-in defaults and no model priors: a tool is preferred because
   this store watched it work.
2. **Only validated tools are offered.** A tool with one lucky run is not a preference; it is a coincidence.
3. **A tool that fails more than it succeeds is never preferred**, even when it is the only candidate - "the only tool
   that has ever been tried" is not a recommendation.
4. **The count travels with the preference** ("`pytest -n 2`, 4 in a row, 0 recent failures"), because a preference
   without its evidence cannot be argued with.
5. **A tie returns nothing.** Two equally-good tools is a coin flip, and a coin flip dressed as guidance is worse than
   silence.
6. **Reading only.** This recommends; the caller decides, and nothing here runs a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from . import procedure_confidence, procedure_validation

# The separator cannot appear in a task name or a tool name, so a preference id can always be split back apart.
SEPARATOR = "::"

PREFERRED = "preferred"
NO_VALIDATED_TOOL = "no_validated_tool"
TIE = "tie"
NO_EVIDENCE = "no_evidence"
UNREADABLE = "unreadable"


def preference_id(task_type: str, tool: str) -> str:
    return f"{str(task_type).strip().lower()}{SEPARATOR}{str(tool).strip()}"


def split_preference_id(preference: str) -> tuple[str, str]:
    task_type, _, tool = str(preference).partition(SEPARATOR)
    return task_type, tool


@dataclass(frozen=True)
class Candidate:
    tool: str
    verdict: procedure_validation.Verdict

    @property
    def validated(self) -> bool:
        return self.verdict.may_offer

    def render(self) -> str:
        return f"{self.tool}: {self.verdict.streak} in a row, {self.verdict.failures} failure(s)"

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "streak": self.verdict.streak, "failures": self.verdict.failures,
                "verdict": self.verdict.verdict}


@dataclass
class Preference:
    task_type: str
    verdict: str
    tool: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""

    @property
    def preferred(self) -> str:
        return self.tool if self.verdict == PREFERRED else ""

    def render(self) -> str:
        if self.verdict == PREFERRED:
            chosen = next(item for item in self.candidates if item.tool == self.tool)
            return f"for {self.task_type}: prefer {self.tool} ({chosen.render()})"
        return f"for {self.task_type}: {self.verdict} — {self.reason}"

    def as_dict(self) -> dict[str, Any]:
        return {"task_type": self.task_type, "verdict": self.verdict, "tool": self.tool,
                "reason": self.reason,
                "candidates": [item.as_dict() for item in self.candidates]}


def record_outcome(store: Any, task_type: str, tool: str, outcome: str, *, note: str = "",
                   at: Any = None) -> dict[str, Any]:
    """Record that a tool was used for a task and how it went."""
    if not str(task_type or "").strip() or not str(tool or "").strip():
        return {"ok": False, "reason": "a preference needs both a task type and a tool"}
    return procedure_confidence.record(store, preference_id(task_type, tool), outcome, note=note, at=at)


def known_tools(store: Any, task_type: str, *, limit: int = 50) -> list[str]:
    """Tools with any recorded history for this task."""
    prefix = f"{str(task_type).strip().lower()}{SEPARATOR}"
    tools: list[str] = []
    for procedure in procedure_validation.known_procedures(store, limit=limit * 4):
        if procedure.startswith(prefix):
            tools.append(procedure[len(prefix):])
    return tools[:limit]


def prefer(store: Any, task_type: str, *, required: int = procedure_validation.REQUIRED_SUCCESSES) -> Preference:
    """Which tool to reach for on this task. Reads only."""
    task = str(task_type or "").strip().lower()
    if not task:
        return Preference(task_type="", verdict=NO_EVIDENCE, reason="a task type is required")
    tools = known_tools(store, task)
    if not tools:
        return Preference(task_type=task, verdict=NO_EVIDENCE,
                          reason="no tool has been recorded for this task")
    candidates: list[Candidate] = []
    for tool in tools:
        verdict = procedure_validation.validate(store, preference_id(task, tool), required=required)
        if verdict.verdict == procedure_validation.UNREADABLE:
            return Preference(task_type=task, verdict=UNREADABLE, reason=verdict.reason)
        candidates.append(Candidate(tool=tool, verdict=verdict))
    validated = [item for item in candidates if item.validated]
    if not validated:
        # Rule 3: "the only tool tried" is not a recommendation.
        best = max(candidates, key=lambda item: item.verdict.streak)
        return Preference(task_type=task, verdict=NO_VALIDATED_TOOL, candidates=candidates,
                          reason=f"no tool has {required} consecutive successes "
                                 f"(best: {best.tool} at {best.verdict.streak})")
    validated.sort(key=lambda item: (-item.verdict.streak, item.verdict.failures, item.tool))
    leader = validated[0]
    if len(validated) > 1 and validated[1].verdict.streak == leader.verdict.streak \
            and validated[1].verdict.failures == leader.verdict.failures:
        # Rule 5: a coin flip dressed as guidance is worse than silence.
        return Preference(task_type=task, verdict=TIE, candidates=candidates,
                          reason=f"{leader.tool} and {validated[1].tool} are equally supported "
                                 f"({leader.verdict.streak} in a row each)")
    return Preference(task_type=task, verdict=PREFERRED, tool=leader.tool, candidates=candidates)


def survey(store: Any, task_types: Iterable[str], *,
           required: int = procedure_validation.REQUIRED_SUCCESSES) -> list[Preference]:
    """Prefer a tool for each task, in the caller's order."""
    return [prefer(store, task, required=required) for task in task_types]
