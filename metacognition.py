"""Outcome-calibrated source monitoring for Cortex recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .retrieval import RetrievalResult


SOURCE_PRIORS = {
    "TOOL_VERIFIED": 0.92,
    "USER_EXPLICIT": 0.88,
    "DOCUMENT_EXTRACTED": 0.80,
    "REFLECTION": 0.62,
    "AGENT_INFERENCE": 0.52,
}


@dataclass(frozen=True)
class MetacognitiveAssessment:
    """A pre-outcome judgment about one candidate memory."""

    memory_id: str
    source_category: str
    raw_probability: float
    calibrated_probability: float
    decision: str
    reason: str
    calibration_scope: str
    calibration_samples: int
    features: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "source_category": self.source_category,
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "decision": self.decision,
            "reason": self.reason,
            "calibration_scope": self.calibration_scope,
            "calibration_samples": self.calibration_samples,
            "features": dict(self.features),
        }


def assess_retrieval(
    result: RetrievalResult,
    *,
    calibration: dict[str, Any] | None = None,
) -> MetacognitiveAssessment:
    """Estimate reliability from inspectable evidence, then apply learned calibration.

    This is deliberately separate from the retrieval score. Retrieval answers
    "is this relevant?" while metacognition answers "should this evidence be
    trusted without another check?".
    """

    memory = result.memory
    components = result.components
    source_category = str(memory.get("source_category") or "AGENT_INFERENCE")
    source_prior = SOURCE_PRIORS.get(source_category, 0.66)
    direct_relevance = max(
        float(components.get("lexical", 0.0)),
        float(components.get("phrase", 0.0)),
        float(components.get("semantic", 0.0)),
    )
    positive = (
        int(memory.get("success_count", 0))
        + int(memory.get("confirmed_count", 0))
        + int(memory.get("helpful_count", 0))
        + int(memory.get("validated_count", 0))
    )
    negative = int(memory.get("harmful_count", 0)) + int(memory.get("false_positive_count", 0))
    feedback_signal = (positive + 2.0) / (positive + negative + 4.0)
    stale_risk = max(0.0, min(1.0, float(components.get("stale_risk", 0.0))))
    wrong_rate = max(0.0, min(1.0, float(components.get("wrong_rate", 0.0))))

    features = {
        "source_prior": round(source_prior, 6),
        "memory_confidence": round(float(memory.get("confidence", 0.6)), 6),
        "trust": round(float(memory.get("trust", 0.7)), 6),
        "currentness": round(float(components.get("currentness", 0.7)), 6),
        "utility": round(float(components.get("utility", 0.5)), 6),
        "direct_relevance": round(direct_relevance, 6),
        "feedback_signal": round(feedback_signal, 6),
        "stale_risk": round(stale_risk, 6),
        "wrong_rate": round(wrong_rate, 6),
        "dirty": bool(memory.get("dirty")),
        "superseded": bool(components.get("superseded", 0.0)),
        "state": str(memory.get("state") or "active"),
    }
    raw_probability = (
        0.20 * features["memory_confidence"]
        + 0.16 * features["trust"]
        + 0.14 * features["currentness"]
        + 0.13 * features["utility"]
        + 0.13 * features["direct_relevance"]
        + 0.10 * (1.0 - features["stale_risk"])
        + 0.08 * features["source_prior"]
        + 0.06 * features["feedback_signal"]
        - 0.18 * features["wrong_rate"]
        - (0.16 if features["dirty"] else 0.0)
        - (0.12 if features["superseded"] else 0.0)
        - (0.03 if features["state"] == "cold" else 0.0)
    )
    raw_probability = _clamp(raw_probability)

    learned = calibration or {}
    calibrated_probability = _clamp(float(learned.get("probability", raw_probability)))
    scope = str(learned.get("scope") or "prior")
    samples = max(0, int(learned.get("sample_count") or 0))
    decision = _decision(calibrated_probability)
    reason = _reason(features, decision, scope, samples)
    return MetacognitiveAssessment(
        memory_id=str(memory["id"]),
        source_category=source_category,
        raw_probability=round(raw_probability, 6),
        calibrated_probability=round(calibrated_probability, 6),
        decision=decision,
        reason=reason,
        calibration_scope=scope,
        calibration_samples=samples,
        features=features,
    )


def _decision(probability: float) -> str:
    if probability >= 0.70:
        return "use"
    if probability >= 0.48:
        return "verify"
    return "abstain"


def _reason(features: dict[str, Any], decision: str, scope: str, samples: int) -> str:
    strengths: list[str] = []
    cautions: list[str] = []
    if features["source_prior"] >= 0.80:
        strengths.append("strong source provenance")
    if features["direct_relevance"] >= 0.65:
        strengths.append("strong direct match")
    if features["feedback_signal"] >= 0.70:
        strengths.append("helpful outcome history")
    if features["dirty"]:
        cautions.append("dependent evidence changed")
    if features["superseded"]:
        cautions.append("newer evidence supersedes it")
    if features["stale_risk"] >= 0.40:
        cautions.append("elevated stale-data risk")
    if features["wrong_rate"] >= 0.15:
        cautions.append("prior harmful outcomes")
    if features["source_prior"] < 0.60:
        cautions.append("inferred rather than verified")

    evidence = strengths[:2] or ["mixed supporting signals"]
    if cautions:
        evidence.append(cautions[0])
    if scope != "prior" and samples:
        evidence.append(f"calibrated from {samples} comparable outcomes")
    elif decision != "use":
        evidence.append("outcome calibration is still sparse")
    return "; ".join(evidence).capitalize() + "."


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
