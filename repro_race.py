"""Reproduce the stale-assessment race scenario."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import cortex  # noqa: F401
except ModuleNotFoundError:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cortex",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Cortex package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)

import tempfile
from cortex.store import CortexStore, StaleCreationProposalError, creation_proposal_revision


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = CortexStore(Path(tmp) / "c.db")
        # Structural reference content gets classified as reference, so in a
        # trained recall set it stays evidence_only even for a "remember" action.
        content = """{
  \"server\": \"acorn.example.com\",
  \"port\": 8642,
  \"gateway\": \"blue\"
}"""

        # Create the proposal BEFORE the duplicate exists, so the assessment
        # does not contain duplicate_memory_id.
        proposal = store.propose_memory_creation(content, source_type="assistant_turn")
        print("proposal status", proposal["status"])
        print("duplicate_memory_id", (proposal.get("assessment") or {}).get("duplicate_memory_id"))

        # Now create a duplicate memory with evidence-only eligibility by making
        # the active recall set trained and the record role reference.
        with store._lock:
            store._conn.execute("UPDATE memory_recall_sets SET kind='trained' WHERE status='active'")
            store._conn.commit()
        memory_id, created = store.add_memory(
            content,
            source_category="AGENT_INFERENCE",
            source_type="assistant_turn",
            approval_state="unreviewed",
            record_role="reference",
        )
        print("memory_id", memory_id, "created", created)
        print("recall eligible primary", store.is_memory_recall_eligible(memory_id))
        print("recall eligible evidence", store.is_memory_recall_eligible(memory_id, evidence_lookup=True))
        print("memory record_role", store.get_memory(memory_id).get("record_role"))

        # Refresh the proposal so the review reads current assessment.
        proposal = store.get_memory_creation_proposal(proposal["proposal_id"])
        print("refreshed duplicate_memory_id", (proposal.get("assessment") or {}).get("duplicate_memory_id"))

        try:
            result = store.review_memory_creation(
                proposal["proposal_id"],
                "remember",
                actor="cortex-auto-judge:synthetic",
                approval_authority="automatic",
                expected_revision=creation_proposal_revision(proposal),
            )
            print("result", result)
        except StaleCreationProposalError as e:
            print("StaleCreationProposalError", e)
        print("proposal status after", store.get_memory_creation_proposal(proposal["proposal_id"])["status"])
        store.close()


if __name__ == "__main__":
    main()
