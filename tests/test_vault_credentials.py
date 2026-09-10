"""Regression tests for the vault credential-section skip hotfix.

Credential headings carry live secret VALUES.  Before the hotfix the only guard
was the per-fragment heuristic in ``_skip_low_value_fragment``, which happily
ingested a realistic credential block (three substantial secret lines look like
ordinary prose to a token count).  Purging the memories was not durable because
the vault source file still held the secrets, so the next indexer pass simply
re-created the rows.  The fix adds a section-level skip keyed on the heading
leaf, which these tests pin down.

The fake secrets below are invented strings that match no secret-redaction
pattern in ``cortex.security``, so any appearance of them in stored content can
only be explained by the credential section having been ingested.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


from tests._bootstrap import ROOT  # noqa: F401  (loads the repo package as ``cortex``)

from cortex.store import CortexStore
from cortex.vault import VaultIndexer, _skip_low_value_section


FAKE_SECRET = "SWORDFISH-TOKEN-abc123"
FAKE_SECRET_2 = "SWORDFISH-TOKEN-def456"
FAKE_SECRET_3 = "SWORDFISH-TOKEN-ghi789"

# A credential block that is substantive enough to pass the fragment heuristic:
# seven meaningful tokens, no placeholders, no underscore-heavy identifiers.
# That is exactly the case the section-level skip has to catch on its own.
CREDENTIAL_BODY = "\n".join(
    (
        f"Plex: {FAKE_SECRET}",
        f"NPM admin: {FAKE_SECRET_2}",
        f"Komga: {FAKE_SECRET_3}",
    )
)

# Headings that must be skipped.  "Homelab › Credentials" proves the
# " › "-joined parent prefix is stripped to its leaf before matching.
CREDENTIAL_HEADINGS = (
    "Credentials",
    "Credential",
    "Homelab › Credentials",
    "API Keys",
    "API-Key",
    "Tokens",
    "Login",
    "Passwords",
    "Secrets",
    "Authentication",
    "Auth",
)

# Ordinary headings that must keep being ingested.
ORDINARY_HEADINGS = (
    "Services",
    "Networking",
    "Ports",
    "Storage",
    "Next Steps",
    "Homelab › Networking",
)


class CredentialSectionSkipUnitTests(unittest.TestCase):
    """Unit coverage for the heading-based section skip."""

    def test_credential_headings_skip_and_ordinary_headings_do_not(self) -> None:
        for heading in CREDENTIAL_HEADINGS:
            with self.subTest(heading=heading):
                self.assertTrue(
                    _skip_low_value_section(heading, CREDENTIAL_BODY),
                    f"{heading!r} carries live secrets and must be skipped",
                )
        for heading in ORDINARY_HEADINGS:
            with self.subTest(heading=heading):
                self.assertFalse(
                    _skip_low_value_section(heading, CREDENTIAL_BODY),
                    f"{heading!r} is ordinary content and must still be indexed",
                )

    def test_matching_uses_the_heading_leaf_only(self) -> None:
        # A credential leaf under an ordinary parent is still skipped ...
        self.assertTrue(_skip_low_value_section("Homelab › Credentials › Passwords", CREDENTIAL_BODY))
        # ... while an ordinary leaf under an ordinary parent is not.
        self.assertFalse(
            _skip_low_value_section("Homelab › Credentials › Rotation Schedule", CREDENTIAL_BODY)
        )


class CredentialVaultIndexingTests(unittest.TestCase):
    """Integration coverage: a real indexer run over a temp vault."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.vault = root / "vault"
        self.vault.mkdir(parents=True)
        self.host = self.vault / "Homelab Host.md"
        self.host.write_text(
            "# Homelab Host\n\n"
            "## Credentials\n\n"
            f"Plex: {FAKE_SECRET}\n"
            f"NPM admin: {FAKE_SECRET_2}\n"
            f"Komga: {FAKE_SECRET_3}\n"
            "Rotation runbook: [[Router]]\n\n"
            "## Networking\n\n"
            "The router hands out leases on the 10.20.0.0/24 subnet and the "
            "wireguard tunnel carries dashboard traffic between the two sites.\n",
            encoding="utf-8",
        )
        self.router = self.vault / "Router.md"
        self.router.write_text(
            "# Router\n\n"
            "The router holds the lease table and forwards every dashboard "
            "request to the storage node.\n",
            encoding="utf-8",
        )
        self.store = CortexStore(root / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _contents(self, source_path: str) -> list[str]:
        return [
            self.store.get_memory(row["memory_id"])["content"]
            for row in self.store.document_chunks(source_path, active_only=True)
        ]

    def test_credential_section_never_reaches_memory(self) -> None:
        result = VaultIndexer(self.store, self.vault).apply()
        self.assertTrue(result["audit"]["ok"])

        chunks = self.store.document_chunks("Homelab Host.md", active_only=True)
        contents = self._contents("Homelab Host.md")
        joined = "\n".join(contents)

        # (a) No stored memory carries the credential values or the section.
        for secret in (FAKE_SECRET, FAKE_SECRET_2, FAKE_SECRET_3):
            self.assertNotIn(secret, joined, "credential value reached a memory")
        self.assertNotIn("Section: Homelab Host › Credentials", joined)
        self.assertEqual([row["heading"] for row in chunks], ["Homelab Host › Networking"])

        # (b) The ordinary section is still ingested.
        self.assertTrue(any("10.20.0.0/24" in content for content in contents), contents)
        self.assertTrue(any("wireguard tunnel" in content for content in contents), contents)

    def test_wikilink_edges_survive_the_credential_skip(self) -> None:
        result = VaultIndexer(self.store, self.vault).apply()
        self.assertTrue(result["audit"]["ok"])

        # Only the non-credential section becomes a memory for the host note ...
        host_chunks = self.store.document_chunks("Homelab Host.md", active_only=True)
        self.assertEqual(len(host_chunks), 1)
        router_chunks = self.store.document_chunks("Router.md", active_only=True)
        self.assertTrue(router_chunks)
        host_id = host_chunks[0]["memory_id"]

        # ... yet the wikilink written inside the skipped section still produces
        # a vault edge, because link extraction works on full-note text.  The
        # edge is what preserves graph coverage; its *context* is suppressed
        # because it lives in a credential section.
        edges = [
            edge
            for edge in (self.store.explain(host_id)["edges"] if host_id else [])
            if edge["relation"] == "vault_link"
        ]
        self.assertEqual(len(edges), 1, edges)
        self.assertIn("explicitly links to Router", edges[0]["explanation"])
        self.assertNotIn("Rotation runbook", edges[0]["explanation"])
        self.assertNotIn(FAKE_SECRET, repr(edges[0]))
        self.assertEqual(result["vault_links_created"], 1)

    def test_credential_section_line_never_reaches_edge_evidence(self) -> None:
        """A secret sharing a line with a wikilink must not ride into edges.

        ``_wikilink_contexts`` scans whole lines, so before this fix the
        section-level skip alone still left the raw credential line in
        ``edge_evidence.summary`` and ``edge_evidence.metadata_json`` as the
        edge's ``link_context`` (verified 2026-09-10).
        """
        result = VaultIndexer(self.store, self.vault).apply()
        self.assertTrue(result["audit"]["ok"])

        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT summary, metadata_json FROM edge_evidence"
            ).fetchall()
        self.assertTrue(rows, "the fixture must produce at least one edge")

        for row in rows:
            record = f"{row['summary']}\n{row['metadata_json']}"
            for secret in (FAKE_SECRET, FAKE_SECRET_2, FAKE_SECRET_3):
                self.assertNotIn(
                    secret, record, "credential value reached edge evidence"
                )
            self.assertNotIn(
                "Rotation runbook",
                record,
                "credential-section text reached edge evidence",
            )

        # The edge itself still exists — dropping a section from the context
        # text must not cost graph coverage.
        self.assertEqual(result["vault_links_created"], 1)

    def test_ordinary_section_link_context_is_preserved(self) -> None:
        """Trimming credential sections must not strip ordinary link contexts.

        The fix filters the note text before extracting link contexts, so the
        obvious failure mode in the other direction is silently losing every
        context.  This pins that an ordinary-section wikilink keeps its context.
        """
        (self.vault / "Storage.md").write_text(
            "# Storage\n\nThe tank node holds the media library.\n",
            encoding="utf-8",
        )
        (self.vault / "Networking.md").write_text(
            "# Networking\n\n"
            "## Notes\n\n"
            "The lease table lives on [[Storage]] and rotates weekly.\n",
            encoding="utf-8",
        )

        result = VaultIndexer(self.store, self.vault).apply()
        self.assertTrue(result["audit"]["ok"])

        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT summary, metadata_json FROM edge_evidence "
                "WHERE metadata_json LIKE '%Networking.md%'"
            ).fetchall()
        self.assertTrue(
            rows, "an ordinary-section wikilink must still produce an edge"
        )

        blob = "\n".join(f"{row['summary']}\n{row['metadata_json']}" for row in rows)
        self.assertIn(
            "rotates weekly",
            blob,
            "an ordinary-section link context must be preserved, not trimmed away",
        )


if __name__ == "__main__":
    unittest.main()
