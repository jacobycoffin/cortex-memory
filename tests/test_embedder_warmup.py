"""Warming the embedding model must never cost a turn.

The first recall of a session pays ~800 ms to load the ONNX session, inside a 1,500 ms deadline. Loading it in a daemon
thread at initialize moves that cost off the turn path — but only if every failure mode stays harmless. A missing model, an
unavailable runtime or a hung loader must all leave recall exactly as it was.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import CortexMemoryProvider


class WarmupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def provider(self, **config) -> CortexMemoryProvider:
        provider = CortexMemoryProvider({"db_path": "$HERMES_HOME/cortex/test.db", **config})
        provider.initialize("warm-session", hermes_home=self.tmp.name, agent_context="primary")
        return provider

    def test_initialize_survives_a_broken_embedder(self) -> None:
        """A raising loader happens in another thread and must not reach the caller at all."""

        def explode(*_args, **_kwargs):
            raise RuntimeError("model files are gone")

        with patch("cortex.embeddings.get_embedder", explode):
            provider = self.provider()
        self.assertIsNotNone(provider._store)
        provider.shutdown()

    def test_a_hung_loader_does_not_delay_initialize(self) -> None:
        """The whole point is that this work is off the turn path."""

        def hang(*_args, **_kwargs):
            time.sleep(5)
            return None

        with patch("cortex.embeddings.get_embedder", hang):
            started = time.monotonic()
            provider = self.provider()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0, f"initialize waited {elapsed:.2f}s on the embedder")
        provider.shutdown()

    def test_the_warmup_thread_is_started_by_default(self) -> None:
        started: list[str] = []
        real_thread = __import__("threading").Thread

        class Recording(real_thread):
            def start(self) -> None:  # noqa: D401 - test double
                started.append(self.name)
                super().start()

        with patch("cortex.hermes_provider.threading.Thread", Recording):
            provider = self.provider()
        self.assertIn("cortex-embed-warm", started)
        provider.shutdown()

    def test_warmup_can_be_switched_off(self) -> None:
        started: list[str] = []
        real_thread = __import__("threading").Thread

        class Recording(real_thread):
            def start(self) -> None:
                started.append(self.name)
                super().start()

        with patch("cortex.hermes_provider.threading.Thread", Recording):
            provider = self.provider(warm_embeddings=False)
        self.assertNotIn("cortex-embed-warm", started)
        provider.shutdown()


if __name__ == "__main__":
    unittest.main()
