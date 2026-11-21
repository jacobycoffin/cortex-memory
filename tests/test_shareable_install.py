"""Opt-in switches, config writing, sidecar runtime, local review, and agent detection."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.autojudge import AutoJudgeConfig, jev_switch


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CORTEX_AUTO_JUDGE", "CORTEX_JEV"))}
    env.update(extra)
    return env


class JevSwitchTests(unittest.TestCase):
    def engine(self, **env: str) -> tuple[str, str]:
        with patch.dict(os.environ, _clean_env(**env), clear=True):
            config = AutoJudgeConfig.from_env()
            return config.engine, config.link_engine

    def test_default_is_chat_even_with_a_typesafe_key(self) -> None:
        self.assertEqual(self.engine(TYPESAFE_API_KEY="present"), ("chat", "chat"))
        with patch.dict(os.environ, _clean_env(TYPESAFE_API_KEY="present"), clear=True):
            self.assertIsNone(jev_switch())

    def test_switch_on_selects_jev_unless_engine_is_explicit(self) -> None:
        self.assertEqual(self.engine(CORTEX_JEV_ENABLED="1"), ("jev", "jev"))
        self.assertEqual(self.engine(CORTEX_JEV_ENABLED="1", CORTEX_AUTO_JUDGE_ENGINE="chat")[0], "chat")

    def test_switch_off_overrides_engine_jev(self) -> None:
        self.assertEqual(self.engine(CORTEX_JEV_ENABLED="0", CORTEX_AUTO_JUDGE_ENGINE="jev",
                                     CORTEX_AUTO_JUDGE_LINK_ENGINE="jev"), ("chat", "chat"))

    def test_auto_judge_is_off_by_default(self) -> None:
        with patch.dict(os.environ, _clean_env(OPENROUTER_API_KEY="x"), clear=True):
            self.assertFalse(AutoJudgeConfig.from_env().enabled)


class ConfigWriterTests(unittest.TestCase):
    SAMPLE = "# top comment\nmemory:\n  # keep me\n  provider: builtin\nplugins:\n  cortex:\n    query_cache_ttl_seconds: 99\n"

    def test_only_missing_keeps_operator_values_and_comments(self) -> None:
        import importlib.util

        if not (importlib.util.find_spec("yaml") or importlib.util.find_spec("ruamel")):
            self.skipTest("needs PyYAML or ruamel.yaml (Hermes ships one of them)")
        from cortex.hermes_config import _load_yaml, write_cortex_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(self.SAMPLE)
            section = write_cortex_config(path, {"background_preload": True, "query_cache_ttl_seconds": 45},
                                          provider="cortex", only_missing=True)
            self.assertEqual(section["query_cache_ttl_seconds"], 99)
            self.assertTrue(section["background_preload"])
            text = path.read_text()
            data = _load_yaml(text)
            self.assertEqual(data["memory"]["provider"], "cortex")
            try:
                import ruamel.yaml  # noqa: F401
            except ImportError:
                return
            self.assertIn("# keep me", text)

    def test_load_yaml_without_pyyaml_uses_ruamel(self) -> None:
        try:
            import ruamel.yaml  # noqa: F401
        except ImportError:
            self.skipTest("ruamel.yaml not installed")
        from cortex import hermes_config

        with patch.dict(sys.modules, {"yaml": None}):
            data = hermes_config._load_yaml(self.SAMPLE)
        self.assertEqual(data["plugins"]["cortex"]["query_cache_ttl_seconds"], 99)


class RuntimePathTests(unittest.TestCase):
    def test_sidecar_is_appended_only_when_present(self) -> None:
        from cortex import runtime_path

        with tempfile.TemporaryDirectory() as tmp:
            ver = f"py{sys.version_info.major}.{sys.version_info.minor}"
            with patch.dict(os.environ, {"HERMES_HOME": tmp}), patch.object(sys, "path", list(sys.path)):
                os.environ.pop("CORTEX_PYDEPS", None)
                self.assertIsNone(runtime_path.ensure_runtime_path())
                target = Path(tmp) / "cortex" / "pydeps" / ver
                target.mkdir(parents=True)
                self.assertEqual(runtime_path.ensure_runtime_path(), str(target))
                self.assertEqual(sys.path[-1], str(target))


class ReviewCliTests(unittest.TestCase):
    def test_list_approve_reject(self) -> None:
        from cortex.cli import main
        from cortex.store import CortexStore

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cortex.db")
            store = CortexStore(db)
            for text in ("we decided to use Redis for the zebra cache.", "I prefer dark mode in editors."):
                store.propose_memory_creation(text, kind="decision", source_type="user_turn",
                                              source_category="USER_STATED", session_id="t",
                                              storage_policy="review_required")
            store.close()

            def run(*args: str):
                out = io.StringIO()
                with patch.object(sys, "argv", ["cortex-memory", "--db", db, "review", *args]), redirect_stdout(out):
                    self.assertEqual(main(), 0)
                return json.loads(out.getvalue())

            pending = run("list")
            self.assertEqual(len(pending), 2)
            approved = run("approve", pending[0]["proposal_id"][:8])
            self.assertTrue(approved[0]["memory_id"])
            rejected = run("reject", pending[1]["proposal_id"])
            self.assertIsNone(rejected[0]["memory_id"])
            self.assertEqual(run("list"), [])


@unittest.skipIf(shutil.which("bash") is None, "bash required")
class DetectionTests(unittest.TestCase):
    def detect(self, home: Path, path_dirs: list[Path], **env: str) -> subprocess.CompletedProcess:
        base = {"HOME": str(home), "PATH": ":".join([*map(str, path_dirs), "/usr/bin", "/bin"]), "TERM": "dumb"}
        base.update(env)
        return subprocess.run(["bash", str(ROOT / "scripts" / "install.sh"), "--detect"],
                              env=base, capture_output=True, text=True, timeout=60)

    def fake_bin(self, root: Path, name: str) -> Path:
        directory = root / f"bin-{name}"
        directory.mkdir()
        script = directory / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        return directory

    def test_detection_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            home.mkdir()
            claude = self.fake_bin(root, "claude")
            hermes = self.fake_bin(root, "hermes")
            out = self.detect(home, [claude])
            self.assertIn("Agent: claude", out.stdout + out.stderr)
            out = self.detect(home, [hermes])
            self.assertIn("Agent: hermes", out.stdout + out.stderr)
            out = self.detect(home, [claude, hermes], CLAUDECODE="1")
            self.assertIn("Agent: claude", out.stdout + out.stderr)
            out = self.detect(home, [claude, hermes], HERMES_SESSION_ID="x")
            self.assertIn("Agent: hermes", out.stdout + out.stderr)
            out = self.detect(home, [claude, hermes])
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("--agent", out.stderr + out.stdout)
            out = self.detect(home, [])
            self.assertNotEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main()
