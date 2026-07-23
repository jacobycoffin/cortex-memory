from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT


class AutoJudgeInstallArtifactTests(unittest.TestCase):
    def test_remote_timer_install_requires_explicit_data_egress_consent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "plugin"
            plugin.mkdir()
            (plugin / "__main__.py").write_text("", encoding="utf-8")
            env = {
                **os.environ,
                "HERMES_HOME": str(root / "hermes"),
                "CORTEX_PLUGIN_DIR": str(plugin),
                "XDG_CONFIG_HOME": str(root / "config"),
                "CORTEX_AUTO_JUDGE_ENDPOINT": "https://provider.example/v1/chat/completions",
            }
            env.pop("CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT", None)

            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "install_auto_judge_timer.sh")],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to enable remote auto-judge", result.stderr)
            self.assertFalse((root / "config" / "systemd" / "user").exists())

    def test_endpoint_userinfo_is_rejected_even_with_remote_consent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "plugin"
            plugin.mkdir()
            (plugin / "__main__.py").write_text("", encoding="utf-8")
            env = {
                **os.environ,
                "HERMES_HOME": str(root / "hermes"),
                "CORTEX_PLUGIN_DIR": str(plugin),
                "XDG_CONFIG_HOME": str(root / "config"),
                "CORTEX_AUTO_JUDGE_ENDPOINT": (
                    "https://localhost:443@provider.example/v1/chat/completions"
                ),
                "CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT": "1",
            }

            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "install_auto_judge_timer.sh")],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not contain URL userinfo", result.stderr)
            self.assertFalse((root / "config" / "systemd" / "user").exists())

    def test_existing_remote_configuration_requires_consent_even_if_invocation_is_loopback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "plugin"
            plugin.mkdir()
            (plugin / "__main__.py").write_text("", encoding="utf-8")
            hermes_home = root / "hermes"
            env_file = hermes_home / "cortex" / "auto-judge.env"
            env_file.parent.mkdir(parents=True)
            env_file.write_text(
                'CORTEX_AUTO_JUDGE_ENABLED=1\n'
                'CORTEX_AUTO_JUDGE_ENDPOINT="http://127.0.0.1:9000/v1/chat/completions"\n'
                'CORTEX_AUTO_JUDGE_ENDPOINT="https://provider.example/v1/chat/completions"\n',
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            systemctl = fake_bin / "systemctl"
            systemctl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            systemctl.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                "HERMES_HOME": str(hermes_home),
                "CORTEX_PLUGIN_DIR": str(plugin),
                "XDG_CONFIG_HOME": str(root / "config"),
                "CORTEX_AUTO_JUDGE_ENDPOINT": "http://127.0.0.1:9000/v1/chat/completions",
            }
            env.pop("CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT", None)

            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "install_auto_judge_timer.sh")],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to enable remote auto-judge", result.stderr)

    def test_timer_is_clock_aligned_every_five_minutes(self) -> None:
        timer = (ROOT / "scripts" / "cortex-auto-judge.timer").read_text(encoding="utf-8")
        self.assertIn("OnCalendar=*:0/5", timer)
        self.assertIn("Persistent=true", timer)

    def test_service_is_quiet_and_does_not_import_the_full_hermes_environment(self) -> None:
        service = (ROOT / "scripts" / "cortex-auto-judge.service.in").read_text(
            encoding="utf-8"
        )
        self.assertIn("auto-judge --quiet", service)
        self.assertIn("StandardOutput=null", service)
        self.assertIn("EnvironmentFile=-@@AUTO_JUDGE_ENV@@", service)
        self.assertIn("TimeoutStartSec=12m", service)
        self.assertNotIn("EnvironmentFile=-@@HERMES_HOME@@/.env", service)

    def test_local_install_wires_and_uninstall_removes_the_timer(self) -> None:
        installer = (ROOT / "scripts" / "install_local.sh").read_text(encoding="utf-8")
        auto_installer = (ROOT / "scripts" / "install_auto_judge_timer.sh").read_text(
            encoding="utf-8"
        )
        uninstaller = (ROOT / "scripts" / "uninstall_local.sh").read_text(encoding="utf-8")
        self.assertIn('CORTEX_INSTALL_AUTO_JUDGE_TIMER:-0', installer)
        self.assertNotIn('CORTEX_INSTALL_AUTO_JUDGE_TIMER:-1', installer)
        self.assertIn('install_auto_judge_timer.sh', installer)
        self.assertIn('CORTEX_AUTO_JUDGE_ENABLED=1', auto_installer)
        self.assertIn('CORTEX_AUTO_JUDGE_DATA_EGRESS_CONSENT:-0', auto_installer)
        self.assertIn('Refusing to enable remote auto-judge', auto_installer)
        self.assertIn('CORTEX_AUTO_JUDGE_CREDENTIAL_FILE=', auto_installer)
        self.assertIn('chmod 600 "$ENV_FILE"', auto_installer)
        self.assertIn('systemctl --user stop cortex-auto-judge.service', uninstaller)
        self.assertIn('cortex-auto-judge.timer', uninstaller)
        self.assertIn('cortex-auto-judge.service', uninstaller)

    def test_uninstall_disables_timer_before_stopping_service_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hermes_home = root / "hermes"
            target = hermes_home / "plugins" / "cortex"
            target.mkdir(parents=True)
            (target / "__main__.py").write_text("", encoding="utf-8")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "systemctl.log"
            systemctl = fake_bin / "systemctl"
            systemctl.write_text(
                "#!/bin/sh\n"
                'printf "%s\\n" "$*" >> "$FAKE_SYSTEMCTL_LOG"\n'
                'case "$*" in\n'
                '  *"is-active --quiet cortex-auto-judge.service"*) exit 0 ;;\n'
                '  *"stop cortex-auto-judge.service"*) exit "${FAIL_SERVICE_STOP:-0}" ;;\n'
                "esac\n"
                "exit 0\n",
                encoding="utf-8",
            )
            systemctl.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                "HERMES_HOME": str(hermes_home),
                "XDG_CONFIG_HOME": str(root / "config"),
                "FAKE_SYSTEMCTL_LOG": str(log),
            }

            first = subprocess.run(
                ["bash", str(ROOT / "scripts" / "uninstall_local.sh")],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            operations = log.read_text(encoding="utf-8").splitlines()
            disable_timer = next(
                i
                for i, item in enumerate(operations)
                if "disable --now cortex-auto-judge.timer" in item
            )
            stop_service = next(
                i
                for i, item in enumerate(operations)
                if "stop cortex-auto-judge.service" in item
            )
            self.assertLess(disable_timer, stop_service)

            target.mkdir(parents=True)
            (target / "__main__.py").write_text("", encoding="utf-8")
            failed = subprocess.run(
                ["bash", str(ROOT / "scripts" / "uninstall_local.sh")],
                env={**env, "FAIL_SERVICE_STOP": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(failed.returncode, 0)
            self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
