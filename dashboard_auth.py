"""Password and signed-session support for the Cortex dashboard."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from typing import Any


PBKDF2_ITERATIONS = 310_000
SESSION_COOKIE = "cortex_session"
SESSION_SECONDS = 12 * 60 * 60


def generate_temporary_password() -> str:
    """Return a readable high-entropy temporary password."""

    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
    return "-".join("".join(secrets.choice(alphabet) for _ in range(5)) for _ in range(4))


def validate_new_password(password: str, username: str) -> str | None:
    if len(password) < 12:
        return "Use at least 12 characters."
    if len(password) > 256:
        return "Password is too long."
    if password.casefold() == username.casefold() or username.casefold() in password.casefold():
        return "Password cannot contain the username."
    return None


class DashboardAuth:
    """Persistent password hash plus stateless, revocable session cookies."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._lock = threading.RLock()
        self._state: dict[str, Any] | None = None
        # (mtime_ns, size) of the auth file `_state` was read from. The file
        # can be rewritten by a *separate* process (e.g. a password reset via
        # the CLI while the dashboard keeps running), so every load compares
        # the on-disk stat and re-reads on change instead of trusting the
        # cache indefinitely. Without this, a running instance keeps accepting
        # the old password/sessions and rejects the new password.
        self._stat: tuple[int, int] | None = None

    @property
    def configured(self) -> bool:
        with self._lock:
            return self.path.exists() or self._state is not None

    def ensure(self, username: str, password: str, *, must_change: bool = False) -> None:
        """Create the auth file once, usually by migrating legacy environment credentials."""

        with self._lock:
            if self.path.exists():
                self._load()
                return
            self._write_new(username, password, must_change=must_change, session_version=1)

    def reset(
        self,
        *,
        username: str = "cortex",
        password: str | None = None,
        must_change: bool = True,
    ) -> str:
        password = password or generate_temporary_password()
        problem = validate_new_password(password, username)
        if problem:
            raise ValueError(problem)
        with self._lock:
            previous = self._load(optional=True) or {}
            version = int(previous.get("session_version", 0)) + 1
            self._write_new(username, password, must_change=must_change, session_version=version)
        return password

    def username(self) -> str:
        return str(self._require_state()["username"])

    def must_change_password(self) -> bool:
        return bool(self._require_state().get("must_change_password", False))

    def verify_password(self, username: str, password: str) -> bool:
        state = self._require_state()
        expected = _decode(str(state["password_hash"]))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _decode(str(state["salt"])),
            int(state.get("iterations", PBKDF2_ITERATIONS)),
        )
        return hmac.compare_digest(username, str(state["username"])) and hmac.compare_digest(actual, expected)

    def change_password(self, username: str, current_password: str, new_password: str) -> str:
        if not self.verify_password(username, current_password):
            raise ValueError("Current password is incorrect.")
        problem = validate_new_password(new_password, username)
        if problem:
            raise ValueError(problem)
        with self._lock:
            current = self._require_state()
            version = int(current.get("session_version", 0)) + 1
            self._write_new(username, new_password, must_change=False, session_version=version)
        return self.issue_session()

    def issue_session(self, *, now: int | None = None) -> str:
        state = self._require_state()
        issued = int(time.time() if now is None else now)
        payload = {
            "exp": issued + SESSION_SECONDS,
            "iat": issued,
            "u": str(state["username"]),
            "v": int(state["session_version"]),
        }
        encoded = _encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = hmac.new(_decode(str(state["session_secret"])), encoded.encode("ascii"), hashlib.sha256).digest()
        return f"{encoded}.{_encode(signature)}"

    def verify_session(self, token: str, *, now: int | None = None) -> dict[str, Any] | None:
        try:
            encoded, supplied_signature = token.split(".", 1)
            state = self._require_state()
            expected_signature = hmac.new(
                _decode(str(state["session_secret"])), encoded.encode("ascii"), hashlib.sha256
            ).digest()
            if not hmac.compare_digest(_decode(supplied_signature), expected_signature):
                return None
            payload = json.loads(_decode(encoded))
            current = int(time.time() if now is None else now)
            if int(payload["exp"]) <= current:
                return None
            if payload.get("u") != state["username"]:
                return None
            if int(payload.get("v", -1)) != int(state["session_version"]):
                return None
            return payload
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, binascii.Error):
            return None

    def session_from_cookie(self, header: str | None) -> dict[str, Any] | None:
        if not header:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(header)
        except CookieError:
            return None
        morsel = cookie.get(SESSION_COOKIE)
        return self.verify_session(morsel.value) if morsel else None

    def _write_new(self, username: str, password: str, *, must_change: bool, session_version: int) -> None:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
        )
        state = {
            "iterations": PBKDF2_ITERATIONS,
            "must_change_password": bool(must_change),
            "password_hash": _encode(digest),
            "salt": _encode(salt),
            "session_secret": _encode(secrets.token_bytes(32)),
            "session_version": int(session_version),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "username": username,
            "version": 1,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        self._state = state
        self._stat = self._current_stat()

    def _current_stat(self) -> tuple[int, int] | None:
        """Return the auth file's identity probe, or None if it is unreadable."""
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _load(self, *, optional: bool = False) -> dict[str, Any] | None:
        if self._state is not None and self._current_stat() == self._stat:
            return self._state
        # Cache miss or the file changed under us (another process reset the
        # password) — re-read so this instance enforces the current password
        # and session version. A deleted file fails closed via _require_state.
        if optional and not self.path.exists():
            self._state = None
            self._stat = None
            return None
        state = json.loads(self.path.read_text(encoding="utf-8"))
        required = {"username", "salt", "password_hash", "session_secret", "session_version"}
        if not required.issubset(state):
            raise ValueError("dashboard auth file is incomplete")
        self._state = state
        self._stat = self._current_stat()
        return state

    def _require_state(self) -> dict[str, Any]:
        with self._lock:
            state = self._load(optional=True)
            if state is None:
                raise RuntimeError("dashboard authentication is not configured")
            return state


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
