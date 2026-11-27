"""
Local, offline text embeddings for Cortex.

Runs a small ONNX sentence-embedding model (BAAI/bge-small-en-v1.5, 384-dim)
using optional `onnxruntime`, `tokenizers`, and `numpy` dependencies. Install
them with `pip install '.[embeddings]'` or use an existing harness environment.
The core stays dependency-free; there is no network access at query time and
no data leaving the machine. Model setup is a separate explicit operation.

Verified 2026-09-10: with CLS pooling + L2 normalisation this reproduces
`fastembed`'s output for the same model at cosine 1.0000, so the embeddings
Cortex serves are identical to the ones used in the evaluation that justified
this feature.

Design notes
------------
* The ONNX session is loaded **lazily** on first use. Importing this module must
  not cost ~200 MB of RSS in the gateway process for a feature that may never be
  exercised in a given run.
* `intra_op_num_threads`/`inter_op_num_threads` are pinned to 1. This host is a
  2-vCPU box already running several services; letting ONNX Runtime size its own
  thread pool is how you oversubscribe the CPU and starve everything else.
* Every public entry point degrades to `None`/empty rather than raising when the
  model or the runtime is unavailable, so a missing model can never take the
  memory system down. Callers decide what to do about it.
"""

from __future__ import annotations

import os
import struct
import time
import threading
from pathlib import Path
from typing import Any, Sequence


# Timing helpers live here rather than importing `recall_timing`, because this module is also
# loaded standalone (`from embeddings import Embedder` in tests and scripts), where a relative
# import cannot resolve. Both are total: a broken clock returns 0.0 instead of raising, so
# instrumentation can never be the reason a recall fails.
def _clock() -> float:
    try:
        return float(time.perf_counter())
    except Exception:  # noqa: BLE001
        return 0.0


def _elapsed_ms(started: float) -> float:
    try:
        value = (float(time.perf_counter()) - float(started)) * 1000.0
    except Exception:  # noqa: BLE001
        return 0.0
    return value if value >= 0.0 else 0.0

MODEL_ID = "BAAI/bge-small-en-v1.5"
DIM = 384
MAX_LEN = 512


def default_model_dir() -> Path:
    """Where the ONNX model lives.

    ``CORTEX_MODEL_DIR`` wins when set (the standalone MCP server sets it to
    ``$CORTEX_HOME/models/bge-small-en-v1.5``). Otherwise it is resolved at call
    time from the same ``HERMES_HOME`` convention the rest of the plugin uses, rather than hardcoding a deployment path, so the module
    works on any host and the release checker stays happy.
    """
    explicit = os.environ.get("CORTEX_MODEL_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")
    return Path(home).expanduser() / "models" / "bge-small-en-v1.5"


# Kept for callers that import it directly; resolved lazily by Embedder so a
# later HERMES_HOME change is still honoured.
DEFAULT_MODEL_DIR = default_model_dir()

_MODEL_FILE = "model_optimized.onnx"
_TOKENIZER_FILE = "tokenizer.json"


def pack_vector(vector: Sequence[float]) -> bytes:
    """Serialise a vector to little-endian float32 bytes for SQLite BLOB storage."""
    return struct.pack(f"<{len(vector)}f", *[float(x) for x in vector])


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of `pack_vector`."""
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob[: count * 4]))


class Embedder:
    """Lazy, thread-safe wrapper around the local ONNX embedding model.

    Usage:
        emb = Embedder()                # cheap: nothing is loaded yet
        vectors = emb.embed(texts)      # loads the model on first call
    """

    def __init__(self, model_dir: str | Path | None = None, threads: int = 1):
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.model_id = MODEL_ID
        self.dim = DIM
        self._threads = max(1, int(threads))
        self._session: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        # Per-call timings, so a caller can report the lazy model load separately from inference
        # instead of hiding an ~800 ms one-time cost inside a single "semantic" stage.
        # `last_embed_ms` is the whole call; `last_load_ms` is the part that was the load.
        self.last_load_ms: float = 0.0
        self.last_embed_ms: float = 0.0

    # ---------------------------------------------------------------- loading

    @property
    def available(self) -> bool:
        """True when the model files are present on disk."""
        return (self.model_dir / _MODEL_FILE).is_file() and (
            self.model_dir / _TOKENIZER_FILE
        ).is_file()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        # Both the fast path and the locked path require the tokenizer as well
        # as the session. Publishing them separately used to leave a window in
        # which a second thread saw `_session` set, returned "ready", and then
        # called encode_batch on a still-None tokenizer (audit finding R4).
        if self._session is not None and self._tokenizer is not None:
            return True
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return True
            if not self.available:
                self._load_error = f"model files missing under {self.model_dir}"
                return False
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                opts = ort.SessionOptions()
                # Pin threads: this box is small and shared. See module docstring.
                opts.intra_op_num_threads = self._threads
                opts.inter_op_num_threads = self._threads
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(
                    str(self.model_dir / _MODEL_FILE),
                    providers=["CPUExecutionProvider"],
                    sess_options=opts,
                )
                tokenizer = Tokenizer.from_file(str(self.model_dir / _TOKENIZER_FILE))
                tokenizer.enable_truncation(max_length=MAX_LEN)
                tokenizer.enable_padding(length=None)   # dynamic padding per call
            except Exception as exc:                    # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._session = None
                self._tokenizer = None
                return False
            # Publish both together only after everything succeeded, so no
            # observer ever sees a session without its tokenizer.
            self._session = session
            self._tokenizer = tokenizer
            return True

    # -------------------------------------------------------------- inference

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns [] if the model is unavailable."""
        self.last_load_ms = 0.0
        self.last_embed_ms = 0.0
        started = _clock()
        clean = [t if isinstance(t, str) else "" for t in texts]
        if not clean:
            return []
        already_loaded = self._session is not None and self._tokenizer is not None
        if not self._ensure_loaded():
            self.last_embed_ms = _elapsed_ms(started)
            return []
        if not already_loaded:
            # The first call in a process pays the ONNX session and tokenizer construction. That
            # is a one-time cost of the *process*, not of the query, and it is what makes the
            # first prefetch of a session slow.
            self.last_load_ms = _elapsed_ms(started)

        import numpy as np

        try:
            encoded = self._tokenizer.encode_batch(clean)
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            feeds: dict[str, Any] = {"input_ids": input_ids}
            names = {i.name for i in self._session.get_inputs()}
            if "attention_mask" in names:
                feeds["attention_mask"] = attention
            if "token_type_ids" in names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            hidden = self._session.run(None, feeds)[0]

            # CLS pooling — verified to reproduce fastembed exactly for this model
            # (mean pooling gives ~0.94 cosine, i.e. a DIFFERENT vector space).
            pooled = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-12, None)
            return pooled.tolist()
        finally:
            # Published even when inference raises: a turn that spent 900 ms in here and then
            # failed still spent it, and a stored zero would read as "the embedder was never
            # tried".
            self.last_embed_ms = _elapsed_ms(started)

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text. Returns [] if the model is unavailable."""
        out = self.embed([text])
        return out[0] if out else []


_default: Embedder | None = None
_default_lock = threading.Lock()


def get_embedder(model_dir: str | Path | None = None) -> Embedder:
    """Process-wide default embedder (lazy; safe to call often)."""
    global _default
    if model_dir is not None:
        return Embedder(model_dir)
    with _default_lock:
        if _default is None:
            _default = Embedder()
        return _default
