#!/usr/bin/env python3
"""Explicitly download and verify the pinned local embedding test model.

Only this setup command uses the network. Cortex inference never downloads a
model. Files come from an immutable public revision and are SHA-256 checked
before they atomically replace any existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


MODEL_REPOSITORY = "Qdrant/bge-small-en-v1.5-onnx-Q"
MODEL_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
MODEL_FILES = {
    "model_optimized.onnx": "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, expected_digest in MODEL_FILES.items():
        target = output / name
        if target.is_file() and file_sha256(target) == expected_digest:
            continue
        temporary: Path | None = None
        try:
            url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=output, prefix=f".{name}.", delete=False) as dest:
                    temporary = Path(dest.name)
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        dest.write(chunk)
            if file_sha256(temporary) != expected_digest:
                raise ValueError(f"SHA-256 mismatch for {name}; existing model was preserved")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for verified ONNX/tokenizer files")
    args = parser.parse_args()
    download_model(args.output)
    print(f"Verified {MODEL_REPOSITORY} at revision {MODEL_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
