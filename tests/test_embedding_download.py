"""Pinned setup downloads preserve existing models until verification passes."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_embedding_model as downloader


class EmbeddingDownloadTests(unittest.TestCase):
    def test_download_verifies_content_and_reuses_verified_artifacts(self) -> None:
        artifact = b"synthetic model artifact"
        files = {"example.onnx": hashlib.sha256(artifact).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp, patch.object(downloader, "MODEL_FILES", files):
            output = Path(tmp)
            with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(artifact)) as request:
                downloader.download_model(output)
            self.assertEqual((output / "example.onnx").read_bytes(), artifact)
            self.assertIn(downloader.MODEL_REVISION, request.call_args.args[0])
            with patch.object(downloader.urllib.request, "urlopen") as request:
                downloader.download_model(output)
                request.assert_not_called()

    def test_corrupt_download_preserves_existing_file_and_removes_temporary_data(self) -> None:
        expected = hashlib.sha256(b"expected synthetic artifact").hexdigest()
        with tempfile.TemporaryDirectory() as tmp, patch.object(downloader, "MODEL_FILES", {"example.onnx": expected}):
            output = Path(tmp)
            target = output / "example.onnx"
            target.write_bytes(b"previous artifact")
            with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    downloader.download_model(output)
            self.assertEqual(target.read_bytes(), b"previous artifact")
            self.assertEqual(list(output.iterdir()), [target])

    def test_provider_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(downloader, "MODEL_FILES", {"example.onnx": "invalid"}):
            output = Path(tmp)
            target = output / "example.onnx"
            target.write_bytes(b"previous artifact")
            with patch.object(downloader.urllib.request, "urlopen", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    downloader.download_model(output)
            self.assertEqual(target.read_bytes(), b"previous artifact")
            self.assertEqual(list(output.iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
