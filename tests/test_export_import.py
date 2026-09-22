"""Tests for cassette export/import (ZIP) and the new CLI commands."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List


from agentreplay import Cassette, Recorder



class _StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)


def _make_cassette(path: Path) -> Path:
    """Create a tiny cassette for testing."""
    stub = _StubLLM([{"text": "hello", "usage": {}}])
    with Recorder.create(path, framework="raw", agent_name="test") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "hi"}], model="stub")
    return path


# ---------------------------------------------------------------------- #
# Cassette export/import
# ---------------------------------------------------------------------- #
def test_cassette_export_zip(tmp_path: Path):
    """export_zip should create a ZIP containing cassette.json, events.jsonl, blobs/."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"

    c = Cassette.open(cassette, readonly=True)
    result_path = c.export_zip(zip_path)

    assert result_path == zip_path
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "cassette.json" in names
        assert "events.jsonl" in names
        # At least one blob file
        blob_files = [n for n in names if n.startswith("blobs/")]
        assert len(blob_files) > 0


def test_cassette_import_zip(tmp_path: Path):
    """import_zip should reconstruct the cassette from a ZIP."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"
    target = tmp_path / "imported"

    c = Cassette.open(cassette, readonly=True)
    c.export_zip(zip_path)

    imported = Cassette.import_zip(zip_path, target)
    assert imported.meta.agent_name == "test"
    assert len(imported.events) == 1
    # The imported cassette should have the same blobs
    assert len(imported.blobs) == len(c.blobs)


def test_cassette_export_import_roundtrip(tmp_path: Path):
    """Export → import should produce an equivalent cassette."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"
    target = tmp_path / "imported"

    original = Cassette.open(cassette, readonly=True)
    original.export_zip(zip_path)
    imported = Cassette.import_zip(zip_path, target)

    # Same metadata (except id and created_at which may differ)
    assert imported.meta.framework == original.meta.framework
    assert imported.meta.agent_name == original.meta.agent_name
    assert imported.meta.num_events == original.meta.num_events

    # Same events
    orig_events = list(original.events)
    imported_events = list(imported.events)
    assert len(orig_events) == len(imported_events)
    assert orig_events[0].call_id == imported_events[0].call_id

    # Same blob count
    assert len(imported.blobs) == len(original.blobs)


def test_import_zip_rejects_nonempty_target(tmp_path: Path):
    """import_zip should refuse to overwrite a non-empty directory."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing.txt").write_text("x")

    c = Cassette.open(cassette, readonly=True)
    c.export_zip(zip_path)

    from agentreplay.errors import CassetteError
    with __import__("pytest").raises(CassetteError, match="not empty"):
        Cassette.import_zip(zip_path, target)


