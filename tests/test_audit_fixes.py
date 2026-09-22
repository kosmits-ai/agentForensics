"""Tests for the audit fixes: public API exports, doctor command, reprs, etc."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from click.testing import CliRunner

from agentreplay import (
    Cassette,
    Recorder,
    Replayer,
    Session,
    RecordingClient,
    RecordingStream,
    ReplayStream,
    mutate_response,
    run_corpus,
    StepContext,
    get_logger,
    set_verbose,
    __version__,
)



class _StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)


def _make_cassette(path: Path) -> Path:
    stub = _StubLLM([{"text": "hello", "usage": {}}])
    with Recorder.create(path, framework="raw", agent_name="test") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "hi"}], model="stub")
    return path


# ---------------------------------------------------------------------- #
# Public API exports
# ---------------------------------------------------------------------- #
def test_all_21_missing_symbols_are_exported():
    """Every symbol that was missing from __init__ should now be importable."""
    import agentreplay
    required = [
        "RecordingClient", "RecordingTool", "RecordingHTTP",
        "RecordingStream", "ReplayStream",
        "mutate_response", "mutate_and_replay", "apply_patch_set",
        "run_corpus", "RegressionReport", "RegressionResult",
        "StepContext",
        "Diff", "FieldDiff", "StepDiff",
        "render_diff", "diff_payloads", "diff_keys",
        "canonical_json", "hash_payload",
        "set_verbose", "get_logger",
    ]
    for name in required:
        assert hasattr(agentreplay, name), f"{name} not exported from agentreplay"
        assert name in agentreplay.__all__, f"{name} not in __all__"


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


# ---------------------------------------------------------------------- #
# __repr__ methods
# ---------------------------------------------------------------------- #
def test_recorder_repr(tmp_path: Path):
    with Recorder.create(tmp_path / "c", framework="raw") as rec:
        r = repr(rec)
    assert "Recorder" in r
    assert "cassette=" in r


def test_replayer_repr(tmp_path: Path):
    _make_cassette(tmp_path / "c")
    with Replayer.open(tmp_path / "c") as rep:
        r = repr(rep)
    assert "Replayer" in r
    assert "cassette=" in r


def test_session_repr(tmp_path: Path):
    with Session.record(tmp_path / "c", framework="raw") as s:
        r = repr(s)
    assert "Session" in r



# ---------------------------------------------------------------------- #
# auto.init() with LIVE mode
# ---------------------------------------------------------------------- #
def test_auto_init_live_mode(monkeypatch):
    """auto.init() should return a live session when AGENTREPLAY_MODE=live."""
    monkeypatch.setenv("AGENTREPLAY_MODE", "live")
    monkeypatch.setenv("AGENTREPLAY_CASSETTE", "/tmp/dummy")
    from agentreplay.auto import init
    session = init()
    assert session is not None
    assert session.mode.value == "live"


# ---------------------------------------------------------------------- #
# Session.__exit__ handles Replayer
# ---------------------------------------------------------------------- #
def test_session_context_manager_with_replayer(tmp_path: Path):
    """Session should properly enter/exit when wrapping a Replayer."""
    _make_cassette(tmp_path / "c")
    with Session.replay(tmp_path / "c") as s:
        assert s is not None
        # In REPLAY mode, the cassette serves the recorded response ("hello"),
        # NOT the stub's response. The stub should never be called.
        client = s.wrap_custom_client(_StubLLM([]))
        r = client.complete(messages=[{"role": "user", "content": "hi"}], model="stub")
        assert r == {"text": "hello", "usage": {}}


# ---------------------------------------------------------------------- #
# bytes handling in canonicalize
# ---------------------------------------------------------------------- #
def test_canonicalize_handles_bytes():
    """canonicalize should handle bytes values by decoding them."""
    from agentreplay import canonicalize
    assert canonicalize(b"hello") == "hello"
    # Non-UTF-8 bytes should fall back to hex
    assert canonicalize(b"\xff\xfe") == "fffe"


def test_canonicalize_bytes_in_dict():
    """canonicalize should handle bytes values nested in dicts."""
    from agentreplay import canonicalize
    result = canonicalize({"body": b"hello", "text": "world"})
    assert result == {"body": "hello", "text": "world"}


# ---------------------------------------------------------------------- #
# RegressionReport render shows divergence details
# ---------------------------------------------------------------------- #
def test_regression_report_render_shows_divergence_details(tmp_path: Path):
    """The render() output should include step_id and call_type for failures."""
    from agentreplay.ci import RegressionReport, RegressionResult
    report = RegressionReport()
    report.results.append(
        RegressionResult(
            cassette_id="cass-test",
            cassette_path="/tmp/test",
            passed=False,
            duration_ms=1.0,
            error="Divergence at step 'step:0:0' (llm)",
            diff={
                "step_id": "step:0:0",
                "call_type": "llm",
                "recorded_call_id": "abc123",
                "actual_call_id": "def456",
            },
        )
    )
    output = report.render()
    assert "step_id: step:0:0" in output
    assert "call_type: llm" in output
    assert "recorded_call_id: abc123" in output



