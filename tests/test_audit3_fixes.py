"""Tests for the round-3 audit fixes: clock counter, tool exception replay,
streaming data loss, zip slip, schema version, ci exception catching, etc."""
from __future__ import annotations

import asyncio
import json
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer, Mode
from agentreplay.constants import CallType
from agentreplay.interceptors.streaming import RecordingStream, ReplayStream


class _StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)


# ---------------------------------------------------------------------- #
# C1: Clock interceptor produces unique call-site IDs
# ---------------------------------------------------------------------- #
def test_clock_multiple_calls_in_same_step_get_unique_ids(tmp_path: Path):
    """Two clock.time() calls in the same step must produce different
    call-site IDs so both values replay correctly."""
    cassette = tmp_path / "cass"
    with Recorder.create(cassette, framework="raw") as rec:
        clock = rec.clock
        t1 = clock.time()
        t2 = clock.time()

    c = Cassette.open(cassette, readonly=True)
    events = list(c.events)
    assert len(events) == 2
    # The two events must have DIFFERENT call_ids
    assert events[0].call_id != events[1].call_id
    # The two events must have DIFFERENT step_ids (counter included)
    assert events[0].step_id != events[1].step_id


def test_clock_multiple_calls_replay_bit_exact(tmp_path: Path):
    """Multiple clock.time() calls must replay the EXACT recorded values,
    not just the last one."""
    cassette = tmp_path / "cass"
    with Recorder.create(cassette, framework="raw") as rec:
        clock = rec.clock
        t1 = clock.time()
        t2 = clock.time()
        assert t1 != t2  # wall clock should differ

    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        clock = rep.clock
        r1 = clock.time()
        r2 = clock.time()

    assert r1 == t1
    assert r2 == t2


# ---------------------------------------------------------------------- #
# C2: Tool exceptions are replayed (re-raised)
# ---------------------------------------------------------------------- #
def test_tool_exception_replay_raises_same_exception_type(tmp_path: Path):
    """When a tool raised ValueError during recording, replay should
    re-raise ValueError (not return None)."""
    cassette = tmp_path / "cass"

    def bad_tool(x: int) -> int:
        raise ValueError("boom")

    with Recorder.create(cassette, framework="raw") as rec:
        tool = rec.wrap_tool(bad_tool, name="bad")
        with pytest.raises(ValueError, match="boom"):
            tool(x=1)

    # Now replay — should re-raise
    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        tool = rep.wrap_tool(bad_tool, name="bad")
        with pytest.raises(ValueError, match="boom"):
            tool(x=1)


# ---------------------------------------------------------------------- #
# C3: Streaming data loss on early break
# ---------------------------------------------------------------------- #
def test_streaming_early_break_still_records(tmp_path: Path):
    """If the consumer breaks out of a stream early, the event should
    still be written to the cassette (try/finally)."""
    cassette = tmp_path / "cass"
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

    stub = _StubLLM([])
    stub.responses = [chunks]  # not used; we'll use streaming directly

    with Recorder.create(cassette, framework="raw") as rec:
        from agentreplay.interceptors.streaming import RecordingStream, make_streamed_response

        captured: list = []

        def on_complete(captured_chunks):
            captured.append(captured_chunks)
            rec.cassette.write_event(
                step_id="step:0:0",
                call_type=CallType.LLM,
                call_id="test-stream-id",
                request={"messages": []},
                response=make_streamed_response(captured_chunks),
                started_at=0.0,
                duration_ms=1.0,
                metadata={"streamed": True},
            )

        stream = RecordingStream(iter(chunks), on_complete=on_complete)
        # Break early after first chunk
        for chunk in stream:
            break

    # The event should have been written despite early break
    c = Cassette.open(cassette, readonly=True)
    assert len(list(c.events)) == 1


# ---------------------------------------------------------------------- #
# C4: Async streaming support
# ---------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_replay_stream_async_iteration():
    """ReplayStream should support `async for chunk in stream`."""
    chunks = [{"text": "a"}, {"text": "b"}]
    stream = ReplayStream(chunks)
    result = []
    async for chunk in stream:
        result.append(chunk)
    assert result == chunks


@pytest.mark.asyncio
async def test_recording_stream_async_iteration_with_fallback():
    """RecordingStream should support async iteration when the real stream
    is sync (falls back to sync iteration in async context)."""
    chunks = [{"text": "x"}, {"text": "y"}]
    captured: list = []

    def on_complete(c):
        captured.append(c)

    stream = RecordingStream(iter(chunks), on_complete=on_complete)
    result = []
    async for chunk in stream:
        result.append(chunk)

    assert result == chunks
    assert len(captured) == 1
    assert captured[0] == chunks


# ---------------------------------------------------------------------- #
# C5: Zip Slip vulnerability
# ---------------------------------------------------------------------- #
def test_import_zip_rejects_zip_slip(tmp_path: Path):
    """import_zip should reject a ZIP with entries that would extract
    outside the target root."""
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../etc/evil.txt", "pwned")

    from agentreplay.errors import CassetteError
    with pytest.raises(CassetteError, match="Zip Slip"):
        Cassette.import_zip(evil_zip, tmp_path / "target")


# ---------------------------------------------------------------------- #
# H1: Schema version validation
# ---------------------------------------------------------------------- #
def test_schema_version_mismatch_warns(tmp_path: Path):
    """Opening a cassette with a different schema_version should warn."""
    cassette = tmp_path / "cass"
    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(_StubLLM([{"text": "x", "usage": {}}]))
        client.complete(messages=[{"role": "user", "content": "q"}], model="stub")

    # Manually edit the cassette.json to change schema_version
    meta_path = cassette / "cassette.json"
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = "99.0.0"
    meta_path.write_text(json.dumps(meta))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Cassette.open(cassette, readonly=True)
        assert any("schema version" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------- #
# H11: RecordingRandom.shuffle uses recorded permutation on replay
# ---------------------------------------------------------------------- #
def test_shuffle_replay_uses_recorded_permutation(tmp_path: Path):
    """shuffle() should restore the recorded permutation on replay,
    not call the live RNG."""
    cassette = tmp_path / "cass"

    with Recorder.create(cassette, framework="raw") as rec:
        rng = rec.random
        seq = [3, 1, 4, 1, 5, 9, 2, 6]
        rng.shuffle(seq)
        recorded = list(seq)

    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        rng = rep.random
        seq = [3, 1, 4, 1, 5, 9, 2, 6]
        rng.shuffle(seq)
        assert seq == recorded


# ---------------------------------------------------------------------- #
# H12: ci.run_corpus catches non-AgentReplay exceptions
# ---------------------------------------------------------------------- #
def test_run_corpus_catches_generic_exceptions(tmp_path: Path):
    """If agent_run raises a non-AgentReplay exception, the corpus run
    should record it as a failure and continue."""
    from agentreplay.ci import run_corpus

    # Create a cassette
    cassette = tmp_path / "cass"
    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(_StubLLM([{"text": "x", "usage": {}}]))
        client.complete(messages=[{"role": "user", "content": "q"}], model="stub")

    # agent_run that raises a non-AgentReplay exception
    def crashing_agent_run(replayer):
        raise KeyError("agent bug")

    report = run_corpus(tmp_path, crashing_agent_run)
    assert len(report.results) == 1
    assert not report.results[0].passed
    assert "KeyError" in report.results[0].error


# ---------------------------------------------------------------------- #
# M6: set_verbose not overridden by get_logger
# ---------------------------------------------------------------------- #
def test_set_verbose_not_overridden_by_get_logger():
    """set_verbose(True) should persist even after get_logger() is called."""
    import logging
    from agentreplay.logging import set_verbose, get_logger

    set_verbose(True)

    # get_logger should NOT reset the level
    get_logger("test")

    root = logging.getLogger("agentreplay")
    assert root.level == logging.DEBUG

    # Cleanup
    set_verbose(False)



