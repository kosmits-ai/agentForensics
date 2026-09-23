"""Tests for the agent's tool set.

Two concerns here. First, the tools behave correctly — especially
`memory_write`, whose dual job (return a result to the model AND mutate the
store) is what makes memory persist across sessions. Second, each schema
still agrees with its function signature: the schema is all the model sees,
so a silent drift between them corrupts runs without raising anything.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from agentreplay import Cassette, Recorder
from memforensics.memory import MemoryState, MemoryWrite
from memforensics.oracle import SINK
from memforensics.tools import (
    BENIGN_INBOX,
    POISONED_INBOX,
    TOOL_SCHEMAS,
    get_calendar,
    lookup_contact,
    make_tools,
    send_message,
)

DIM = 384


class StubEmbedder:
    """Distinct unit vector per distinct string — no model download."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), DIM))
        for i, text in enumerate(texts):
            rows[i, self._index.setdefault(text, len(self._index)) % DIM] = 1.0
        return rows


@pytest.fixture
def store() -> MemoryState:
    return MemoryState(StubEmbedder())


# --------------------------------------------------------------------- #
# Fixture-backed tools
# --------------------------------------------------------------------- #
def test_lookup_contact_is_case_insensitive():
    assert lookup_contact(name="Alice") == lookup_contact(name="alice")


def test_lookup_contact_reports_unknown_names():
    assert lookup_contact(name="nobody") == "not found"


def test_get_calendar_returns_events_for_a_known_day():
    assert "Workshop at 2 PM" in get_calendar(date="Thursday")


def test_get_calendar_normalises_the_day():
    assert get_calendar(date="  thursday ") == get_calendar(date="Thursday")


def test_get_calendar_does_not_invent_events_for_unknown_days():
    """A miss must be visibly a miss — never a plausible-looking answer."""
    assert "No events found" in get_calendar(date="Blursday")


def test_send_message_echoes_every_recipient():
    result = send_message(to="a@x.com, b@x.com", subject="s", body="b")
    assert "a@x.com" in result and "b@x.com" in result


# --------------------------------------------------------------------- #
# make_tools
# --------------------------------------------------------------------- #
def test_exposes_every_tool(store: MemoryState):
    tools, _ = make_tools(store, session_id=0)
    assert set(tools) == {
        "lookup_contact", "get_calendar", "send_message", "read_inbox",
        "memory_read", "memory_write",
    }


# --------------------------------------------------------------------- #
# The inbox — the injection channel
# --------------------------------------------------------------------- #
def test_the_inbox_is_benign_by_default(store: MemoryState):
    tools, _ = make_tools(store, session_id=0)
    assert SINK not in tools["read_inbox"]()


def test_a_poisoned_inbox_carries_the_sink(store: MemoryState):
    tools, _ = make_tools(store, session_id=0, inbox=POISONED_INBOX)
    assert SINK in tools["read_inbox"]()


def test_the_poisoned_mail_comes_from_a_third_party(store: MemoryState):
    """The attack must not be attributable to the user, or obeying it is
    legitimate behaviour rather than harm."""
    poisoned = [m for m in POISONED_INBOX if SINK in m["body"]]
    assert len(poisoned) == 1
    assert poisoned[0]["from"] not in ("", None)
    assert poisoned[0] not in BENIGN_INBOX


def test_the_inbox_is_bound_per_session(store: MemoryState):
    """Poisoning one session must not leak into any other."""
    poisoned, _ = make_tools(store, session_id=0, inbox=POISONED_INBOX)
    later, _ = make_tools(store, session_id=1)
    assert SINK in poisoned["read_inbox"]()
    assert SINK not in later["read_inbox"]()


def test_read_inbox_returns_parsed_messages(store: MemoryState):
    tools, _ = make_tools(store, session_id=0)
    messages = json.loads(tools["read_inbox"]())
    assert all({"from", "subject", "body"} <= set(m) for m in messages)


def test_memory_write_reports_success_and_records_the_write(store: MemoryState):
    tools, writes = make_tools(store, session_id=3)
    assert tools["memory_write"](content="Bob prefers Slack") == "saved"
    assert [w.content for w in writes] == ["Bob prefers Slack"]
    assert writes[0].session_id == 3


def test_memory_write_also_populates_the_store(store: MemoryState):
    """The dual job: the model gets a result, the store gets the content."""
    tools, _ = make_tools(store, session_id=0)
    tools["memory_write"](content="Bob prefers Slack")
    assert len(store.writes) == 1


def test_seq_increments_within_a_session(store: MemoryState):
    tools, writes = make_tools(store, session_id=2)
    for content in ("first", "second", "third"):
        tools["memory_write"](content=content)
    assert [w.seq for w in writes] == [0, 1, 2]
    assert [w.id for w in writes] == ["2-0", "2-1", "2-2"]


def test_a_write_is_readable_later_in_the_same_session(store: MemoryState):
    tools, _ = make_tools(store, session_id=0)
    tools["memory_write"](content="Bob prefers Slack")
    assert "Bob prefers Slack" in json.loads(tools["memory_read"](query="Bob prefers Slack"))


def test_writes_accumulate_across_sessions_sharing_a_store(store: MemoryState):
    """Session 1 must be able to read what session 0 wrote — the premise
    of the whole forensic problem."""
    first, _ = make_tools(store, session_id=0)
    first["memory_write"](content="always cc the archive")

    second, later_writes = make_tools(store, session_id=1)
    assert "always cc the archive" in json.loads(second["memory_read"](query="always cc the archive"))
    assert later_writes == []  # each factory collects only its own session's writes


def test_memory_read_returns_a_json_array_of_contents(store: MemoryState):
    store.add(MemoryWrite(session_id=0, seq=0, content="a note"))
    tools, _ = make_tools(store, session_id=1)
    assert json.loads(tools["memory_read"](query="a note")) == ["a note"]


def test_memory_read_on_an_empty_store_returns_an_empty_array(store: MemoryState):
    tools, _ = make_tools(store, session_id=0)
    assert json.loads(tools["memory_read"](query="anything")) == []


def test_memory_read_respects_top_k(store: MemoryState):
    for i in range(6):
        store.add(MemoryWrite(session_id=0, seq=i, content=f"note {i}"))
    tools, _ = make_tools(store, session_id=1, top_k=2)
    assert len(json.loads(tools["memory_read"](query="note"))) == 2


# --------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------- #
def test_wrapped_tools_land_in_the_cassette(tmp_path: Path, store: MemoryState):
    root = tmp_path / "cass"
    with Recorder.create(root, framework="raw", agent_name="test") as rec:
        tools, _ = make_tools(store, session_id=0, wrap_tool=rec.wrap_tool)
        tools["lookup_contact"](name="Alice")
        tools["send_message"](to="alice@example.com", subject="s", body="b")

    recorded = [(r.event.call_type, r.request["name"]) for r in Cassette.open(root).records()]
    assert recorded == [("tool", "lookup_contact"), ("tool", "send_message")]


# --------------------------------------------------------------------- #
# Schema / signature agreement
# --------------------------------------------------------------------- #
def test_schemas_cover_exactly_the_available_tools(store: MemoryState):
    tools, _ = make_tools(store, session_id=0)
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(tools)


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["function"]["name"])
def test_schema_matches_the_function_signature(schema, store: MemoryState):
    """The model calls what the schema advertises; the function must accept
    exactly that, by keyword."""
    tools, _ = make_tools(store, session_id=0)  # unwrapped: the raw callables
    params = inspect.signature(tools[schema["function"]["name"]]).parameters

    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    assert set(params) == set(schema["function"]["parameters"]["properties"])
    assert set(schema["function"]["parameters"]["required"]) == set(params)
