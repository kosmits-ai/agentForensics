"""agentreplay — deterministic replay & counterfactual debugging for AI agents.

Public API:
    from agentreplay import Cassette, Recorder, Replayer, RecordingClient
    from agentreplay.frameworks import wrap_openai, wrap_anthropic

The library captures every non-deterministic input to an agent run (LLM
completions, tool/HTTP responses, clock, RNG) and guarantees that replaying
a recorded run reproduces the exact original trajectory — bit-for-bit,
with zero additional model calls.
"""
from agentreplay.cassette import Cassette, CassetteMeta, CassetteError
from agentreplay.constants import Mode, CallType
from agentreplay.diff import (
    Diff,
    FieldDiff,
    StepDiff,
    diff_structural,
    diff_payloads,
    render_diff,
)
from agentreplay.errors import (
    AgentReplayError,
    DivergenceError,
    CassetteNotFoundError,
    MutationError,
)
from agentreplay.hashing import canonicalize, canonical_json, hash_call_site, hash_payload, diff_keys
from agentreplay.interceptors import (
    ClockInterceptor,
    RNGInterceptor,
    RecordingClient,
    RecordingClock,
    RecordingHTTP,
    RecordingRandom,
    RecordingStream,
    RecordingTool,
    ReplayStream,
)
from agentreplay.logging import get_logger, set_verbose
from agentreplay.mutate import mutate_response, mutate_and_replay, apply_patch_set
from agentreplay.recorder import Recorder, StepContext
from agentreplay.replayer import Replayer
from agentreplay.session import Session
from agentreplay.storage import BlobStore, EventLog, MetaIndex
from agentreplay.types import (
    CallSiteID,
    Event,
    EventRecord,
    RequestPayload,
    ResponsePayload,
    StepID,
)

# CI regression runner (imported here so users can `from agentreplay import run_corpus`)
from agentreplay.ci import RegressionReport, RegressionResult, run_corpus

__all__ = [
    # Cassette + storage
    "Cassette",
    "CassetteMeta",
    "CassetteError",
    "BlobStore",
    "EventLog",
    "MetaIndex",
    # Hashing
    "canonicalize",
    "canonical_json",
    "hash_call_site",
    "hash_payload",
    "diff_keys",
    # Modes & types
    "Mode",
    "CallType",
    "CallSiteID",
    "Event",
    "EventRecord",
    "RequestPayload",
    "ResponsePayload",
    "StepID",
    # Recorder / Replayer / Session
    "Recorder",
    "Replayer",
    "Session",
    "StepContext",
    # Interceptors
    "RecordingClient",
    "RecordingTool",
    "RecordingHTTP",
    "RecordingClock",
    "RecordingRandom",
    "RecordingStream",
    "ReplayStream",
    "ClockInterceptor",
    "RNGInterceptor",
    # Counterfactual mutation
    "mutate_response",
    "mutate_and_replay",
    "apply_patch_set",
    # Errors
    "AgentReplayError",
    "DivergenceError",
    "CassetteNotFoundError",
    "MutationError",
    # Diff
    "Diff",
    "FieldDiff",
    "StepDiff",
    "diff_structural",
    "diff_payloads",
    "render_diff",
    # Logging
    "get_logger",
    "set_verbose",
    # Version
    "__version__",
    "run_corpus",
    "RegressionReport",
    "RegressionResult"
]

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("agentreplay")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
