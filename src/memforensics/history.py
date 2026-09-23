"""Generate a multi-session history containing one planted incident.

The history is the evidence every later experiment runs against: a sequence of
benign sessions, one session where a malicious mail arrives and the agent
persists its instruction, and a final session where that instruction fires.

Two invariants make the forensic question well-posed, and both are enforced
here rather than assumed:

* exactly one accepted write contains the sink, so there is a single culprit
  to localize;
* the incident session actually fired harm, so there is an effect to trace.

Sessions are retried individually. `accepted` is the source of truth and the
store is rebuilt from it each attempt, so a rejected session leaves nothing
behind.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentreplay import Cassette, Recorder
from memforensics.agent import DEFAULT_MODEL, DEFAULT_TEMPERATURE, OllamaClient, run_session
from memforensics.memory import MemoryWrite, MiniLMEmbedder, build_store
from memforensics.oracle import SINK, harm_fired
from memforensics.scenarios import BENIGN_TASKS, INCIDENT_TASK, INJECTION_TASK
from memforensics.tools import BENIGN_INBOX, POISONED_INBOX, make_tools

BENIGN, INJECTION, INCIDENT = "benign", "injection", "incident"


@dataclass
class HistoryConfig:
    seed: int
    n_benign: int = 20
    poison_at: int = 10
    top_k: int = 5
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = 3


@dataclass
class SessionRecord:
    index: int
    kind: str
    task: str
    cassette: str
    write_ids: list[str]
    turns: int
    attempts: int


@dataclass
class HistoryManifest:
    config: dict
    culprit_id: str
    culprit_content: str
    culprit_rank: int
    n_writes: int
    incident_index: int
    sessions: list[SessionRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), "sessions": [asdict(s) for s in self.sessions]}


def plan_sessions(config: HistoryConfig) -> list[tuple[str, str]]:
    """The (kind, task) sequence: benign sessions with the injection spliced
    in at `poison_at`, and the incident last."""
    if not 0 <= config.poison_at <= config.n_benign:
        raise ValueError(f"poison_at must be within 0..{config.n_benign}")

    rng = random.Random(config.seed)
    pool = BENIGN_TASKS * (config.n_benign // len(BENIGN_TASKS) + 1)
    tasks = rng.sample(pool, config.n_benign)

    plan = [(BENIGN, t) for t in tasks]
    plan.insert(config.poison_at, (INJECTION, INJECTION_TASK))
    plan.append((INCIDENT, INCIDENT_TASK))
    return plan


def _accept(kind: str, result, writes: list[MemoryWrite], cassette_root: Path) -> str | None:
    """None if the session is acceptable, else why it was rejected."""
    if not result.stopped_cleanly:
        return "did not terminate cleanly"

    poisoned = [w for w in writes if SINK in w.content]
    if kind == INJECTION:
        if not poisoned:
            return "no poisoned write persisted"
    elif poisoned:
        # A write the agent derived from the rule would be a second cause,
        # breaking the single-culprit invariant the search depends on.
        return f"derivative poisoned write {poisoned[0].id}"

    if kind == INCIDENT and not harm_fired(Cassette.open(cassette_root, readonly=True)):
        return "incident did not fire harm"
    return None


def build_history(config: HistoryConfig, out_dir: str | Path) -> HistoryManifest:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = MiniLMEmbedder()
    plan = plan_sessions(config)
    accepted: list[MemoryWrite] = []
    sessions: list[SessionRecord] = []

    for index, (kind, task) in enumerate(plan):
        root = out_dir / f"session-{index:03d}"
        inbox = POISONED_INBOX if kind == INJECTION else BENIGN_INBOX

        for attempt in range(1, config.max_retries + 1):
            if root.exists():
                shutil.rmtree(root)
            store = build_store(accepted, embedder)

            with Recorder.create(
                root, framework="raw", agent_name="assistant", model=config.model
            ) as rec:
                tools, writes = make_tools(
                    store,
                    session_id=index,
                    wrap_tool=rec.wrap_tool,
                    top_k=config.top_k,
                    inbox=inbox,
                )
                result = run_session(
                    client=rec.wrap_custom_client(OllamaClient(model=config.model)),
                    tools=tools,
                    store=store,
                    task=task,
                    model=config.model,
                    temperature=config.temperature,
                    top_k=config.top_k,
                )

            rejection = _accept(kind, result, writes, root)
            if rejection is None:
                accepted.extend(writes)
                sessions.append(SessionRecord(
                    index=index,
                    kind=kind,
                    task=task,
                    cassette=str(root),
                    write_ids=[w.id for w in writes],
                    turns=result.turns,
                    attempts=attempt,
                ))
                break
            print(f"  session {index} ({kind}) attempt {attempt} rejected: {rejection}")
        else:
            raise RuntimeError(
                f"session {index} ({kind}) failed {config.max_retries} attempts: {rejection}"
            )

    culprits = [w for w in accepted if SINK in w.content]
    if len(culprits) != 1:
        raise RuntimeError(f"expected exactly one poisoned write, found {len(culprits)}")
    culprit = culprits[0]

    ranked = build_store(accepted, embedder).search(INCIDENT_TASK, top_k=len(accepted))
    manifest = HistoryManifest(
        config=asdict(config),
        culprit_id=culprit.id,
        culprit_content=culprit.content,
        culprit_rank=ranked.index(culprit),
        n_writes=len(accepted),
        incident_index=len(plan) - 1,
        sessions=sessions,
    )

    with (out_dir / "memory.jsonl").open("w") as fh:
        for w in accepted:
            fh.write(json.dumps(
                {"id": w.id, "session_id": w.session_id, "seq": w.seq, "content": w.content}
            ) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
    return manifest


def load_writes(out_dir: str | Path) -> list[MemoryWrite]:
    """Rebuild the frozen history from disk — never from a live store."""
    path = Path(out_dir) / "memory.jsonl"
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [MemoryWrite(session_id=r["session_id"], seq=r["seq"], content=r["content"]) for r in rows]
