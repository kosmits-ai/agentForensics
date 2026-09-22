"""Framework adapters.

Each adapter wraps a popular agent framework's LLM client / tool entry
points so a team using that framework can add AgentReplay with a single
two-line initialization, as described in §5.5 of the product proposal.

Supported frameworks:

    * :mod:`agentreplay.frameworks.openai_sdk`     — OpenAI Agents SDK / raw OpenAI client
    * :mod:`agentreplay.frameworks.anthropic_sdk`  — Anthropic SDK / Anthropic Agents SDK
    * :mod:`agentreplay.frameworks.langgraph`      — LangGraph (first-class target)
    * :mod:`agentreplay.frameworks.crewai`         — CrewAI
    * :mod:`agentreplay.frameworks.autogen`        — AutoGen (v0.2 and v0.4+)
    * :mod:`agentreplay.frameworks.raw`            — custom / framework-less agents

The adapters are intentionally *lazy*: importing ``agentreplay.frameworks``
does NOT import OpenAI / Anthropic / LangGraph / CrewAI / AutoGen. Each
adapter only imports its framework when called, so teams that do not use
a framework pay no import cost for it.
"""
from agentreplay.frameworks.raw import wrap_raw_client


def __getattr__(name: str):  # pragma: no cover - lazy module loader
    if name == "wrap_openai":
        from agentreplay.frameworks.openai_sdk import wrap_openai
        return wrap_openai
    if name == "wrap_anthropic":
        from agentreplay.frameworks.anthropic_sdk import wrap_anthropic
        return wrap_anthropic
    raise AttributeError(name)


__all__ = ["wrap_raw_client"]
