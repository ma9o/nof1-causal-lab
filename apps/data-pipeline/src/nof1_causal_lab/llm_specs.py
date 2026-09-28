"""Backend-specific LLM definitions shared by configuration and orchestration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

type HarnessName = Literal["none", "claude-code", "codex", "pi"]
type EmbeddedReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
type HarnessEffort = Literal["low", "medium", "high", "xhigh", "max"]
type PiThinking = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


class EmbeddedLLMSpec(BaseModel):
    """OpenRouter model and its optional generation controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness: Literal["none"] = "none"
    model: str
    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: EmbeddedReasoningEffort | None = None


class ClaudeCodeLLMSpec(BaseModel):
    """Claude Code model and subprocess controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness: Literal["claude-code"] = "claude-code"
    model: str
    bin: str | None = None
    effort: HarnessEffort | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    fallback_model: str | None = None


class CodexLLMSpec(BaseModel):
    """Codex model and subprocess controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness: Literal["codex"] = "codex"
    model: str
    bin: str | None = None
    timeout: int | None = None
    reasoning_effort: HarnessEffort | None = None
    service_tier: str | None = None


class PiLLMSpec(BaseModel):
    """Pi provider, model, and subprocess controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness: Literal["pi"] = "pi"
    model: str
    bin: str | None = None
    timeout: int | None = None
    provider: str | None = None
    thinking: PiThinking | None = None


type HarnessLLMSpec = Annotated[
    ClaudeCodeLLMSpec | CodexLLMSpec | PiLLMSpec, Field(discriminator="harness")
]
type LLMProfileSpec = Annotated[
    EmbeddedLLMSpec | ClaudeCodeLLMSpec | CodexLLMSpec | PiLLMSpec,
    Field(discriminator="harness"),
]
