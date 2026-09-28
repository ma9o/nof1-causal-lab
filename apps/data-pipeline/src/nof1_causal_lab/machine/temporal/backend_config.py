"""Resolve ingestion worker backend settings."""

from typing import Any

from nof1_causal_lab.machine.temporal.messages import LLMBackendConfig


def first_config_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def llm_backend_config(
    profile_llm: Any,
    defaults: Any,
    max_tool_turns: int | None,
) -> LLMBackendConfig:
    if profile_llm.harness == "none":
        embedded = defaults.embedded
        return LLMBackendConfig(
            harness="none",
            model=profile_llm.model,
            max_tokens=first_config_value(profile_llm.max_tokens, embedded.max_tokens),
            timeout=first_config_value(profile_llm.timeout, embedded.timeout),
            reasoning_effort=first_config_value(
                profile_llm.reasoning_effort,
                embedded.reasoning_effort,
            ),
        )

    if profile_llm.harness == "claude-code":
        claude = defaults.claude_code
        return LLMBackendConfig(
            harness="claude-code",
            model=profile_llm.model,
            bin=first_config_value(profile_llm.bin, claude.bin),
            effort=first_config_value(profile_llm.effort, claude.effort),
            max_turns=first_config_value(profile_llm.max_turns, max_tool_turns, claude.max_turns),
            max_budget_usd=first_config_value(
                profile_llm.max_budget_usd,
                claude.max_budget_usd,
            ),
            fallback_model=first_config_value(
                profile_llm.fallback_model,
                claude.fallback_model,
            ),
        )

    if profile_llm.harness == "codex":
        codex = defaults.codex
        return LLMBackendConfig(
            harness="codex",
            model=profile_llm.model,
            bin=first_config_value(profile_llm.bin, codex.bin),
            reasoning_effort=first_config_value(
                profile_llm.reasoning_effort,
                codex.reasoning_effort,
            ),
            service_tier=first_config_value(profile_llm.service_tier, codex.service_tier),
            timeout=profile_llm.timeout,
        )

    if profile_llm.harness == "pi":
        pi = defaults.pi
        return LLMBackendConfig(
            harness="pi",
            model=profile_llm.model,
            bin=first_config_value(profile_llm.bin, pi.bin),
            provider=first_config_value(profile_llm.provider, pi.provider),
            thinking=first_config_value(profile_llm.thinking, pi.thinking),
            timeout=profile_llm.timeout,
        )

    raise ValueError(f"unknown LLM harness {profile_llm.harness!r}")
