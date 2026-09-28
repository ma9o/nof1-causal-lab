"""Tests for config.py: dataclass methods and load_config parsing."""

import textwrap

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.llm_specs import CodexLLMSpec, EmbeddedLLMSpec, LLMProfileSpec, PiLLMSpec
from nof1_causal_lab.machine.temporal.backend_config import llm_backend_config
from nof1_causal_lab.machine.temporal.messages import LLMSubroutineInput
from nof1_causal_lab.sampler_config import validate_sampler_config
from nof1_causal_lab.utils.config import (
    ClaudeCodeDefaults,
    CodexDefaults,
    EmbeddedLLMDefaults,
    ExtractionWorkersConfig,
    InferenceConfig,
    IngestionConfig,
    LLMDefaults,
    MarginalParticleGibbsConfig,
    PiDefaults,
    PipelineBehaviorConfig,
    PipelineConfig,
    PriorElicitationConfig,
    StructureProposalConfig,
    get_secret,
    get_secret_async,
    load_config,
    validate_config,
)
from tests.helpers import run_async

pytestmark = pytest.mark.contract

# =============================================================================
# InferenceConfig.to_sampler_config
# =============================================================================


class TestToSamplerConfig:
    def test_runtime_contract_rejects_unknown_fields(self):
        config = InferenceConfig().to_sampler_config()

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            validate_sampler_config({**config, "unknown_option": True})

    def test_marginal_particle_gibbs_explicit(self):
        cfg = InferenceConfig(method="marginal_particle_gibbs")
        result = cfg.to_sampler_config()
        assert result["method"] == "marginal_particle_gibbs"
        assert result["n_particles"] == 64
        assert result["n_parameter_particles"] == 2
        assert result["latent_smoother"] == "dsmc"
        assert result["dsmc_leaf_proposal"] == "amala_exact"
        assert result["latent_delta"] == 0.2
        assert result["amala_kappa"] == 0.75
        assert result["amala_grad_clip"] == float("inf")
        assert result["param_step_size"] == 0.02
        assert result["param_target_accept"] == 0.35
        assert result["latent_init_method"] == "predictive"
        assert result["retain_latent_paths"] is True

    def test_unknown_method_raises(self):
        cfg = InferenceConfig()
        with pytest.raises(ValidationError, match="method"):
            validate_sampler_config({**cfg.to_sampler_config(), "method": "hmc"})

    def test_custom_chains_and_seed(self):
        cfg = InferenceConfig(num_chains=8, seed=42)
        result = cfg.to_sampler_config()
        assert result["num_chains"] == 8
        assert result["seed"] == 42

    def test_marginal_particle_gibbs_settings(self):
        cfg = InferenceConfig(
            method="marginal_particle_gibbs",
            marginal_particle_gibbs=MarginalParticleGibbsConfig(
                n_particles=17,
                n_parameter_particles=3,
                latent_smoother="dsmc",
                latent_delta=0.31,
                amala_kappa=0.25,
                amala_grad_clip=55.0,
                param_step_size=0.04,
                param_step_size_min=1e-5,
                param_step_size_max=0.5,
                param_target_accept=0.42,
                adaptation_rate=0.08,
                init_method="random",
                pathfinder_num_elbo_samples=9,
                pathfinder_maxiter=10,
                n_pathfinder_starts=2,
                pathfinder_init_scale=None,
                auto_preconditioner_method="none",
                auto_preconditioner_maxiter=11,
            ),
        )
        result = cfg.to_sampler_config()
        assert result["method"] == "marginal_particle_gibbs"
        assert result["n_particles"] == 17
        assert result["n_parameter_particles"] == 3
        assert result["latent_smoother"] == "dsmc"
        assert result["latent_delta"] == 0.31
        assert result["amala_kappa"] == 0.25
        assert result["amala_grad_clip"] == 55.0
        assert result["param_step_size"] == 0.04
        assert result["param_step_size_min"] == 1e-5
        assert result["param_step_size_max"] == 0.5
        assert result["param_target_accept"] == 0.42
        assert result["adaptation_rate"] == 0.08
        assert result["init_method"] == "random"
        assert result["latent_init_method"] == "predictive"
        assert result["pathfinder_num_elbo_samples"] == 9
        assert result["pathfinder_maxiter"] == 10
        assert result["n_pathfinder_starts"] == 2
        assert result["pathfinder_init_scale"] is None
        assert result["auto_preconditioner_method"] == "none"
        assert result["auto_preconditioner_maxiter"] == 11


# =============================================================================
# load_config (with temp config file)
# =============================================================================


MINIMAL_CONFIG = textwrap.dedent("""\
    ingestion:
      llm:
        harness: none
        model: openrouter/gpt-4
    structure_proposal:
      sample_chunks: 3
      chunk_size: 500
      llm:
        harness: none
        model: openrouter/gpt-4
    extraction_workers:
      chunk_size: 300
      llm:
        harness: none
        model: openrouter/gpt-4
    prior_elicitation:
      llm:
        harness: none
        model: openrouter/gpt-4
""")

FULL_CONFIG = textwrap.dedent("""\
    llm:
      embedded:
        max_tokens: 4096
        timeout: 120
        reasoning_effort: low
      claude_code:
        effort: medium
      codex:
        reasoning_effort: medium
        service_tier: fast
      pi:
        bin: pi-custom
        provider: openai-codex
        thinking: high

    ingestion:
      max_tool_turns: 30
      llm:
        harness: none
        model: openrouter/claude-3


    structure_proposal:
      sample_chunks: 5
      chunk_size: 800
      latent_max_tool_turns: 25
      measurement_max_tool_turns: 35
      llm:
        harness: none
        model: openrouter/claude-3

    extraction_workers:
      chunk_size: 400
      max_concurrent_workers: 6
      max_tool_turns: 45
      llm:
        harness: none
        model: openrouter/claude-3

    prior_elicitation:
      max_tool_turns: 100
      literature_search:
        enabled: false
      llm:
        harness: none
        model: openrouter/claude-3

    inference:
      method: marginal_particle_gibbs
      num_warmup: 500
      num_samples: 2000
      num_chains: 2
      seed: 123
      compute_loo_diagnostics: false
      map:
        n_ieks_iters: 10
      marginal_particle_gibbs:
        n_particles: 24
        n_parameter_particles: 3
        latent_smoother: dsmc
        latent_delta: 0.29
        amala_kappa: 0.2
        amala_grad_clip: 77.0
        param_step_size: 0.03
        param_step_size_min: 0.000001
        param_step_size_max: 0.7
        param_target_accept: 0.4
        adaptation_rate: 0.03
        init_method: random
        pathfinder_num_elbo_samples: 7
        pathfinder_maxiter: 8
        n_pathfinder_starts: 2
        pathfinder_init_scale:
        auto_preconditioner_method: none
        auto_preconditioner_maxiter: 13
""")


class TestLoadConfig:
    @pytest.mark.parametrize(
        "invalid",
        [
            "llm:\n  embedded:\n    reasoning_effort: invalid\n",
            "llm:\n  codex:\n    reasoning_effort: invalid\n",
            "inference:\n  method: map\n",
            "inference:\n  marginal_particle_gibbs:\n    init_method: invalid\n",
            "inference:\n  num_chains: many\n",
            "inference:\n  unknown_setting: true\n",
        ],
    )
    def test_rejects_invalid_yaml_before_returning_typed_config(self, tmp_path, invalid):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_CONFIG + invalid)
        load_config.cache_clear()
        with pytest.raises(ValidationError):
            load_config(path)

    def test_load_minimal(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(MINIMAL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.structure_proposal.llm.model == "openrouter/gpt-4"
        assert cfg.structure_proposal.llm.harness == "none"
        assert cfg.structure_proposal.sample_chunks == 3
        assert cfg.structure_proposal.latent_max_tool_turns == 40
        assert cfg.structure_proposal.measurement_max_tool_turns == 40
        assert cfg.extraction_workers.chunk_size == 300
        assert cfg.extraction_workers.max_concurrent_workers == 4
        assert cfg.extraction_workers.max_tool_turns == 40
        assert cfg.prior_elicitation.llm.model == "openrouter/gpt-4"
        assert cfg.prior_elicitation.max_tool_turns == 40
        # Defaults for optional sections
        assert cfg.inference.method == "marginal_particle_gibbs"
        assert cfg.llm.embedded.max_tokens == 65536
        assert cfg.llm.codex.reasoning_effort == "xhigh"
        assert cfg.llm.codex.service_tier == "fast"
        assert cfg.llm.pi.provider == "openai-codex"
        assert cfg.llm.pi.thinking == "high"

        load_config.cache_clear()

    def test_load_full(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.ingestion.max_tool_turns == 30
        assert cfg.structure_proposal.latent_max_tool_turns == 25
        assert cfg.structure_proposal.measurement_max_tool_turns == 35
        assert cfg.extraction_workers.max_concurrent_workers == 6
        assert cfg.extraction_workers.max_tool_turns == 45
        assert cfg.prior_elicitation.max_tool_turns == 100
        assert cfg.prior_elicitation.literature_search.enabled is False
        assert cfg.inference.method == "marginal_particle_gibbs"
        assert cfg.inference.num_warmup == 500
        assert cfg.inference.num_samples == 2000
        assert cfg.inference.num_chains == 2
        assert cfg.inference.seed == 123
        assert cfg.inference.compute_loo_diagnostics is False
        assert cfg.inference.map.n_ieks_iters == 10
        assert cfg.inference.marginal_particle_gibbs.n_particles == 24
        assert cfg.inference.marginal_particle_gibbs.n_parameter_particles == 3
        assert cfg.inference.marginal_particle_gibbs.latent_smoother == "dsmc"
        assert cfg.inference.marginal_particle_gibbs.latent_delta == 0.29
        assert cfg.inference.marginal_particle_gibbs.amala_kappa == 0.2
        assert cfg.inference.marginal_particle_gibbs.amala_grad_clip == 77.0
        assert cfg.inference.marginal_particle_gibbs.param_step_size == 0.03
        assert cfg.inference.marginal_particle_gibbs.param_step_size_min == 1e-6
        assert cfg.inference.marginal_particle_gibbs.param_step_size_max == 0.7
        assert cfg.inference.marginal_particle_gibbs.param_target_accept == 0.4
        assert cfg.inference.marginal_particle_gibbs.adaptation_rate == 0.03
        assert cfg.inference.marginal_particle_gibbs.init_method == "random"
        assert cfg.inference.marginal_particle_gibbs.latent_init_method == "predictive"
        assert cfg.inference.marginal_particle_gibbs.pathfinder_num_elbo_samples == 7
        assert cfg.inference.marginal_particle_gibbs.pathfinder_maxiter == 8
        assert cfg.inference.marginal_particle_gibbs.n_pathfinder_starts == 2
        assert cfg.inference.marginal_particle_gibbs.pathfinder_init_scale is None
        assert cfg.inference.marginal_particle_gibbs.auto_preconditioner_method == "none"
        assert cfg.inference.marginal_particle_gibbs.auto_preconditioner_maxiter == 13
        assert cfg.llm.embedded.max_tokens == 4096
        assert cfg.llm.embedded.reasoning_effort == "low"
        assert cfg.llm.claude_code.effort == "medium"
        assert cfg.llm.codex.reasoning_effort == "medium"
        assert cfg.llm.codex.service_tier == "fast"
        assert cfg.llm.pi == PiDefaults(bin="pi-custom", provider="openai-codex", thinking="high")

        load_config.cache_clear()

    def test_sampler_config_roundtrip(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        sampler = cfg.inference.to_sampler_config()
        assert sampler["method"] == "marginal_particle_gibbs"
        assert sampler["num_warmup"] == 500
        assert sampler["n_ieks_iters"] == 10

        load_config.cache_clear()


# =============================================================================
# validate_config
# =============================================================================


def _make_pipeline_config(**profile_llm_overrides) -> PipelineConfig:
    """Build a valid PipelineConfig with optional per-context llm overrides."""
    defaults = {
        "ingestion": EmbeddedLLMSpec(harness="none", model="openrouter/x"),
        "structure_proposal": EmbeddedLLMSpec(harness="none", model="openrouter/x"),
        "extraction_workers": EmbeddedLLMSpec(harness="none", model="openrouter/x"),
        "prior_elicitation": EmbeddedLLMSpec(harness="none", model="openrouter/x"),
    }
    defaults.update(profile_llm_overrides)
    return PipelineConfig(
        ingestion=IngestionConfig(llm=defaults["ingestion"]),
        structure_proposal=StructureProposalConfig(llm=defaults["structure_proposal"]),
        extraction_workers=ExtractionWorkersConfig(llm=defaults["extraction_workers"]),
        prior_elicitation=PriorElicitationConfig(llm=defaults["prior_elicitation"]),
        inference=InferenceConfig(),
        llm=LLMDefaults(
            embedded=EmbeddedLLMDefaults(),
            claude_code=ClaudeCodeDefaults(),
            codex=CodexDefaults(),
        ),
        pipeline=PipelineBehaviorConfig(),
    )


class TestValidateConfig:
    def test_happy_path_all_embedded(self):
        config = _make_pipeline_config()
        assert validate_config(config) == []

    @pytest.mark.parametrize("harness", ["claude-code", "codex", "pi"])
    def test_extraction_workers_require_embedded_backend(self, harness):
        with pytest.raises(ValidationError, match=r"llm\.harness"):
            TypeAdapter(ExtractionWorkersConfig).validate_python(
                {"llm": {"harness": harness, "model": "worker-model"}}
            )

    def test_embedded_model_must_be_openrouter_prefix(self):
        config = _make_pipeline_config(
            ingestion=EmbeddedLLMSpec(model="gpt-5.4"),
        )
        assert any("openrouter/" in error for error in validate_config(config))

    @pytest.mark.parametrize(
        "payload",
        [
            {"harness": "anthropic"},
            {"harness": "none", "effort": "high"},
            {"harness": "none", "service_tier": "fast"},
            {"harness": "none", "reasoning_effort": "max"},
            {"harness": "claude-code", "reasoning_effort": "high"},
            {"harness": "claude-code", "effort": "ultra"},
            {"harness": "codex", "max_budget_usd": 5.0},
            {"harness": "codex", "reasoning_effort": "minimal"},
            {"harness": "pi", "reasoning_effort": "high"},
            {"harness": "pi", "thinking": "max"},
        ],
    )
    def test_profile_schema_rejects_incompatible_backend_fields(self, payload):
        with pytest.raises(ValidationError):
            TypeAdapter(LLMProfileSpec).validate_python({"model": "openrouter/x", **payload})

    def test_pi_accepts_provider_model_and_thinking(self):
        config = _make_pipeline_config(
            ingestion=PiLLMSpec(
                provider="openai-codex",
                model="gpt-5.4-mini",
                thinking="high",
                timeout=3600,
            ),
        )
        assert validate_config(config) == []

    def test_codex_max_effort_survives_default_resolution_and_transport(self):
        llm = llm_backend_config(
            CodexLLMSpec(model="codex-test", reasoning_effort="max"),
            LLMDefaults(codex=CodexDefaults(bin="custom-codex")),
            max_tool_turns=None,
        )
        request = LLMSubroutineInput(
            workspace_id="test",
            run_id="run",
            subroutine_id="raw_data",
            context_kind="raw_data_ingestion",
            context_ref="context.json",
            llm=llm,
            max_tool_turns=5,
        )
        restored = LLMSubroutineInput.model_validate_json(request.model_dump_json())
        assert restored.llm == CodexLLMSpec(
            model="codex-test",
            bin="custom-codex",
            reasoning_effort="max",
            service_tier="fast",
        )

    def test_load_config_raises_on_stage2_harness_violation(self, tmp_path, monkeypatch):
        bad_config = textwrap.dedent("""\
            ingestion:
              llm:
                harness: none
                model: openrouter/gpt-4
            structure_proposal:
              sample_chunks: 3
              chunk_size: 500
              llm:
                harness: none
                model: openrouter/gpt-4
            extraction_workers:
              chunk_size: 300
              llm:
                harness: claude-code
                model: sonnet
            prior_elicitation:
              llm:
                harness: none
                model: openrouter/gpt-4
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(bad_config)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        with pytest.raises(ValueError, match=r"extraction_workers\.llm\.harness"):
            load_config()

        load_config.cache_clear()


# =============================================================================
# get_secret
# =============================================================================


class TestGetSecret:
    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")
        assert get_secret("TEST_SECRET_ABC") == "from-env"

    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ_789", raising=False)
        assert get_secret("DEFINITELY_NOT_SET_XYZ_789") is None

    def test_async_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")
        assert run_async(get_secret_async("TEST_SECRET_ABC")) == "from-env"


# =============================================================================
# ensure_harness_prereqs
# =============================================================================


class TestEnsureHarnessPrereqs:
    def _reset(self):
        from nof1_causal_lab.utils import config as config_module

        config_module._verified_harnesses.clear()

    def test_missing_openrouter_key_raises_for_embedded(self, monkeypatch):
        from nof1_causal_lab.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            ensure_harness_prereqs("none")

    def test_caches_successful_check(self, monkeypatch):
        """Once verified, removing the env var doesn't re-trigger the check."""
        from nof1_causal_lab.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Still cached — does not raise.
        ensure_harness_prereqs("none")
