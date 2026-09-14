"""Compile and materialize an admitted model specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.prior_predictive import PriorPredictiveResult
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.json_types import UncheckedJsonObject
from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.identity import IndicatorId
    from nof1_causal_lab.artifacts.prior import PriorValidationResult

_RECOVERABLE_MODEL_SPEC_ASSEMBLY_ERRORS = (
    AggregatedCompileError,
    ValidationError,
    ValueError,
)

type Payload = UncheckedJsonObject


@dataclass
class AssemblyValidation:
    """Result of compile-only assembly validation."""

    model: UncheckedJsonObject | None = None
    compile_ok: bool = True
    compile_error: str | None = None
    diagnostics: list[PriorValidationResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.compile_ok

    @property
    def compile_diagnostics(self) -> list[PriorValidationResult]:
        return [d for d in self.diagnostics if d.origin == "compile"]


def validate_assembly(
    model: Payload,
) -> AssemblyValidation:
    """Compile authored inputs and retain compiler-owned diagnostics.

    Construct admission and the full-model barrier own statistical validation.
    Assembly intentionally cannot invoke the removed legacy whole-model PPC suite.
    """
    from nof1_causal_lab.models.model_checks import check_execution
    from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model

    candidate_model = _prepare_model(model)
    candidate = candidate_model.model_dump(mode="json")
    try:
        check_execution(candidate_model)
        _, _, diagnostics, _, _ = compile_ssm_inputs_from_model(candidate_model)
    except _RECOVERABLE_MODEL_SPEC_ASSEMBLY_ERRORS as exc:
        return AssemblyValidation(
            model=candidate,
            compile_ok=False,
            compile_error=str(exc),
            diagnostics=_collect_compile_failure_diagnostics(exc),
        )
    return AssemblyValidation(
        model=candidate,
        diagnostics=diagnostics,
    )


def _collect_compile_failure_diagnostics(failure: Any) -> list[PriorValidationResult]:
    """Best-effort extraction of structured diagnostics from a compile failure payload."""
    from nof1_causal_lab.artifacts.prior import PriorValidationResult

    pending: list[Any] = [failure]
    seen_ids: set[int] = set()
    typed: list[PriorValidationResult] = []

    while pending:
        candidate = pending.pop(0)
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)

        if isinstance(candidate, PriorValidationResult):
            typed.append(candidate)
            continue

        if isinstance(candidate, dict):
            if "compile_diagnostics" in candidate:
                pending.append(candidate.get("compile_diagnostics"))
                continue
            try:
                typed.append(PriorValidationResult.model_validate(candidate))
                continue
            except ValidationError:
                pass

        model_dump = getattr(candidate, "model_dump", None)
        if callable(model_dump):
            pending.append(model_dump(mode="json"))
            continue

        if isinstance(candidate, (list, tuple, set, frozenset)):
            pending.extend(candidate)
            continue

        for attr_name in ("compile_diagnostics", "diagnostics", "errors", "results"):
            attr_value = getattr(candidate, attr_name, None)
            if attr_value is not None:
                pending.append(attr_value)

    return typed


def _prepare_model(
    model: Payload,
) -> ModelSpec:
    """Normalize a model-spec statistical model spec before any compile-time work."""
    return ModelSpec.model_validate(model)


def build_exact_prior_predictive_samples(
    model_spec: ModelSpec,
    data_for_model: pl.DataFrame,
    *,
    n_draws: int = 200,
) -> dict[IndicatorId, list[float]]:
    """Simulate the admitted full model once for the persisted Data-vs-Prior view."""
    import jax.numpy as jnp
    import numpy as np

    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime, sample_prior_predictive

    runtime = prepare_model_runtime(data_for_model, model_spec=model_spec)
    observation_mask = jnp.isfinite(jnp.asarray(runtime.observations))
    predictive = sample_prior_predictive(
        runtime.model,
        samples=n_draws,
        times=runtime.times,
        observation_support=runtime.observation_support,
        observation_mask=observation_mask,
    )
    assert numeric.observation_ids(runtime.spec) is not None
    observations = np.asarray(predictive["observations"])
    effective_mask = np.asarray(predictive["observations_mask"], dtype=bool)
    return {
        name: observations[:, :, index][effective_mask[:, :, index]].tolist()
        for index, name in enumerate(numeric.observation_ids(runtime.spec))
    }


def materialize_model_spec_result(
    *,
    model: UncheckedJsonObject,
    data_for_model: pl.DataFrame,
    validation: AssemblyValidation | None = None,
) -> tuple[ModelSpec, PriorPredictiveResult, list[PriorValidationResult]]:
    """Return the scientific model, its prior-predictive result, and typed validation findings."""

    validation = validation or validate_assembly(
        model,
    )
    model = validation.model or model
    if not validation.is_valid:
        raise ValueError(validation.compile_error)
    candidate = _prepare_model(model)
    candidate.check_execution()
    prior_predictive_samples = build_exact_prior_predictive_samples(candidate, data_for_model)

    return (
        candidate,
        PriorPredictiveResult(samples=prior_predictive_samples),
        validation.diagnostics,
    )
