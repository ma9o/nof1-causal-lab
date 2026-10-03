"""Evaluate inexpensive specification findings without fitting or simulating."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.checks import Evaluated, NotEvaluated, SpecificationAssessment

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

    from nof1_causal_lab.models.ssm.compile.inputs import (
        FitCompilationResult,
        ModelCompilationResult,
    )


def check_specification(
    compiled: ModelCompilationResult, inputs: FitCompilationResult
) -> tuple[SpecificationAssessment, ...]:
    """Keep incomplete and unsupported candidates editable and report their readiness."""
    from typing import assert_never

    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        CompiledModel,
        IncompleteModel,
        UnsupportedFit,
    )

    match compiled:
        case IncompleteModel():
            return (
                NotEvaluated(
                    subject="model_execution",
                    reason="MODEL_INCOMPLETE",
                    detail=compiled.message,
                ),
            )
        case UnsupportedFit():
            return (
                Evaluated(subject="model_execution", outcome="failed", evidence=compiled.message),
            )
        case CompiledModel():
            pass
        case _:
            assert_never(compiled)
    findings: list[SpecificationAssessment] = [
        Evaluated(
            subject="model_execution",
            outcome="passed",
            evidence="Model definitions, references, and measurements support execution.",
        )
    ]
    match inputs:
        case IncompleteModel() | UnsupportedFit():
            findings.append(
                Evaluated(subject="fit_laws", outcome="failed", evidence=inputs.message)
            )
        case CompiledFitInputs():
            findings.append(
                Evaluated(
                    subject="fit_laws",
                    outcome="passed",
                    evidence="Parameter laws support the current fitting engine.",
                )
            )
            findings.extend(
                Evaluated(
                    subject=diagnostic.code,
                    outcome="passed" if diagnostic.is_valid else "failed",
                    evidence=diagnostic.issue or diagnostic.parameter,
                )
                for diagnostic in inputs.diagnostics
            )
        case _:
            assert_never(inputs)
    return tuple(findings)


def check_model_data(
    inputs: FitCompilationResult, panel: pl.DataFrame, *, time_origin: datetime | None
) -> tuple[SpecificationAssessment, ...]:
    """Evaluate fitting input compatibility without running a sampler or simulator."""
    from nof1_causal_lab.models.ssm.compile.inputs import IncompleteModel
    from nof1_causal_lab.models.ssm.preflight import validate_observations_for_fit
    from nof1_causal_lab.models.ssm.runtime import PreparedFit, prepare_fit

    prepared = prepare_fit(inputs, panel, time_origin=time_origin)
    if not isinstance(prepared, PreparedFit):
        return (
            NotEvaluated(
                subject="fit_preflight", reason="MODEL_INCOMPLETE", detail=prepared.message
            )
            if isinstance(prepared, IncompleteModel)
            else Evaluated(subject="fit_preflight", outcome="failed", evidence=prepared.message),
        )
    failure = validate_observations_for_fit(prepared.inputs.prior_runtime_bundle, prepared.panel)
    finding: SpecificationAssessment = (
        Evaluated(subject="fit_preflight", outcome="failed", evidence=failure.message)
        if failure is not None
        else Evaluated(
            subject="fit_preflight",
            outcome="passed",
            evidence="Measurement support, discrete levels, standardization and eligible location laws match the prepared observations.",
        )
    )
    return (finding,)
