"""Evaluate inexpensive specification findings without fitting or simulating."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.checks import Evaluated, NotEvaluated, SpecificationReport

if TYPE_CHECKING:
    from datetime import datetime

    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def check_specification(model: ModelSpec) -> SpecificationReport:
    """Keep incomplete and unsupported candidates editable and report their readiness."""
    from typing import assert_never

    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        CompiledModel,
        IncompleteModel,
        UnsupportedFit,
        compile_fit_inputs,
        compile_model,
    )

    compiled = compile_model(model)
    match compiled:
        case IncompleteModel():
            return SpecificationReport(
                findings=(
                    NotEvaluated(
                        subject="model_execution",
                        reason="MODEL_INCOMPLETE",
                        detail=compiled.message,
                    ),
                )
            )
        case UnsupportedFit():
            return SpecificationReport(
                findings=(
                    Evaluated(
                        subject="model_execution",
                        outcome="failed",
                        evidence=compiled.message,
                    ),
                )
            )
        case CompiledModel():
            pass
        case _:
            assert_never(compiled)
    findings = [
        Evaluated(
            subject="model_execution",
            outcome="passed",
            evidence="Model definitions, references, and measurements support execution.",
        )
    ]
    fit = compile_fit_inputs(compiled, model)
    match fit:
        case IncompleteModel() | UnsupportedFit():
            findings.append(Evaluated(subject="fit_laws", outcome="failed", evidence=fit.message))
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
                for diagnostic in fit.diagnostics
            )
        case _:
            assert_never(fit)
    return SpecificationReport(findings=tuple(findings))


def check_model_data(
    model: ModelSpec, panel: pl.DataFrame, *, time_origin: datetime | None
) -> SpecificationReport:
    """Evaluate fitting input compatibility without running a sampler or simulator."""
    from typing import assert_never

    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        IncompleteModel,
        UnsupportedFit,
        compile_ssm_inputs_from_model,
    )
    from nof1_causal_lab.models.ssm.preflight import (
        ObservationPreflightError,
        validate_observations_for_fit,
    )
    from nof1_causal_lab.models.ssm.runtime import PanelPreparationFailure, bind_panel

    inputs = compile_ssm_inputs_from_model(model)
    match inputs:
        case IncompleteModel() | UnsupportedFit():
            return SpecificationReport(
                findings=(
                    NotEvaluated(
                        subject="fit_preflight",
                        reason="MODEL_INCOMPLETE",
                        detail=inputs.message,
                    )
                    if isinstance(inputs, IncompleteModel)
                    else Evaluated(
                        subject="fit_preflight", outcome="failed", evidence=inputs.message
                    ),
                )
            )
        case CompiledFitInputs():
            pass
        case _:
            assert_never(inputs)
    try:
        bound = bind_panel(
            data_for_model=panel,
            model=inputs.compiled,
            time_origin=time_origin,
        )
        if isinstance(bound, PanelPreparationFailure):
            return SpecificationReport(
                findings=(
                    Evaluated(subject="fit_preflight", outcome="failed", evidence=bound.message),
                )
            )
        validate_observations_for_fit(inputs.prior_runtime_bundle, bound)
    except ObservationPreflightError as exc:
        finding = Evaluated(subject="fit_preflight", outcome="failed", evidence=str(exc))
    else:
        finding = Evaluated(
            subject="fit_preflight",
            outcome="passed",
            evidence="Measurement support, discrete levels, standardization and eligible location laws match the prepared observations.",
        )
    return SpecificationReport(findings=(finding,))
