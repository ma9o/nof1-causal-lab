"""Evaluate inexpensive specification findings without fitting or simulating."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.checks import SpecificationFinding, SpecificationReport
from nof1_causal_lab.compilation_errors import IncompleteModelError

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def check_specification(model: ModelSpec) -> SpecificationReport:
    """Keep incomplete and unsupported candidates editable and report their readiness."""
    from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

    findings = []
    try:
        model.check_execution()
    except IncompleteModelError as exc:
        findings.append(
            SpecificationFinding(check="model_execution", status="not_evaluated", message=str(exc))
        )
    except ValueError as exc:
        findings.append(
            SpecificationFinding(check="model_execution", status="failed", message=str(exc))
        )
    else:
        findings.append(
            SpecificationFinding(
                check="model_execution",
                status="passed",
                message="Model definitions, references, measurements, and anchors support execution.",
            )
        )
        try:
            _, _, diagnostics = compile_priors(model)
        except ValueError as exc:
            findings.append(
                SpecificationFinding(check="fit_laws", status="failed", message=str(exc))
            )
        else:
            findings.append(
                SpecificationFinding(
                    check="fit_laws",
                    status="passed",
                    message="Parameter laws support the current fitting engine.",
                )
            )
            for diagnostic in diagnostics:
                findings.append(
                    SpecificationFinding(
                        check=diagnostic.code,
                        status="passed" if diagnostic.is_valid else "failed",
                        message=diagnostic.issue or diagnostic.parameter,
                    )
                )
    return SpecificationReport(findings=tuple(findings))


def check_model_data(model: ModelSpec, panel) -> SpecificationReport:
    """Evaluate fitting input compatibility without running a sampler or simulator."""
    from nof1_causal_lab.models.ssm.preflight import validate_observations_for_fit
    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime

    try:
        model.check_execution()
    except (IncompleteModelError, ValueError) as exc:
        return SpecificationReport(
            findings=(
                SpecificationFinding(
                    check="fit_preflight", status="not_evaluated", message=str(exc)
                ),
            )
        )
    try:
        runtime = prepare_model_runtime(data_for_model=panel, model_spec=model)
        validate_observations_for_fit(runtime.model, runtime.observations)
    except ValueError as exc:
        finding = SpecificationFinding(check="fit_preflight", status="failed", message=str(exc))
    else:
        finding = SpecificationFinding(
            check="fit_preflight",
            status="passed",
            message="Measurement support, discrete levels, standardization and eligible location laws match the prepared observations.",
        )
    return SpecificationReport(findings=(finding,))
