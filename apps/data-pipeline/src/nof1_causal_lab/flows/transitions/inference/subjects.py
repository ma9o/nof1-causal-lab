"""Attach scientific references to numerical findings before artifact persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.artifacts.identity import ParameterRef
from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    MCMCDiagnostics,
    PosteriorMarginal,
    PosteriorPair,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact
    from nof1_causal_lab.json_types import JsonObject

# Numerical-engine rows carry runtime coordinates until this boundary validates
# their public diagnostic schemas and replaces coordinates with scientific refs.
type RuntimeFindingPayload = dict[str, Any]


def reference_posterior_findings(
    compiled: CompiledSSMArtifact,
    marginals: list[RuntimeFindingPayload],
    pairs: list[RuntimeFindingPayload],
    mcmc: RuntimeFindingPayload | None,
) -> tuple[list[JsonObject], list[JsonObject], JsonObject | None]:
    """Translate the compiler's exact mapping once; unknown coordinates are errors."""
    parameters = {parameter.id: parameter for parameter in compiled.parameters}
    subjects = {
        coordinate: ParameterRef(parameter_id=binding.parameter_id, element_id=element_id)
        for binding in compiled.parameter_bindings
        for element_id, coordinate in binding.coordinates.items()
    }
    auxiliary = set(compiled.auxiliary_coordinates)

    def attach(row, fields):
        result = dict(row)
        for coordinate_field, subject_field, label_field in fields:
            coordinate = ParameterCoordinate.model_validate(result.pop(coordinate_field))
            if coordinate in auxiliary:
                return None
            if coordinate not in subjects:
                raise ValueError(
                    f"Posterior finding references an unbound runtime coordinate: {coordinate.label}"
                )
            subject = subjects[coordinate]
            result[subject_field] = subject.model_dump(mode="json")
            result[label_field] = parameters[subject.parameter_id].elements[subject.element_id]
        return result

    scalar_fields = (("coordinate", "subject", "parameter"),)
    marginal_rows = [
        PosteriorMarginal.model_validate(value).model_dump(mode="json")
        for row in marginals
        if (value := attach(row, scalar_fields)) is not None
    ]
    pair_rows = [
        PosteriorPair.model_validate(value).model_dump(mode="json")
        for row in pairs
        if (
            value := attach(
                row,
                (
                    ("coordinate_x", "subject_x", "param_x"),
                    ("coordinate_y", "subject_y", "param_y"),
                ),
            )
        )
        is not None
    ]
    mcmc_result = None
    if mcmc is not None:
        mcmc_result = dict(mcmc)
        for field in ("per_parameter", "trace_data", "rank_histograms"):
            if mcmc_result.get(field) is not None:
                mcmc_result[field] = [
                    value
                    for row in mcmc_result[field]
                    if (value := attach(row, scalar_fields)) is not None
                ]
        mcmc_result = MCMCDiagnostics.model_validate(mcmc_result).model_dump(mode="json")
    return marginal_rows, pair_rows, mcmc_result
