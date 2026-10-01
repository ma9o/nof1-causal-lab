"""Attach scientific references to numerical findings before artifact persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.artifacts.identity import ParameterRef
from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    PosteriorMarginal,
    PosteriorPair,
)

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs

# Numerical-engine rows carry runtime coordinates until this boundary validates
# their scientific result schemas and replaces coordinates with scientific refs.
type RuntimeFindingPayload = dict[str, Any]


def reference_posterior_findings(
    inputs: CompiledFitInputs,
    marginals: list[RuntimeFindingPayload],
    pairs: list[RuntimeFindingPayload],
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Translate the compiler's exact mapping once; unknown coordinates are errors."""
    compiled_bindings = inputs.bindings
    auxiliary_coordinates = inputs.auxiliary_coordinates
    bindings = {binding.parameter_id: binding for binding in compiled_bindings}
    subjects = {
        coordinate: ParameterRef(parameter_id=binding.parameter_id, element_id=element_id)
        for binding in compiled_bindings
        for element_id, coordinate in binding.coordinates.items()
    }
    auxiliary = set(auxiliary_coordinates)

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
            result[label_field] = bindings[subject.parameter_id].elements[subject.element_id]
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
    return marginal_rows, pair_rows
