"""The compiler's complete coordinate-to-scientific-reference map."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identity import ParameterRef

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
    from nof1_causal_lab.models.ssm.inference.diagnostics_viz import ParameterReferences


def parameter_references(inputs: CompiledFitInputs) -> ParameterReferences:
    """Auxiliary coordinates are explicit; every unexpected coordinate remains an error."""
    references: dict[ParameterCoordinate, tuple[str, ParameterRef] | None] = {
        coordinate: (
            binding.elements[element],
            ParameterRef(parameter_id=binding.parameter_id, element_id=element),
        )
        for binding in inputs.compiled.bindings
        for element, coordinate in binding.coordinates.items()
    }
    for coordinate in inputs.compiled.auxiliary_coordinates:
        references[coordinate] = None
    return references
