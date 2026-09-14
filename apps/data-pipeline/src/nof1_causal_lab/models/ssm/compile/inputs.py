"""Public pure-compilation entry points for executable SSM inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm.compile.prior_compilation import (
    bind_parameters,
    compile_priors,
)
from nof1_causal_lab.models.ssm.compile.prior_indexing import (
    SemanticBindingRegistry,
    build_semantic_prior_bindings,
)
from nof1_causal_lab.models.ssm.compile.support import (
    build_structural_support_from_model,
    get_construct_dt_days,
)

if TYPE_CHECKING:
    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.artifacts.prior import PriorValidationResult
    from nof1_causal_lab.models.ssm.compile.bindings import CompiledParameterBinding


def _attach_compile_binding_provenance(
    diagnostics: list[PriorValidationResult],
    bindings: list[CompiledParameterBinding],
) -> list[PriorValidationResult]:
    """Attach direct-writer parameter provenance to compile diagnostics when possible."""
    binding_index: dict[tuple[str, int], list[str]] = {}
    for binding in bindings:
        binding_index.setdefault((binding.site_name, binding.flat_index), []).append(
            binding.parameter_id
        )

    for diagnostic in diagnostics:
        if diagnostic.compiled_site_name is None or diagnostic.compiled_flat_index is None:
            continue
        related_parameters = binding_index.get(
            (diagnostic.compiled_site_name, diagnostic.compiled_flat_index)
        )
        if related_parameters:
            diagnostic.related_parameters = related_parameters

    return diagnostics


def compile_ssm_inputs_from_model(
    model: ModelSpec,
) -> tuple[
    dict[str, dist.Distribution],
    list[CompiledParameterBinding],
    list[PriorValidationResult],
    dict[tuple[int, int], float],
    list[ParameterCoordinate],
]:
    """Compile executable SSM inputs from a validated semantic statistical model spec surface."""

    model.require_priors()
    model.require_execution_structure()
    from nof1_causal_lab.models.ssm import numerics as numeric

    numeric.validate_execution(model)
    edge_lag_days = numeric.edge_lag_days(model)
    index_maps = build_semantic_prior_bindings(model)
    bindings, auxiliary = bind_parameters(index_maps, model, model.parameters)
    prior_registry, _, diagnostics = compile_priors(model, edge_lag_days=edge_lag_days)
    diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
    return prior_registry, bindings, diagnostics, edge_lag_days, auxiliary


__all__ = [
    "SemanticBindingRegistry",
    "bind_parameters",
    "build_structural_support_from_model",
    "build_semantic_prior_bindings",
    "compile_priors",
    "compile_ssm_inputs_from_model",
    "get_construct_dt_days",
]
