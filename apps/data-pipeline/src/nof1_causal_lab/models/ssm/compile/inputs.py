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
from nof1_causal_lab.models.ssm.compile.spec_translation import (
    build_structural_support_from_plan,
    get_construct_dt_days,
    get_structural_latent_layout,
    translate_spec,
)
from nof1_causal_lab.utils.structural_plan import get_manifest_indicators

if TYPE_CHECKING:
    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.compiled_ssm import CompiledParameterBinding
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.artifacts.prior import PriorValidationResult
    from nof1_causal_lab.artifacts.statistical_model_spec import ParameterSpec, StatisticalModelSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm.model import SSMSpec


def _require_explicit_causal_structure(
    ssm_spec: SSMSpec,
    *,
    structural_plan: StructuralPlan | None,
) -> None:
    """Reject implicit structural degrees of freedom on causal-design code paths."""
    if structural_plan is None:
        return

    required_block_fields = (
        "dynamics_spec",
        "diffusion_block.diffusion_chol_support",
        "diffusion_block.diffusion_chol_template",
        "lambda_block.free_support",
        "lambda_block.template",
        "manifest_means_block.free_support",
        "manifest_means_block.template",
        "manifest_chol_block.diag_support",
        "manifest_chol_block.template",
        "t0_means_block.free_support",
        "t0_means_block.template",
        "t0_chol_block.diag_support",
        "t0_chol_block.correlation_support",
        "t0_chol_block.template",
        "input_effect_block.free_support",
        "input_effect_block.template",
        "static_state_sd_block.free_support",
        "static_state_sd_block.template",
        "static_factor_loadings",
    )

    def _resolve_field(path: str):
        value = ssm_spec
        for part in path.split("."):
            value = getattr(value, part)
        return value

    missing_fields = [
        field_name for field_name in required_block_fields if _resolve_field(field_name) is None
    ]

    if not missing_fields:
        return

    raise ValueError(
        "StructuralPlan compilation requires an explicit compiled structure on SSMSpec. "
        f"Missing block fields: {', '.join(missing_fields)}. Compile from "
        "StatisticalModelSpec + StructuralPlan so "
        "translate_spec() can derive the full structural payload, or supply an already "
        "translated SSMSpec with explicit block supports and templates."
    )


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


def _order_likelihoods_by_structural_plan(
    statistical_model_spec: StatisticalModelSpec,
    structural_plan: StructuralPlan,
) -> StatisticalModelSpec:
    """Canonicalize manifest array order to the StructuralPlan contract."""
    plan_order = [str(indicator["id"]) for indicator in get_manifest_indicators(structural_plan)]
    likelihood_by_variable = {
        likelihood.indicator_id: likelihood for likelihood in statistical_model_spec.likelihoods
    }
    authored_names = [likelihood.indicator_id for likelihood in statistical_model_spec.likelihoods]
    if len(likelihood_by_variable) != len(authored_names):
        raise ValueError("StatisticalModelSpec contains duplicate likelihood variables")
    if set(authored_names) != set(plan_order):
        raise ValueError(
            "StatisticalModelSpec likelihoods do not exactly cover StructuralPlan manifests: "
            f"missing={sorted(set(plan_order) - set(authored_names))}, "
            f"unplanned={sorted(set(authored_names) - set(plan_order))}."
        )
    return statistical_model_spec.model_copy(
        update={"likelihoods": [likelihood_by_variable[variable] for variable in plan_order]}
    )


def compile_ssm_inputs_from_statistical_model_spec(
    statistical_model_spec: StatisticalModelSpec,
    *,
    structural_plan: StructuralPlan,
) -> tuple[
    SSMSpec,
    dict[str, dist.Distribution],
    list[CompiledParameterBinding],
    list[PriorValidationResult],
    dict[tuple[int, int], float],
    list[ParameterSpec],
    list[ParameterCoordinate],
]:
    """Compile executable SSM inputs from a validated semantic statistical model spec surface."""
    from nof1_causal_lab.models.ssm.compile.parameter_identity import validate_parameter_owners

    validate_parameter_owners(statistical_model_spec.parameters, structural_plan)
    ordered_statistical_model_spec = _order_likelihoods_by_structural_plan(
        statistical_model_spec, structural_plan
    )
    ssm_spec, edge_lag_days = translate_spec(
        ordered_statistical_model_spec,
        structural_plan,
    )
    _require_explicit_causal_structure(ssm_spec, structural_plan=structural_plan)

    from nof1_causal_lab.models.prior_planning import complete_parameter_priors

    index_maps = build_semantic_prior_bindings(
        ssm_spec, ordered_statistical_model_spec, structural_plan=structural_plan
    )
    parameters, bindings, auxiliary = bind_parameters(
        index_maps, ssm_spec, structural_plan, ordered_statistical_model_spec.parameters
    )
    finalized = complete_parameter_priors(
        ordered_statistical_model_spec.model_copy(update={"parameters": parameters})
    )
    prior_registry, _, diagnostics = compile_priors(
        finalized,
        ssm_spec,
        edge_lag_days=edge_lag_days,
        structural_plan=structural_plan,
    )
    parameters = finalized.parameters
    diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
    return ssm_spec, prior_registry, bindings, diagnostics, edge_lag_days, parameters, auxiliary


__all__ = [
    "SemanticBindingRegistry",
    "bind_parameters",
    "build_structural_support_from_plan",
    "build_semantic_prior_bindings",
    "compile_priors",
    "compile_ssm_inputs_from_statistical_model_spec",
    "get_construct_dt_days",
    "get_structural_latent_layout",
    "translate_spec",
]
