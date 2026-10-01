"""Public pure-compilation entry points for executable SSM inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
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
from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    build_prior_runtime_bundle,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.artifacts.prior import PriorValidationResult
    from nof1_causal_lab.models.ssm.compile.bindings import CompiledParameterBinding


@dataclass(frozen=True)
class CompiledFitInputs:
    """Compiler-owned fit capability, independent of later panel compatibility.

    Collections are read-only interfaces, not deep immutability. Native NumPyro
    distributions and JAX payloads are shared as values and must not be mutated.
    """

    spec: ModelSpec
    prior_runtime_bundle: PriorRuntimeBundle
    parameter_layout: SSMParameterLayout
    bindings: tuple[CompiledParameterBinding, ...]
    diagnostics: tuple[PriorValidationResult, ...]
    auxiliary_coordinates: tuple[ParameterCoordinate, ...]


@dataclass(frozen=True)
class IncompleteModel:
    """The authored model still needs choices before fitting."""

    message: str


@dataclass(frozen=True)
class UnsupportedFit:
    """The current fitting engine cannot compile these scientific choices."""

    errors: tuple[str, ...]

    @property
    def message(self) -> str:
        return "\n".join(self.errors)


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
) -> CompiledFitInputs | IncompleteModel | UnsupportedFit:
    """Resolve fitting once; incomplete/unsupported choices remain editable.

    Only the compiler's expected diagnostic exceptions become variants. Broken
    internal assumptions (including other ValueErrors) still propagate.
    """
    from nof1_causal_lab.models.ssm import numerics as numeric

    try:
        model.require_priors()
        numeric.validate_execution(model)
        prior_registry, index_maps, diagnostics = compile_priors(model)
        bindings, auxiliary = bind_parameters(index_maps, model, model.execution_parameters)
        diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
        prior_runtime_bundle = build_prior_runtime_bundle(model, prior_registry)
        parameter_layout = SSMParameterLayout.from_spec(model)
    except IncompleteModelError as exc:
        return IncompleteModel(str(exc))
    except AggregatedCompileError as exc:
        return UnsupportedFit(tuple(exc.errors))
    return CompiledFitInputs(
        spec=model,
        prior_runtime_bundle=prior_runtime_bundle,
        parameter_layout=parameter_layout,
        bindings=tuple(bindings),
        diagnostics=tuple(diagnostics),
        auxiliary_coordinates=tuple(auxiliary),
    )


__all__ = [
    "CompiledFitInputs",
    "IncompleteModel",
    "UnsupportedFit",
    "SemanticBindingRegistry",
    "bind_parameters",
    "build_structural_support_from_model",
    "build_semantic_prior_bindings",
    "compile_priors",
    "compile_ssm_inputs_from_model",
    "get_construct_dt_days",
]
