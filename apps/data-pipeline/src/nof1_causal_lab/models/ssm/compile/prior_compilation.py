"""Pure prior-compilation and binding stages for SSM compilation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpyro.distributions as dist
import scipy.linalg

from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.parameter import (
    ParameterCoordinate,
    PriorAuthoringTransform,
    SiteKind,
)
from nof1_causal_lab.artifacts.prior import (
    PriorPathologyCertificate,
    PriorValidationResult,
)
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import CompiledParameterBinding
from nof1_causal_lab.models.ssm.compile.common import (
    axis_names_with_fallback,
)
from nof1_causal_lab.models.ssm.compile.prior_indexing import (
    SemanticBindingRegistry,
    build_semantic_prior_bindings,
)
from nof1_causal_lab.models.ssm.compile.support import get_construct_dt_days
from nof1_causal_lab.models.ssm.execution.contracts import NUMERICAL_EPSILON
from nof1_causal_lab.models.ssm.parameterization import build_site_registry
from nof1_causal_lab.models.ssm.priors import (
    default_prior_for_descriptor,
    site_constraint,
    validate_site_prior,
)
from nof1_causal_lab.models.ssm.structure.sites import SemanticBinding, SiteDescriptor, site_size
from nof1_causal_lab.prior_distributions import (
    batch_prior_distributions,
    interval_effect_to_rate,
    persistence_to_decay,
    prior_reference_value,
)
from nof1_causal_lab.utils.model_structure import get_model_clock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec

logger = logging.getLogger("nof1_causal_lab.models.ssm.compile.inputs")
CompileDiagnostic = PriorValidationResult
PriorFailureStage = Literal[
    "compiled_parameters",
    "latent_dynamics",
    "observation_mean",
    "observation_sample",
    "support_violation",
    "model_build",
    "prior_sampling",
    "unknown",
]
_LOGM_IMAG_TOL = 1e-8
_LOGM_RELATIVE_DEVIATION_WARNING_THRESHOLD = 0.2

_DEGENERATE_PRIOR_PREAMBLE = (
    "model-spec priors must have strictly positive variance. Represent a fixed value "
    "with a fixed component coefficient or ParameterSpec.value."
)


class PriorCompilationError(AggregatedCompileError):
    """Aggregate independent prior-compilation failures into one exception."""

    header = "Prior compilation failed"


def _component_semantic_bindings(model_spec: ModelSpec) -> tuple[SemanticBinding, ...]:
    from nof1_causal_lab.models.ssm.dynamics.spec import iter_dynamics_semantic_bindings

    component_sites = {
        binding.site_name
        for binding in iter_dynamics_semantic_bindings(
            numeric.dynamics_components(model_spec),
            latent_names=tuple(numeric.state_names(model_spec)),
        )
    }
    return tuple(
        binding
        for binding in build_semantic_prior_bindings(model_spec).by_parameter.values()
        if binding.site_name in component_sites
    )


def _decay_bindings(model_spec: ModelSpec) -> tuple[SemanticBinding, ...]:
    return tuple(
        binding
        for binding in _component_semantic_bindings(model_spec)
        if binding.site_kind == SiteKind.DYNAMICS_DECAY
        and binding.transform == PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY
    )


def _linear_effect_bindings(
    model_spec: ModelSpec,
) -> tuple[tuple[SemanticBinding, int, int], ...]:
    """Linear (``beta_``) effect bindings, paired with their non-None
    ``(effect_idx, cause_idx)`` so callers receive narrowed ``int`` indices."""
    result: list[tuple[SemanticBinding, int, int]] = []
    for binding in _component_semantic_bindings(model_spec):
        effect_idx = binding.effect_idx
        cause_idx = binding.cause_idx
        if (
            binding.site_kind == SiteKind.DYNAMICS_WEIGHT
            and binding.transform == PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE
            and effect_idx is not None
            and cause_idx is not None
        ):
            result.append((binding, int(effect_idx), int(cause_idx)))
    return tuple(result)


def _binding_latent_index(binding: SemanticBinding, model_spec: ModelSpec) -> int | None:
    if binding.construct_names:
        latent_names = axis_names_with_fallback(
            numeric.state_names(model_spec),
            expected=numeric.n_states(model_spec),
            prefix="latent",
        )
        return {name: idx for idx, name in enumerate(latent_names)}.get(binding.construct_names[0])
    site = next(
        (
            candidate
            for candidate in build_site_registry(model_spec)
            if candidate.name == binding.site_name
        ),
        None,
    )
    if site is not None and site.positions:
        position = site.positions[min(binding.flat_index, len(site.positions) - 1)]
        if isinstance(position, int):
            return int(position)
    if 0 <= binding.flat_index < numeric.n_states(model_spec):
        return int(binding.flat_index)
    return None


def _resolve_model_clock_interval_days(
    model: ModelSpec,
) -> float | None:
    """Resolve the declared model clock interval without silently defaulting to 1 day."""
    try:
        interval_days = parse_duration_to_hours(get_model_clock(model)) / 24.0
    except ValueError as exc:
        raise ValueError(
            "model.measurement_clock must parse to a positive interval to "
            "compile cross-lag priors without explicit reference_interval_days."
        ) from exc

    if interval_days <= 0:
        raise ValueError(
            "model.measurement_clock must resolve to a positive interval to "
            "compile cross-lag priors."
        )
    return interval_days


def _resolve_cross_lag_interval_days(
    *,
    param_name: str,
    parameter: ParameterSpec,
    model_spec: ModelSpec,
    edge_lag_days: dict[tuple[int, int], float] | None,
    effect_idx: int,
    cause_idx: int,
) -> float:
    """Resolve a positive authoring interval for cross-lag priors."""
    ref_days = parameter.reference_interval_days
    if ref_days is not None:
        interval_days = float(ref_days)
        if interval_days <= 0:
            raise ValueError(
                f"Cross-lag prior '{param_name}' must set reference_interval_days to a "
                f"positive value, got {interval_days:.3g}."
            )
        return interval_days

    lag_days = (edge_lag_days or {}).get((effect_idx, cause_idx))
    if lag_days is not None:
        interval_days = float(lag_days)
        if interval_days <= 0:
            raise ValueError(
                f"Cross-lag prior '{param_name}' maps to non-positive edge lag {interval_days:.3g}."
            )
        return interval_days

    if not numeric.state_names(model_spec):
        raise ValueError(
            f"Cross-lag prior '{param_name}' cannot resolve effect name: "
            "ModelSpec.latent_names is empty."
        )
    effect_name = numeric.state_names(model_spec)[effect_idx]
    interval_days = _resolve_model_clock_interval_days(model_spec)
    if interval_days is not None:
        return interval_days

    raise ValueError(
        f"Cross-lag prior '{param_name}' could not resolve an authoring interval. "
        "Set reference_interval_days explicitly, or compile with edge_lag_days / "
        f"model measurement_clock metadata for effect '{effect_name}'."
    )


def _format_interval_days(days: float) -> str:
    """Render a positive day interval for diagnostics."""
    return f"{float(days):.1f}d"


def _compile_warning(
    *,
    code: str,
    parameter: str,
    issue: str,
    suggested_adjustment: str,
    compiled_site_name: str | None = None,
    compiled_flat_index: int | None = None,
    failure_stage: PriorFailureStage | None = None,
    pathology_certificate: PriorPathologyCertificate | None = None,
) -> CompileDiagnostic:
    """Build a typed non-fatal compile diagnostic."""
    return CompileDiagnostic(
        parameter=parameter,
        is_valid=True,
        code=code,
        origin="compile",
        severity="warning",
        issue=issue,
        suggested_adjustment=suggested_adjustment,
        related_parameters=[parameter],
        compiled_site_name=compiled_site_name,
        compiled_flat_index=compiled_flat_index,
        failure_stage=failure_stage,
        pathology_certificate=pathology_certificate,
    )


def collect_compile_diagnostics(
    model_spec: ModelSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    prior_registry: dict[str, dist.Distribution] | None = None,
    offdiag_interval_days: dict[tuple[int, int], float] | None = None,
) -> list[CompileDiagnostic]:
    """Collect structured compiler diagnostics for downstream consumers."""
    diagnostics: list[CompileDiagnostic] = []

    if prior_registry is not None:
        diagnostics.extend(
            collect_first_order_approximation_warnings(
                prior_registry,
                model_spec=model_spec,
                edge_lag_days=edge_lag_days,
                offdiag_interval_days=offdiag_interval_days,
            )
        )
    return diagnostics


def _log_compile_diagnostics(diagnostics: list[CompileDiagnostic]) -> None:
    for issue in diagnostics:
        logger.warning("%s: %s", issue.parameter, issue.issue)


def collect_first_order_approximation_warnings(
    prior_registry: dict[str, dist.Distribution],
    *,
    model_spec: ModelSpec,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    offdiag_interval_days: dict[tuple[int, int], float] | None = None,
) -> list[CompileDiagnostic]:
    """Return warnings when exact matrix-log DT->CT diagnostics diverge from beta/dt."""
    taylor_drift = _assemble_reference_drift_from_component_priors(prior_registry, model_spec)
    if taylor_drift is None:
        return []

    diag_abs = np.abs(np.diag(taylor_drift))
    eligible_mask = diag_abs >= NUMERICAL_EPSILON
    if not np.any(eligible_mask):
        return []
    eligible_indices = np.where(eligible_mask)[0]
    min_diag = float(np.min(diag_abs[eligible_indices]))
    if min_diag < NUMERICAL_EPSILON:
        return []
    min_diag_latent_idx = int(eligible_indices[int(np.argmin(diag_abs[eligible_indices]))])
    min_diag_name = next(
        (
            binding.parameter_name
            for binding in _decay_bindings(model_spec)
            if _binding_latent_index(binding, model_spec) == min_diag_latent_idx
        ),
        None,
    )
    min_diag_label = f"{min_diag_name}" if min_diag_name else f"latent[{min_diag_latent_idx}]"

    warnings: list[CompileDiagnostic] = []
    latent_names = axis_names_with_fallback(
        numeric.state_names(model_spec),
        expected=numeric.n_states(model_spec),
        prefix="latent",
    )
    for binding, effect_idx, cause_idx in _linear_effect_bindings(model_spec):
        prior = prior_registry.get(binding.site_name)
        if prior is None:
            continue
        offdiag_mu = np.asarray(prior_reference_value(prior)).reshape(-1)
        if offdiag_mu.size == 0:
            continue
        offdiag_value = _value_at(offdiag_mu, binding.flat_index, default=0.0)
        interval_days = _resolve_offdiag_interval_days(
            effect_idx=effect_idx,
            cause_idx=cause_idx,
            edge_lag_days=edge_lag_days,
            offdiag_interval_days=offdiag_interval_days,
        )
        if interval_days is None:
            continue

        cause_name = latent_names[cause_idx]
        effect_name = latent_names[effect_idx]
        offdiag_label = f"{binding.parameter_name} ({cause_name} -> {effect_name})"

        try:
            exact_drift = matrix_log_diagnostic_drift(
                taylor_drift,
                interval_days=interval_days,
            )
        except ValueError as exc:
            warnings.append(
                _compile_warning(
                    code="dt_ct_approximation_warning",
                    parameter=binding.prior_field or binding.site_name,
                    issue=f"{offdiag_label}: exact matrix-log CT diagnostic failed: {exc}",
                    suggested_adjustment=(
                        "Shrink the DT beta prior or elicit the prior directly on a real, stable "
                        "CT drift scale."
                    ),
                    compiled_site_name=binding.site_name,
                    compiled_flat_index=binding.flat_index,
                    failure_stage="compiled_parameters",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dt_ct_approximation",
                        primary_score=1.0,
                    ),
                )
            )
            continue

        exact_value = float(exact_drift[effect_idx, cause_idx])
        deviation = abs(exact_value - float(offdiag_value)) / max(
            abs(exact_value), NUMERICAL_EPSILON
        )
        ratio = abs(exact_value) / min_diag
        if deviation <= _LOGM_RELATIVE_DEVIATION_WARNING_THRESHOLD and ratio <= 0.2:
            continue
        warnings.append(
            _compile_warning(
                code="dt_ct_approximation_warning",
                parameter=binding.prior_field or binding.site_name,
                issue=(
                    f"{offdiag_label}: matrix-log mismatch; exact CT coupling at "
                    f"{_format_interval_days(interval_days)} is {abs(exact_value):.3f} 1/day "
                    f"versus the elementwise beta/dt value {abs(float(offdiag_value)):.3f} "
                    f"1/day; logm deviation is {deviation * 100:.0f}% and the exact coupling "
                    f"is {ratio * 100:.0f}% of the smallest realised CT diagonal damping "
                    f"({min_diag:.3f} 1/day, {min_diag_label})."
                ),
                suggested_adjustment=(
                    "Use the exact matrix-log CT scale when revising this edge: shorten the "
                    "reference interval, shrink the DT beta prior, or elicit the prior directly "
                    "on the CT rate."
                ),
                compiled_site_name=binding.site_name,
                compiled_flat_index=binding.flat_index,
                failure_stage="compiled_parameters",
                pathology_certificate=PriorPathologyCertificate(
                    kind="dt_ct_approximation",
                    primary_score=deviation,
                    secondary_score=ratio,
                ),
            )
        )
    return warnings


def _value_at(values: np.ndarray, flat_index: int, *, default: float) -> float:
    if values.size == 0:
        return float(default)
    if values.size == 1:
        return float(values[0])
    if flat_index < values.size:
        return float(values[flat_index])
    return float(default)


def _assemble_reference_drift_from_component_priors(
    prior_registry: dict[str, dist.Distribution],
    model_spec: ModelSpec,
) -> np.ndarray | None:
    drift = np.zeros((numeric.n_states(model_spec), numeric.n_states(model_spec)), dtype=float)
    populated = False

    for binding in _decay_bindings(model_spec):
        prior = prior_registry.get(binding.site_name)
        latent_idx = _binding_latent_index(binding, model_spec)
        if prior is None or latent_idx is None:
            continue
        decay_mu = np.asarray(prior_reference_value(prior)).reshape(-1)
        if decay_mu.size == 0:
            continue
        drift[latent_idx, latent_idx] = -_value_at(
            decay_mu,
            binding.flat_index,
            default=0.0,
        )
        populated = True

    for binding, effect_idx, cause_idx in _linear_effect_bindings(model_spec):
        prior = prior_registry.get(binding.site_name)
        if prior is None:
            continue
        weight_mu = np.asarray(prior_reference_value(prior)).reshape(-1)
        if weight_mu.size == 0:
            continue
        drift[effect_idx, cause_idx] = _value_at(
            weight_mu,
            binding.flat_index,
            default=0.0,
        )
        populated = True

    return drift if populated else None


def _resolve_offdiag_interval_days(
    *,
    effect_idx: int,
    cause_idx: int,
    edge_lag_days: dict[tuple[int, int], float] | None,
    offdiag_interval_days: dict[tuple[int, int], float] | None,
) -> float | None:
    interval = (offdiag_interval_days or {}).get((effect_idx, cause_idx))
    if interval is None:
        interval = (edge_lag_days or {}).get((effect_idx, cause_idx))
    if interval is None:
        return None
    interval = float(interval)
    if interval <= 0:
        return None
    return interval


def _transition_from_elementwise_dt_terms(
    drift: np.ndarray,
    interval_days: float,
) -> np.ndarray:
    transition = np.eye(drift.shape[0], dtype=float)
    for idx in range(drift.shape[0]):
        transition[idx, idx] = math.exp(float(drift[idx, idx]) * interval_days)

    offdiag_support = ~np.eye(drift.shape[0], dtype=bool)
    transition[offdiag_support] = drift[offdiag_support] * interval_days
    return transition


def matrix_log_diagnostic_drift(
    drift: np.ndarray,
    *,
    interval_days: float,
) -> np.ndarray:
    """Compute the full matrix-log CT drift used by dynamics diagnostics."""
    if interval_days <= 0:
        raise ValueError("matrix-log CT dynamics diagnostics require a positive interval.")

    transition = _transition_from_elementwise_dt_terms(drift, interval_days)
    log_transition = scipy.linalg.logm(transition)
    imaginary_scale = float(np.max(np.abs(np.imag(log_transition))))
    if imaginary_scale > _LOGM_IMAG_TOL:
        raise ValueError(
            "Matrix-log CT dynamics diagnostics require an embeddable real transition matrix; "
            f"max imaginary logm component is {imaginary_scale:.3g}."
        )
    return np.real(log_transition) / interval_days


def _correlation_prior(prior: dist.Distribution) -> dist.Distribution:
    """Apply the declared correlation domain to Normal and bounded priors."""
    if isinstance(prior, dist.Normal):
        return dist.TruncatedNormal(prior.loc, prior.scale, low=-1.0, high=1.0)
    if isinstance(prior, dist.TwoSidedTruncatedDistribution):
        low = np.maximum(np.asarray(prior.low), -1.0)
        high = np.minimum(np.asarray(prior.high), 1.0)
        if np.any(low >= high):
            raise ValueError("Initial-state correlation prior has no support within [-1, 1]")
        return dist.TruncatedNormal(prior.base_dist.loc, prior.base_dist.scale, low=low, high=high)
    if isinstance(prior, dist.Uniform):
        low = np.maximum(np.asarray(prior.low), -1.0)
        high = np.minimum(np.asarray(prior.high), 1.0)
        if np.any(low >= high):
            raise ValueError("Initial-state correlation prior has no support within [-1, 1]")
        return dist.Uniform(low, high)
    return prior


def compile_priors(
    model: ModelSpec,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
) -> tuple[dict[str, dist.Distribution], SemanticBindingRegistry, list[CompileDiagnostic]]:
    """Bind the model's native distributions to their declared execution coordinates."""
    model.require_execution_structure()
    edge_lag_days = numeric.edge_lag_days(model) if edge_lag_days is None else edge_lag_days
    parameters = {
        parameter.id: parameter for parameter in model.parameters if parameter.value is None
    }
    missing = [parameter.id for parameter in parameters.values() if parameter.distribution is None]
    if missing:
        raise ValueError(f"ModelSpec parameters require explicit prior distributions: {missing}")

    active_sites = build_site_registry(model)
    prior_entries: dict[str, dist.Distribution] = {
        site.name: default_prior_for_descriptor(site) for site in active_sites
    }
    site_by_name = {site.name: site for site in active_sites}
    per_site: dict[str, dict[int, dist.Distribution]] = {}

    def attach(site: SiteDescriptor, index: int, prior: dist.Distribution) -> None:
        if index < 0 or index >= site_size(site.shape):
            raise ValueError(
                f"Prior index {index} is outside site {site.name!r} shape {site.shape}"
            )
        if prior.batch_shape or prior.event_shape:
            raise ValueError("Each authored prior must describe one scalar parameter")
        validate_site_prior(site, prior)
        values = per_site.setdefault(site.name, {})
        if index in values:
            raise ValueError(f"Multiple authored priors bind to {site.name!r} coordinate {index}")
        values[index] = prior

    bindings = build_semantic_prior_bindings(model)
    binding_by_parameter = bindings.by_parameter
    errors: list[str] = []
    offdiag_interval_days: dict[tuple[int, int], float] = {}

    for param_name, parameter in parameters.items():
        try:
            prior = model.distribution_for(parameter.id)
            assert prior is not None
            if prior.batch_shape or prior.event_shape:
                raise ValueError(
                    "This compiler requires independent scalar input laws; shared laws remain intact in ModelSpec. Refit from the original input revision."
                )
            prior.validate_args()
            if isinstance(prior, dist.Delta):
                raise ValueError(f"Prior {param_name!r}: {_DEGENERATE_PRIOR_PREAMBLE}")
            binding = binding_by_parameter.get(param_name)
            if binding is None:
                errors.append(
                    f"Prior {param_name!r} for {model.parameter_context(parameter.id).quantity.value!r} could not be structurally bound to the compiled SSM."
                )
                continue

            if binding.transform == PriorAuthoringTransform.SITE_WIDE:
                site = site_by_name.get(binding.site_name)
                if site is None:
                    raise ValueError(
                        f"Prior {param_name!r} maps to inactive site {binding.site_name!r}."
                    )
                for index in range(site_size(site.shape)):
                    attach(site, index, prior)
                continue

            if binding.transform == PriorAuthoringTransform.SITE_ROW:
                site = site_by_name.get(binding.site_name)
                if site is None:
                    raise ValueError(
                        f"Prior {param_name!r} maps to inactive site {binding.site_name!r}."
                    )
                if len(site.shape) != 2:
                    raise ValueError(
                        f"Prior {param_name!r} requires a matrix-valued site, "
                        f"but {binding.site_name!r} has shape {site.shape}."
                    )
                row_idx = binding.flat_index
                n_rows, n_cols = site.shape
                if row_idx >= n_rows:
                    raise ValueError(
                        f"Prior {param_name!r} maps to row {row_idx} of "
                        f"{binding.site_name!r}, which has {n_rows} rows."
                    )
                for col_idx in range(n_cols):
                    attach(site, row_idx * n_cols + col_idx, prior)
                continue

            if binding.transform == PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY:
                construct_name = binding.construct_names[0]
                ref_days = parameter.reference_interval_days
                resolved_ref_days = float(ref_days) if ref_days is not None else None
                if resolved_ref_days is not None and resolved_ref_days <= 0:
                    errors.append(
                        f"AR prior '{param_name}' reference_interval_days must be positive, "
                        f"got {resolved_ref_days:.3g}"
                    )
                    continue
                dt = (
                    resolved_ref_days
                    if resolved_ref_days is not None
                    else get_construct_dt_days(model, construct_name)
                )
                attach(
                    site_by_name[binding.site_name],
                    binding.flat_index,
                    persistence_to_decay(prior, dt),
                )
                continue

            if binding.transform == PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE:
                if model is None:
                    raise ValueError(
                        "Dynamics effect prior compilation requires a translated ModelSpec runtime."
                    )
                if binding.effect_idx is not None and binding.cause_idx is not None:
                    dt = _resolve_cross_lag_interval_days(
                        param_name=param_name,
                        parameter=parameter,
                        model_spec=model,
                        edge_lag_days=edge_lag_days,
                        effect_idx=binding.effect_idx,
                        cause_idx=binding.cause_idx,
                    )
                    offdiag_interval_days[(binding.effect_idx, binding.cause_idx)] = dt
                else:
                    raise ValueError(
                        f"Dynamics effect prior {param_name!r} is missing effect/cause metadata."
                    )
                attach(
                    site_by_name[binding.site_name],
                    binding.flat_index,
                    interval_effect_to_rate(prior, dt),
                )
                continue

            if binding.transform == PriorAuthoringTransform.INITIAL_STATE_CORRELATION:
                prior = _correlation_prior(prior)
            attach(site_by_name[binding.site_name], binding.flat_index, prior)
        except ValueError as exc:
            errors.append(str(exc))
            continue

    if errors:
        raise PriorCompilationError(errors)

    for site_name, entries in per_site.items():
        site = site_by_name.get(site_name)
        if site is None:
            raise ValueError(f"Prior site {site_name!r} maps to no active sample site.")
        coordinates = [prior_entries[site.name]] * site_size(site.shape)
        for index, prior in entries.items():
            coordinates[index] = prior
        prior_entries[site.name] = batch_prior_distributions(
            coordinates, site.shape, support=site_constraint(site)
        )

    prior_registry = prior_entries

    diagnostics: list[CompileDiagnostic] = []
    if model is not None:
        diagnostics = collect_compile_diagnostics(
            model,
            edge_lag_days=edge_lag_days,
            prior_registry=prior_registry,
            offdiag_interval_days=offdiag_interval_days,
        )
        _log_compile_diagnostics(diagnostics)

    return prior_registry, bindings, diagnostics


def bind_parameters(
    bindings: SemanticBindingRegistry,
    model_spec: ModelSpec,
    parameters: Sequence[ParameterSpec],
) -> tuple[list[CompiledParameterBinding], list[ParameterCoordinate]]:
    """Compile scientific definitions into explicit scalar execution bindings."""
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.models.ssm.compile.parameter_identity import (
        component_identity,
    )

    sites = {site.name: site for site in build_site_registry(model_spec)}
    definitions = {parameter.id: parameter for parameter in parameters}
    all_bindings = dict(bindings.by_parameter)

    result = []
    bound_coordinates = set()
    auxiliary = []
    for parameter_id, binding in sorted(all_bindings.items()):
        definition = definitions[parameter_id]
        site = sites[binding.site_name]
        shape = site.shape
        if binding.transform == PriorAuthoringTransform.SITE_WIDE:
            indices = list(np.ndindex(shape))
        elif binding.transform == PriorAuthoringTransform.SITE_ROW:
            indices = [(binding.flat_index, column) for column in range(shape[1])]
        else:
            indices = [tuple(int(i) for i in np.unravel_index(binding.flat_index, shape))]
        elements, coordinates = {}, {}
        for index in indices:
            coordinate = ParameterCoordinate(site_name=site.name, indices=index)
            if coordinate in bound_coordinates:
                raise ValueError(
                    f"Runtime coordinate {coordinate.label} has multiple scientific owners"
                )
            bound_coordinates.add(coordinate)
            component = component_identity(definition, index, binding, site, model_spec)
            if component is None:
                auxiliary.append(coordinate)
                continue
            element_id, label = component
            if element_id in elements:
                raise ValueError(f"Parameter {definition.name!r} has duplicate logical components")
            elements[element_id] = label
            coordinates[element_id] = coordinate
        if not coordinates:
            continue
        result.append(
            CompiledParameterBinding(
                parameter_id=definition.id,
                coordinates=coordinates,
                elements=elements,
                site_name=site.name,
                prior_field=binding.prior_field,
                flat_index=binding.flat_index,
                site_kind=binding.site_kind,
                transform=binding.transform,
                construct_names=list(binding.construct_names),
                indicator_names=list(binding.indicator_names),
                component_index=binding.component_index,
                effect_idx=binding.effect_idx,
                cause_idx=binding.cause_idx,
            )
        )
    # Rectangular likelihood tensors contain padded/unused indicator rows. Their
    # explicit execution-only status keeps them out of reported scientific findings.
    for site in sites.values():
        for index in np.ndindex(site.shape):
            coordinate = ParameterCoordinate(site_name=site.name, indices=index)
            if coordinate in bound_coordinates:
                continue
            if site.site_kind not in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}:
                raise ValueError(
                    f"Runtime coordinate {coordinate.label} has no scientific parameter definition"
                )
            auxiliary.append(coordinate)
    return result, auxiliary
