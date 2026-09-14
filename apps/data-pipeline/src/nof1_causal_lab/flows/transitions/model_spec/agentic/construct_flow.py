"""Gradual construct-by-construct model-spec flow.

Replaces the parameter-block decomposition with construct admission along the
causal DAG's topological order. Each construct is proposed by the LLM through the
``submit_construct`` tool (its emission choice + priors keyed by canonical
parameter name); the cumulative partial model is compiled and gated by the
**exact** prior-predictive reachability battery
(:mod:`nof1_causal_lab.models.ssm.construct_admission`). A construct that fails a
hard check reopens for revision; a soft failure is a decision (revise, or accept
the consequence via ``accept``). When every construct is admitted, the
accumulated :class:`~nof1_causal_lab.artifacts.model_spec.ModelSpec`
are materialized by the Temporal model-spec workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.construct import CausalEdge, Construct, endpoint_validation_scope
from nof1_causal_lab.artifacts.expressions import hill_applications
from nof1_causal_lab.artifacts.parameter_spec import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.flows.runtime_events import emit_model_spec_admission_event
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.model_structure import model_for_constructs
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionState,
    AdmissionTiming,
    ConstructAdmissionReport,
    ConstructContribution,
    DesignInfo,
    admit_construct,
    recheck_member,
    trial_admission_state,
)
from nof1_causal_lab.models.ssm.reachability import CHECK_MODES, CheckResult, stage_outcome
from nof1_causal_lab.utils.model_structure import (
    get_edges,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import polars as pl

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    from .parameter_candidates import ParameterMetadata

# Attempts per construct before the build fails (each attempt is one fresh
# agent session that must call submit_construct with a revised proposal).
_MAX_ATTEMPTS_PER_CONSTRUCT = 4

# --------------------------------------------------------------------------- #
# Prompt projection of a concrete component proposal
# --------------------------------------------------------------------------- #

# Alternative component choices for the authoring prompt: a
# self-limiting quartic well and Hill (saturating) edge terms. Both are positive
# dynamics parameters; they are admitted per construct on demand.
_STRUCTURAL_ROLE = (ParameterRole.DYNAMICS_PARAMETER_POSITIVE, ParameterConstraint.POSITIVE)


@dataclass(frozen=True)
class ParamCatalog:
    """Prompt metadata derived from the coefficients of a concrete ModelSpec proposal.

    Authoring defaults choose identification anchors and component coefficients.
    Compilation validates those choices; this projection supplies labels and
    prior guidance without adding a second scientific definition.
    """

    roles: Mapping[str, tuple[ParameterRole, ParameterConstraint]]
    by_construct: Mapping[str, tuple[str, ...]]
    global_params: frozenset[str]
    parameter_ids: Mapping[str, str]
    metadata: Mapping[str, ParameterMetadata]

    model: ModelSpec

    @classmethod
    def from_model(cls, source_model: ModelSpec) -> ParamCatalog:
        from nof1_causal_lab.models.model_mechanisms import default_model

        from .parameter_candidates import describe_parameters

        model = default_model(source_model)
        base = describe_parameters(model)
        variants = default_model(
            source_model,
            self_limiting=model.state_order,
            hill_edges={
                edge.id for edge in model.execution_edges if edge.cause.id in model.state_order
            },
        )
        metadata = {row["name"]: row for row in (*base, *describe_parameters(variants))}
        roles = {
            name: (ParameterRole(row["role"]), ParameterConstraint(row["constraint"]))
            for name, row in metadata.items()
        }
        by_construct = {}
        global_params = set()
        for row in base:
            owners = row["construct_names"]
            if len(owners) != 1:
                global_params.add(row["name"])
            for owner in owners:
                by_construct.setdefault(owner, []).append(row["name"])
        return cls(
            roles=roles,
            by_construct={key: tuple(value) for key, value in by_construct.items()},
            global_params=frozenset(global_params),
            parameter_ids={row["name"]: row["id"] for row in base},
            metadata=metadata,
            model=model,
        )

    def structural_names(self, construct: str, parents: Sequence[str]) -> set[str]:
        names = {f"self_limit_{construct}"}
        for parent in parents:
            names.update(
                {
                    f"hill_emax_{parent}_{construct}",
                    f"hill_ec50_{parent}_{construct}",
                    f"hill_n_{parent}_{construct}",
                }
            )
        return names

    def prior_names_for(
        self,
        construct: str,
        *,
        admitted_prior_names: Collection[str] = (),
    ) -> set[str]:
        names = set(self.by_construct.get(construct, ()))
        names -= set(admitted_prior_names)
        return names

    def role_for(self, name: str) -> tuple[ParameterRole, ParameterConstraint]:
        return self.roles[name]

    def metadata_for(self, name: str) -> ParameterMetadata:
        return self.metadata[name]


@dataclass(frozen=True)
class AdmissionTurnInventory:
    """Prompt context for one cumulative admission turn."""

    catalog: ParamCatalog
    compiler_prior_names: frozenset[str]
    structural_prior_names: frozenset[str]
    closing_beta_names: frozenset[str]
    incoming_saturating_parents: tuple[str, ...]

    def prior_names(self, admitted_prior_names: Collection[str]) -> set[str]:
        return set(self.compiler_prior_names) - set(admitted_prior_names)


def derive_admission_turn_inventory(
    *,
    construct: str,
    admitted: Collection[str],
    current_catalog: ParamCatalog,
    previous_catalog: ParamCatalog | None,
) -> AdmissionTurnInventory:
    """Derive the turn's parameter surface from consecutive restricted compiles.

    The current restricted plan contains the admitted prefix plus ``construct``.
    Parameters owned by ``construct`` cover its local and incoming-edge sites;
    the compiler delta from the previous prefix adds sites that materialize
    elsewhere when this construct closes a feedback loop or dependency.
    """
    admitted_names = set(admitted)
    previous_names = set(previous_catalog.roles) if previous_catalog is not None else set()
    newly_materialized = set(current_catalog.parameter_ids) - previous_names
    compiler_prior_names = current_catalog.prior_names_for(construct) | newly_materialized

    fixed_effects = {
        name: current_catalog.metadata_for(name)
        for name in compiler_prior_names
        if current_catalog.role_for(name)[0] == ParameterRole.FIXED_EFFECT
    }
    incoming_saturating_parents = tuple(
        sorted(
            str(metadata["cause"])
            for metadata in fixed_effects.values()
            if metadata.get("effect") == construct and metadata.get("cause") in admitted_names
        )
    )
    closing_beta_names = frozenset(
        name
        for name, metadata in fixed_effects.items()
        if metadata.get("cause") == construct and metadata.get("effect") in admitted_names
    )

    structural_names = {f"self_limit_{construct}"}
    saturating_edges = {(parent, construct) for parent in incoming_saturating_parents} | {
        (construct, str(fixed_effects[name]["effect"])) for name in closing_beta_names
    }
    for cause, effect in saturating_edges:
        structural_names.update(
            {
                f"hill_emax_{cause}_{effect}",
                f"hill_ec50_{cause}_{effect}",
                f"hill_n_{cause}_{effect}",
            }
        )

    return AdmissionTurnInventory(
        catalog=current_catalog,
        compiler_prior_names=frozenset(compiler_prior_names),
        structural_prior_names=frozenset(structural_names),
        closing_beta_names=closing_beta_names,
        incoming_saturating_parents=incoming_saturating_parents,
    )


def construct_parents(model: ModelSpec, construct: str) -> list[str]:
    """Direct causal parents of ``construct`` (edge sources into it)."""
    parents: list[str] = []
    for edge in get_edges(model):
        cause = edge["cause"]
        effect = edge["effect"]
        if effect == construct and cause is not None and str(cause) not in parents:
            parents.append(str(cause))
    return parents


def _acceptance_map(
    decisions: Sequence[Mapping[str, Any]] | None,
) -> dict[tuple[str, str], str]:
    """Validate structured, target-scoped soft-check acceptance decisions."""
    accepted: dict[tuple[str, str], str] = {}
    for decision in decisions or ():
        check = str(decision.get("check", "")).strip()
        target = str(decision.get("target", "")).strip()
        rationale = str(decision.get("rationale", "")).strip()
        if not check or not target or not rationale:
            raise ValueError("Every acceptance requires non-empty check, target, and rationale.")
        key = (check, target)
        if key in accepted:
            raise ValueError(f"Duplicate acceptance for {check} [{target}].")
        accepted[key] = rationale
    return accepted


def _closing_edge_effects(
    model: ModelSpec, construct: str, prior_admitted: Collection[str]
) -> list[str]:
    """Already-admitted effect(s) of feedback edges out of ``construct``.

    These are the cycle members whose latent dynamics change when admitting ``construct``
    closes the loop — so they warrant a coupled recheck against the closed-loop model.
    """
    effects: list[str] = []
    for edge in get_edges(model):
        cause = edge["cause"]
        effect = edge["effect"]
        if cause == construct and effect in prior_admitted and str(effect) not in effects:
            effects.append(str(effect))
    return effects


def _closed_loop_target(
    member: ConstructContribution,
    edges: Sequence[CausalEdge],
) -> ConstructContribution:
    """The incoming mechanisms included in the closed-loop model's checks."""
    incoming = [edge for edge in edges if edge.effect.id == member.construct.id and edge.mechanisms]
    return replace(
        member,
        edge_parents=tuple(edge.cause.name for edge in incoming),
        hill_parents=tuple(
            edge.cause.name
            for edge in incoming
            if any(tuple(hill_applications(term.expression)) for term in edge.mechanisms)
        ),
    )


# --------------------------------------------------------------------------- #
# Tool payload → ConstructContribution
# --------------------------------------------------------------------------- #


def contribution_from_payload(
    model: ModelSpec, payload: Mapping[str, Any]
) -> ConstructContribution:
    """Parse a ``submit_construct`` payload into a canonical construct contribution.

    Dynamics are declared in the same mechanism types persisted by the model.
    The referenced free coefficients determine the dynamics parameter catalog.
    """
    construct = Construct.model_validate(payload["construct"])
    with endpoint_validation_scope(payload["edges"], endpoints=(*model.constructs, construct)):
        edges = TypeAdapter(tuple[CausalEdge, ...]).validate_python(payload["edges"])
    proposed = TypeAdapter(tuple[ParameterSpec, ...]).validate_python(payload["parameters"])
    parameters = proposed
    contribution = ConstructContribution(
        construct=construct,
        edges=edges,
        parameters=parameters,
    )
    return _closed_loop_target(contribution, edges)


# --------------------------------------------------------------------------- #
# Design derivation from real longitudinal data (fit-consistent)
# --------------------------------------------------------------------------- #


def build_design_info(
    state: AdmissionState,
    contribution: ConstructContribution,
    data_for_model: pl.DataFrame,
    *,
    n_draws: int,
    seed: int,
) -> DesignInfo:
    """Reachability design for admitting ``contribution`` onto ``state`` (the trial model)."""
    return _design_for_state(
        trial_admission_state(state, contribution),
        data_for_model,
        n_draws=n_draws,
        seed=seed,
    )


def _design_for_state(
    model_state: AdmissionState,
    data_for_model: pl.DataFrame,
    *,
    n_draws: int,
    seed: int,
) -> DesignInfo:
    """Derive the reachability design against the compiled ``model_state``.

    Uses the canonical ``prepare_model_runtime`` so the sampling grid, the
    per-indicator observation indices, and the observed values all live in the
    same time + observation space the fit uses — including support-aware handling
    and the emission-space scaling the raw data does not carry. Both admission
    (against a trial state) and the coupled recheck (against the closed-loop state)
    build their design here.
    """
    import polars as pl

    from nof1_causal_lab.models.model_checks import check_execution
    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime

    model_spec = model_state.completed_model(restrict=True)
    check_execution(model_spec)

    indicator_ids = list(model_spec.manifest_indicator_order)
    indicator_ids.extend(item.source_indicator_id for item in model_spec.known_inputs.values())
    trial_data = data_for_model.filter(pl.col("indicator_id").is_in(indicator_ids))
    runtime = prepare_model_runtime(trial_data, model_spec=model_spec)

    times = np.asarray(runtime.times, dtype=float)
    observations = np.asarray(runtime.observations, dtype=float)
    assert numeric.observation_ids(runtime.spec) is not None
    manifest_ids = list(numeric.observation_ids(runtime.spec))

    obs_index_by_indicator: dict[str, np.ndarray] = {}
    values_by_indicator: dict[str, np.ndarray] = {}
    for i, manifest in enumerate(manifest_ids):
        present = np.where(np.isfinite(observations[:, i]))[0]
        obs_index_by_indicator[manifest] = present
        values_by_indicator[manifest] = observations[present, i]

    return DesignInfo(
        t_grid=jnp.asarray(times),
        manifest_ids=tuple(manifest_ids),
        obs_index_by_indicator=obs_index_by_indicator,
        values_by_indicator=values_by_indicator,
        n_draws=n_draws,
        seed=seed,
        observation_support=runtime.observation_support,
        transition_inputs=runtime.transition_inputs,
    )


# --------------------------------------------------------------------------- #
# Feedback rendering
# --------------------------------------------------------------------------- #


def render_admission_feedback(report: ConstructAdmissionReport) -> str:
    """Render a construct's battery results + verdict as LLM-facing feedback."""
    lines = [f"## Reachability report for `{report.name}`", "", report.outcome, ""]
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"- [{mark}] {r.check} [{r.target}]: {r.value} (target {r.band})")
        if not r.passed:
            if r.note:
                lines.append(f"    {r.note}")
            for d in r.diagnosis:
                lines.append(f"    · {d}")
    if report.annotations:
        lines.append("")
        lines.append("Accepted consequences:")
        lines.extend(f"- {a}" for a in report.annotations)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Admission telemetry payloads (live-view contract)
# --------------------------------------------------------------------------- #
#
# These translate the admission dataclasses into the JSON the web construct-
# admission view reduces. The Temporal activities emit them when a ``workspace_id``
# is present; pure state tests use ``workspace_id=None`` and have no side effects.


def _admission_plan_payload(model: ModelSpec, order: Sequence[str]) -> UncheckedJsonObject:
    """The static admission plan: constructs in admission order + the DAG edges among them."""
    order_set = set(order)
    edges: list[dict[str, str]] = []
    for edge in get_edges(model):
        cause = edge["cause"]
        effect = edge["effect"]
        if cause in order_set and effect in order_set:
            edges.append({"cause": str(cause), "effect": str(effect)})
    constructs = [{"name": name, "parents": construct_parents(model, name)} for name in order]
    return {"constructs": constructs, "edges": edges, "max_attempts": _MAX_ATTEMPTS_PER_CONSTRUCT}


def _admission_parameters_payload(contribution: ConstructContribution) -> list[UncheckedJsonObject]:
    """Canonical parameter values for the admission inspector."""
    return [parameter.model_dump(mode="json") for parameter in contribution.parameters]


def _check_result_payload(result: CheckResult) -> UncheckedJsonObject:
    """A reachability CheckResult in the admission-view contract (mode from the severity table)."""
    return {
        "check": result.check,
        "target": result.target,
        "value": result.value,
        "band": result.band,
        "passed": result.passed,
        "note": result.note,
        "diagnosis": list(result.diagnosis),
        "mode": CHECK_MODES[result.check],
    }


def _timing_payload(timing: AdmissionTiming) -> UncheckedJsonObject:
    return {
        "phase": timing.phase,
        "label": timing.label,
        "duration_ms": timing.duration_ms,
        "checks": list(timing.checks),
    }


def _admission_report_payload(
    report: ConstructAdmissionReport,
    contribution: ConstructContribution,
    attempt: int,
    coupled_recheck: UncheckedJsonObject | None = None,
) -> UncheckedJsonObject:
    """One attempt's battery outcome + authored priors, in the admission-view contract.

    ``coupled_recheck`` (present only when admitting this construct closed a feedback loop)
    carries the closed-loop re-evaluation of the already-admitted cycle member(s).
    """
    payload: UncheckedJsonObject = {
        "name": report.name,
        "attempt": attempt,
        "outcome": report.outcome,
        "admitted": report.admitted,
        "annotations": list(report.annotations),
        "results": [_check_result_payload(r) for r in report.results],
        "timings": [_timing_payload(timing) for timing in report.timings],
        "parameters": _admission_parameters_payload(contribution),
    }
    if coupled_recheck is not None:
        payload["coupled_recheck"] = coupled_recheck
        payload["timings"].append(
            {
                "phase": "coupled_recheck",
                "label": "Coupled subsystem recheck",
                "duration_ms": sum(timing["duration_ms"] for timing in coupled_recheck["timings"]),
                "checks": [],
            }
        )
    return payload


# --------------------------------------------------------------------------- #
# Construct-build session state + tool
# --------------------------------------------------------------------------- #


@dataclass
class ConstructBuildState:
    """Mutable state driving the construct-by-construct admission loop."""

    model: ModelSpec
    data_for_model: pl.DataFrame
    order: list[str]
    n_draws: int = 200
    seed: int = 0
    # Live-telemetry seam (mirrors stage 2): production threads the workspace id so the
    # construct-admission view can stream; the batch/test path leaves it None and emits nothing.
    workspace_id: str | None = None
    attempt: int = 0
    admission: AdmissionState = field(init=False)
    cursor: int = 0
    search_queries: dict[str, str] = field(default_factory=dict)
    search_cache: dict[str, str] = field(default_factory=dict)
    last_report: ConstructAdmissionReport | None = None
    last_coupled_results: tuple[CheckResult, ...] = ()
    last_tool_feedback: str | None = None
    submission_made: bool = False
    # Kept so a loop-closing admission can re-run the battery on already-admitted members.
    admitted_contributions: dict[str, ConstructContribution] = field(default_factory=dict)
    _catalog_by_prefix: dict[frozenset[str], ParamCatalog] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.admission = AdmissionState(model=self.model)

    @property
    def current_construct(self) -> str | None:
        return self.order[self.cursor] if self.cursor < len(self.order) else None

    def parameter_inventory_for(self, construct: str) -> AdmissionTurnInventory:
        """Return the cached compiler delta for the active cumulative prefix."""
        admitted = frozenset(self.admission.names)
        current_prefix = admitted | {construct}
        current_catalog = self._catalog_by_prefix.get(current_prefix)
        if current_catalog is None:
            current_catalog = ParamCatalog.from_model(
                model_for_constructs(self.model, set(current_prefix))
            )
            self._catalog_by_prefix[current_prefix] = current_catalog

        previous_catalog = None
        if admitted:
            previous_catalog = self._catalog_by_prefix.get(admitted)
            if previous_catalog is None:
                previous_catalog = ParamCatalog.from_model(
                    model_for_constructs(self.model, set(admitted))
                )
                self._catalog_by_prefix[admitted] = previous_catalog

        return derive_admission_turn_inventory(
            construct=construct,
            admitted=admitted,
            current_catalog=current_catalog,
            previous_catalog=previous_catalog,
        )

    def submit_construct(
        self,
        *,
        construct: Mapping[str, Any],
        edges: Sequence[Mapping[str, Any]],
        parameters: Sequence[Mapping[str, Any]],
        accept: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        self.submission_made = True
        expected = self.current_construct
        if expected is None:
            return "All constructs are already admitted; no further submission is needed."
        entity = Construct.model_validate(construct)
        expected_entity = next(item for item in self.model.constructs if item.name == expected)
        if entity.id != expected_entity.id:
            return f"Out-of-order submission: the active construct is {expected_entity.id!r}."
        construct_name = entity.name
        if any(indicator.likelihood is None for indicator in entity.indicators):
            return "Every submitted indicator requires its likelihood."
        payload = {
            "construct": entity.model_dump(mode="json"),
            "edges": list(edges),
            "parameters": list(parameters),
        }
        try:
            contribution = contribution_from_payload(self.model, payload)
            trial_admission_state(self.admission, contribution)
        except ValueError as exc:
            return str(exc)
        if self.workspace_id:
            emit_model_spec_admission_event(
                self.workspace_id,
                "construct_checking",
                {"construct": construct_name, "attempt": self.attempt},
            )
        design_started = perf_counter_ns()
        try:
            design = build_design_info(
                self.admission,
                contribution,
                self.data_for_model,
                n_draws=self.n_draws,
                seed=self.seed,
            )
        except AggregatedCompileError as exc:
            # Compile-time identification/translation errors are revision-shaped:
            # return them as tool feedback so the admission loop can repair the
            # submission instead of crashing the activity.
            return str(exc)
        design_timing = AdmissionTiming(
            phase="design_preparation",
            label="Design preparation",
            duration_ms=(perf_counter_ns() - design_started) / 1_000_000,
        )
        prior_admitted = set(self.admission.names)
        try:
            accepted = _acceptance_map(accept)
        except ValueError as exc:
            return str(exc)
        new_state, report = admit_construct(self.admission, contribution, design, accepted=accepted)
        report = replace(report, timings=(design_timing, *report.timings))
        coupled_recheck: UncheckedJsonObject | None = None
        coupled_results: list[CheckResult] = []
        if report.admitted:
            coupled_results, coupled_recheck = self._coupled_recheck(
                construct_name, prior_admitted, new_state
            )
            if coupled_results:
                outcome, annotations = stage_outcome([*report.results, *coupled_results], accepted)
                admitted = outcome.startswith("ADMITTED")
                report = replace(
                    report,
                    outcome=outcome,
                    annotations=annotations,
                    admitted=admitted,
                )
                new_state = replace(
                    new_state,
                    annotations=(*self.admission.annotations, *annotations),
                )
        failed_soft = {
            (result.check, result.target)
            for result in (*report.results, *coupled_results)
            if not result.passed and CHECK_MODES[result.check] == "soft"
        }
        invalid_acceptances = sorted(set(accepted) - failed_soft)
        if invalid_acceptances:
            refs = ", ".join(f"{check} [{target}]" for check, target in invalid_acceptances)
            return f"Acceptance references must name current failing soft checks exactly: {refs}."
        self.last_report = report
        self.last_coupled_results = tuple(coupled_results)
        if report.admitted:
            self.admission = new_state
            self.cursor += 1
            self.admitted_contributions[construct_name] = contribution
        if self.workspace_id:
            emit_model_spec_admission_event(
                self.workspace_id,
                "construct_report",
                _admission_report_payload(report, contribution, self.attempt, coupled_recheck),
            )
        feedback = render_admission_feedback(report)
        if coupled_results:
            lines = [feedback, "", "Coupled feedback-component checks:"]
            for result in coupled_results:
                mark = "PASS" if result.passed else "FAIL"
                lines.append(
                    f"- [{mark}] {result.check} [{result.target}]: {result.value} "
                    f"(target {result.band})"
                )
                if not result.passed:
                    lines.extend(f"    · {diagnosis}" for diagnosis in result.diagnosis)
            feedback = "\n".join(lines)
        return feedback

    def _coupled_recheck(
        self,
        construct: str,
        prior_admitted: Collection[str],
        tentative_state: AdmissionState,
    ) -> tuple[list[CheckResult], UncheckedJsonObject | None]:
        """Gate loop closure by rechecking every already-admitted affected member."""
        members = [
            m
            for m in _closing_edge_effects(self.model, construct, prior_admitted)
            if m in self.admitted_contributions
        ]
        if not members:
            return [], None
        design_started = perf_counter_ns()
        design = _design_for_state(
            tentative_state,
            self.data_for_model,
            n_draws=self.n_draws,
            seed=self.seed,
        )
        timings = [
            AdmissionTiming(
                phase="design_preparation",
                label="Design preparation",
                duration_ms=(perf_counter_ns() - design_started) / 1_000_000,
            )
        ]
        raw_results: list[CheckResult] = []
        for member in members:
            target = _closed_loop_target(
                self.admitted_contributions[member], tentative_state.model.edges
            )
            member_results, member_timings = recheck_member(tentative_state, target, design)
            raw_results.extend(member_results)
            timings.extend(
                replace(
                    timing,
                    phase=f"recheck:{member}:{timing.phase}",
                    label=f"{member}: {timing.label}",
                )
                for timing in member_timings
            )
        if not raw_results:
            return [], None
        return raw_results, {
            "constructs": [*members, construct],
            "closing_edges": [f"{construct}->{m}" for m in members],
            "results": [_check_result_payload(result) for result in raw_results],
            "timings": [_timing_payload(timing) for timing in timings],
        }


SUBMIT_CONSTRUCT_SCHEMA: UncheckedJsonObject = TypeAdapter(ConstructContribution).json_schema()
for _field in ("edge_parents", "hill_parents"):
    SUBMIT_CONSTRUCT_SCHEMA["properties"].pop(_field)
SUBMIT_CONSTRUCT_SCHEMA["properties"]["accept"] = {
    "type": "array",
    "description": ("Optional target-scoped decisions accepting current soft-check consequences."),
    "items": {
        "type": "object",
        "properties": {
            "check": {"type": "string"},
            "target": {"type": "string"},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["check", "target", "rationale"],
        "additionalProperties": False,
    },
}
SUBMIT_CONSTRUCT_SCHEMA["required"] = ["construct", "edges", "parameters"]
SUBMIT_CONSTRUCT_SCHEMA["additionalProperties"] = False


__all__ = [
    "ConstructBuildState",
    "build_design_info",
    "contribution_from_payload",
    "render_admission_feedback",
]
