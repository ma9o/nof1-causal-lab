"""Automatic exact checks of current model laws on a compatible observed schedule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.model_checks import ModelPredictiveReport
from nof1_causal_lab.artifacts.predictive_provenance import PredictiveLawProvenance
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.machine.execution import is_stale
from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.artifacts.checks import SpecificationReport
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.machine.artifacts import ArtifactRecord, EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord

# Includes the exact generation engine, all reducer policies, and the fixed budget.
# Changing any of them invalidates reuse of a previous snapshot's measurements.
PREDICTIVE_POLICY_VERSION = "exact-model-checks-v5"
PREDICTIVE_DRAWS = 200
PREDICTIVE_SEED = 0


def fitted_law_report(records: Iterable[TransitionRecord], revision: GitOid) -> InferenceReport:
    """Read the committed fit that owns inherited laws and their model coordinates."""
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.machine.inference import inference_record

    fitted = inference_record(records, revision)
    if fitted is None:
        raise ValueError("Fitted model laws require their committed inference report")
    return InferenceReport.model_validate(fitted.diagnostics["report"])


def law_provenance(
    store: ArtifactStore, record: ArtifactRecord, model: ModelSpec, panel_revision: GitOid | None
) -> PredictiveLawProvenance:
    """Follow authored ancestry; a native law family alone never establishes fitting."""
    laws = model.model_dump(mode="json")["distributions"]
    current = record
    while True:
        if current.produced_by == "run:posterior":
            fitted = store.read_json_file("model", current.revision, "model.json")["distributions"]
            inherited = {key for key, value in laws.items() if fitted.get(key) == value}
            if inherited:
                fitted_panel = current.derived_from["panel"]
                mixed = inherited != set(laws)
                return PredictiveLawProvenance(
                    kind="mixed" if mixed else "fitted",
                    fitted_panel_revision=fitted_panel,
                    fitted_model_revision=current.revision,
                    interpretation="mixed"
                    if mixed
                    else (
                        "in_sample_posterior_predictive"
                        if fitted_panel == panel_revision
                        # A different panel revision does not prove held-out observations.
                        else "posterior_predictive"
                    ),
                )
        parent = current.derived_from.get("model")
        if parent is None:
            break
        current = store.read_meta("model", parent)
    # Imported joint laws can contain externally conditioned draws. Their family
    # does not establish a training panel, so their interpretation stays unknown.
    from nof1_causal_lab.numpyro_json import distribution_shape

    joint = any(any(distribution_shape(law)) for law in model.distributions.values())
    return PredictiveLawProvenance(
        kind="unknown" if joint else "authored",
        interpretation="unknown" if joint else "prior_predictive",
    )


def check_model_predictive(
    store: ArtifactStore,
    state: EpisodeState,
    model: ModelSpec,
    specification: SpecificationReport,
    *,
    previous: ModelPredictiveReport | None,
) -> tuple[ModelPredictiveReport, bool]:
    """Reuse a matching report or run one whole-model batch; never fit or repair."""
    record = state.current["model"]
    panel = state.get("panel")
    panel_revision = panel.revision if panel is not None else None
    from nof1_causal_lab.actions.data_checks import data_binding_issues, read_data_metadata

    compatible = (
        panel is not None
        and not is_stale(state, "panel")
        and not data_binding_issues(model, read_data_metadata(store, panel.revision))
    )
    law = law_provenance(store, record, model, panel_revision)
    key = scientific_id(
        "check",
        [
            PREDICTIVE_POLICY_VERSION,
            record.model_inputs["compilation"],
            record.model_inputs["belief"],
            panel_revision,
            compatible,
            PREDICTIVE_DRAWS,
            PREDICTIVE_SEED,
            law.model_dump(mode="json"),
        ],
    )
    if previous is not None and previous.input_key == key:
        return previous, True
    report = ModelPredictiveReport(
        input_key=key,
        draws=PREDICTIVE_DRAWS,
        seed=PREDICTIVE_SEED,
        model_revision=record.revision,
        panel_revision=panel_revision,
        status="not_evaluated",
        law=law,
    )
    execution = next(f for f in specification.findings if f.check == "model_execution")
    if execution.status != "passed":
        return report.model_copy(
            update={
                "reason": "MODEL_INCOMPLETE"
                if execution.status == "not_evaluated"
                else "MODEL_NOT_EXECUTABLE",
                "detail": execution.message,
            }
        ), False
    if not compatible:
        return report.model_copy(update={"reason": "NO_COMPATIBLE_PANEL"}), False

    from nof1_causal_lab.artifacts.likelihood import DistributionFamily
    from nof1_causal_lab.models.ssm.predictive.parameters import validate_simulation_laws

    if any(family != DistributionFamily.GAUSSIAN for family in numeric.diffusion_families(model)):
        return report.model_copy(
            update={
                "reason": "SIMULATION_UNSUPPORTED",
                "detail": "Exact forward simulation requires Gaussian process diffusion.",
            }
        ), False
    # Validate authored laws at their capability boundary. The simulation itself
    # is deliberately outside this handler: implementation/worker failures fail the action.
    try:
        validate_simulation_laws(model)
    except ValueError as exc:
        return report.model_copy(
            update={
                "reason": "SIMULATION_UNSUPPORTED",
                "detail": str(exc),
            }
        ), False

    from nof1_causal_lab.models.ssm.observation_support import (
        augment_wide_data_with_support_boundaries,
        validate_discrete_manifest_metadata,
        validate_observation_support,
    )
    from nof1_causal_lab.models.ssm.runtime import prepare_fit_inputs, project_observation_data

    assert panel_revision is not None
    data = store.read_parquet_file("panel", panel_revision, "panel.parquet")
    try:
        time_origin = read_data_metadata(store, panel_revision).time_origin
        if law.fitted_model_revision is not None:
            from nof1_causal_lab.machine.history import StudyRepository

            fit_origin = fitted_law_report(
                StudyRepository(store.workspace_id).attempts(), law.fitted_model_revision
            ).time_origin
            if (time_origin is None) != (fit_origin is None):
                raise ValueError(
                    "Calendar-free laws and calendar-bound observations cannot be aligned"
                )
            time_origin = fit_origin
        wide, rows = project_observation_data(data, model_spec=model, time_origin=time_origin)
        names = numeric.observation_names(model)
        missing = set(names) - set(wide.columns)
        if missing:
            raise ValueError(f"No observations for model indicators: {sorted(missing)}")
        wide = augment_wide_data_with_support_boundaries(rows, wide, names, time_origin=time_origin)
        validate_discrete_manifest_metadata(model, wide)
        validate_observation_support(model, wide)
        _, times, _, _ = prepare_fit_inputs(model, wide)
        if len(times) and law.fitted_model_revision is not None and times[0] < model.time_points[0]:
            raise ValueError("The current panel begins before the fit's first retained state")
    except ValueError as exc:
        return report.model_copy(
            update={
                "reason": "NO_COMPATIBLE_PANEL",
                "detail": str(exc),
            }
        ), False
    times = np.asarray(times)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        return report.model_copy(update={"reason": "INSUFFICIENT_OBSERVATION_TIMES"}), False
    design = SimulationSpec(start=float(times[0]), end=float(times[-1]))
    from nof1_causal_lab.artifacts.checks import PredictiveCheckFinding
    from nof1_causal_lab.models.predictive_simulation import PredictiveObservationMeanOverflow
    from nof1_causal_lab.models.ssm.predictive.simulation import (
        generate_simulation_batch,
        measure_simulation_batch,
    )

    try:
        batch = generate_simulation_batch(
            model,
            design,
            comparison_data=data,
            time_origin=time_origin,
            times=times,
            draws=PREDICTIVE_DRAWS,
            seed=PREDICTIVE_SEED,
        )
    except PredictiveObservationMeanOverflow as exc:
        # This typed scientific failure is raised before an unsafe emission draw.
        # Other generator exceptions still fail the action.
        return report.model_copy(
            update={
                "status": "failed",
                "design": design,
                "findings": (
                    PredictiveCheckFinding(
                        check="C1a finiteness",
                        target=", ".join(exc.bad_manifest_names),
                        value=f"{len(exc.failing_draw_indices)}/{exc.n_draws} draws overflow",
                        band="0 non-finite emission means",
                        passed=False,
                        note=str(exc),
                        reason="NONFINITE_EMISSION_MEAN",
                    ),
                    PredictiveCheckFinding(
                        check="predictive_measurements",
                        target="whole_model",
                        value="not_evaluated",
                        band="A complete finite predictive batch",
                        passed=None,
                        reason="NONFINITE_EMISSION_MEAN",
                        note="Emission overflow prevented completion of the shared batch.",
                    ),
                ),
            }
        ), False
    findings, checks = measure_simulation_batch(
        model, batch, groups=("dynamics", "measurement", "data_comparison")
    )
    return report.model_copy(
        update={
            "status": "failed"
            if any(f.passed is False for f in findings)
            or (
                checks is not None and any(not item.passed for item in checks.per_variable_warnings)
            )
            else "passed",
            "design": design,
            "findings": findings,
            "predictive_checks": checks,
        }
    ), False
