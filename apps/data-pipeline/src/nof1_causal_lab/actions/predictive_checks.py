"""Automatic exact checks of current model laws on a compatible observed schedule."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.checks import (
    Evaluated,
    NotEvaluated,
    NumericCriterionEvidence,
    PredictiveSubject,
)
from nof1_causal_lab.artifacts.identity import IndicatorRef, scientific_id
from nof1_causal_lab.artifacts.model_checks import ModelPredictiveReport
from nof1_causal_lab.artifacts.predictive_provenance import FittedLawProvenance, MixedLawProvenance
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.study.lineage import fitted_law_report, law_provenance, read_data_metadata
from nof1_causal_lab.study.state import is_stale

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.checks import SpecificationReport
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.study.state import StudyState
    from nof1_causal_lab.study.store import ArtifactStore

# Includes the exact generation engine, all reducer policies, and the fixed budget.
# Changing any of them invalidates reuse of a previous snapshot's measurements.
PREDICTIVE_POLICY_VERSION = "exact-model-checks-v6"
PREDICTIVE_DRAWS = 200
PREDICTIVE_SEED = 0


def check_model_predictive(
    store: ArtifactStore,
    state: StudyState,
    model: ModelSpec,
    specification: SpecificationReport,
    *,
    previous: ModelPredictiveReport | None,
) -> tuple[ModelPredictiveReport, bool]:
    """Reuse a matching report or run one whole-model batch; never fit or repair."""
    record = state.current["model"]
    panel = state.get("panel")
    panel_revision = panel.revision if panel is not None else None
    from nof1_causal_lab.models.model_inputs import data_binding_issues

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
    execution = next(f for f in specification.findings if f.subject == "model_execution")
    if isinstance(execution, NotEvaluated) or execution.outcome != "passed":
        return report.not_evaluated(
            "MODEL_INCOMPLETE" if isinstance(execution, NotEvaluated) else "MODEL_NOT_EXECUTABLE"
        ), False
    if not compatible:
        return report.not_evaluated("NO_COMPATIBLE_PANEL"), False

    from nof1_causal_lab.artifacts.likelihood import DistributionFamily
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model

    compiled = compile_executable_model(model)
    if any(
        family != DistributionFamily.GAUSSIAN for family in numeric.diffusion_families(compiled)
    ):
        return report.not_evaluated(
            "SIMULATION_UNSUPPORTED",
            "Exact forward simulation requires Gaussian process diffusion.",
        ), False
    from nof1_causal_lab.models.ssm.preflight import ObservationPreflightError
    from nof1_causal_lab.models.ssm.runtime import PanelPreparationFailure, bind_panel

    assert panel_revision is not None
    data = store.read_parquet_file("panel", panel_revision, "panel.parquet")
    try:
        time_origin = read_data_metadata(store, panel_revision).time_origin
        if isinstance(law, (FittedLawProvenance, MixedLawProvenance)):
            from nof1_causal_lab.study.history import StudyRepository

            fit_origin = fitted_law_report(
                StudyRepository(store.workspace_id).attempts(), law.fitted_model_revision
            ).time_origin
            if (time_origin is None) != (fit_origin is None):
                raise ObservationPreflightError(
                    "Calendar-free laws and calendar-bound observations cannot be aligned"
                )
            time_origin = fit_origin
        bound = bind_panel(data, model=compiled, time_origin=time_origin)
        if isinstance(bound, PanelPreparationFailure):
            return report.not_evaluated("NO_COMPATIBLE_PANEL", bound.message), False
        times = bound.times
        if (
            len(times)
            and isinstance(law, (FittedLawProvenance, MixedLawProvenance))
            and times[0] < model.time_points[0]
        ):
            raise ObservationPreflightError(
                "The current panel begins before the fit's first retained state"
            )
    except ObservationPreflightError as exc:
        return report.not_evaluated("NO_COMPATIBLE_PANEL", str(exc)), False
    times = np.asarray(times)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        return report.not_evaluated("INSUFFICIENT_OBSERVATION_TIMES"), False
    design = SimulationSpec(start=float(times[0]), end=float(times[-1]))
    from nof1_causal_lab.models.predictive_simulation import PredictiveObservationMeanOverflow
    from nof1_causal_lab.models.ssm.predictive.simulation import (
        generate_simulation_batch,
        measure_simulation_batch,
    )

    try:
        batch = generate_simulation_batch(
            bound,
            design,
            draws=PREDICTIVE_DRAWS,
            seed=PREDICTIVE_SEED,
        )
    except PredictiveObservationMeanOverflow as exc:
        # This typed scientific failure is raised before an unsafe emission draw.
        # Other generator exceptions still fail the action.
        return report.evaluated(
            design,
            (
                *(
                    Evaluated(
                        subject=PredictiveSubject(
                            check="C1a finiteness",
                            construct_id=construct.id,
                            target=IndicatorRef(id=indicator.id),
                        ),
                        outcome="failed",
                        evidence=(
                            NumericCriterionEvidence(
                                criterion="overflow_draw_count",
                                value=float(len(exc.failing_draw_indices)),
                                upper=0.0,
                                note=str(exc),
                                display_value=f"{len(exc.failing_draw_indices)}/{exc.n_draws} draws overflow",
                                band_label="0 non-finite emission means",
                            ),
                        ),
                    )
                    for construct in model.constructs
                    for indicator in construct.indicators
                    if indicator.name in exc.bad_manifest_names
                ),
                NotEvaluated(
                    subject=PredictiveSubject(
                        check="predictive_measurements", target="whole_model"
                    ),
                    reason="NONFINITE_EMISSION_MEAN",
                    detail="Emission overflow prevented completion of the shared batch.",
                ),
            ),
        ), False
    findings, checks = measure_simulation_batch(
        compiled, batch, groups=("dynamics", "measurement", "data_comparison"), clock=time.monotonic
    )
    return report.evaluated(design, findings, checks), False
