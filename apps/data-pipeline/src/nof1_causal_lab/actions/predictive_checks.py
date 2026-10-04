"""Automatic exact checks of current model laws on a compatible observed schedule."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.checks import (
    Evaluated,
    NotEvaluated,
    NumericCriterionEvidence,
    PredictiveSubject,
)
from nof1_causal_lab.artifacts.identity import IndicatorRef, scientific_id
from nof1_causal_lab.artifacts.model_checks import (
    EvaluatedPredictiveChecks,
    ModelPredictiveReport,
    UnavailablePredictiveChecks,
)
from nof1_causal_lab.artifacts.predictive_provenance import FittedLawProvenance, MixedLawProvenance
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.study.lineage import fitted_law_report, law_provenance, read_data_metadata
from nof1_causal_lab.study.state import is_stale
from nof1_causal_lab.study.store import cached_value

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.checks import (
        PredictiveAssessment,
        PredictiveCheckReason,
    )
    from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import ModelCompilationResult
    from nof1_causal_lab.study.state import StudyState
    from nof1_causal_lab.study.store import ArtifactStore

# Includes the exact generation engine, all reducer policies, and the fixed budget.
# Changing any of them invalidates reuse of a previous snapshot's measurements.
PREDICTIVE_DRAWS = 200
PREDICTIVE_SEED = 0


def check_model_predictive(
    store: ArtifactStore,
    state: StudyState,
    selection: StructuralSelection,
    compilation: Callable[[], ModelCompilationResult],
) -> tuple[ModelPredictiveReport, bool]:
    """Reuse a matching report or run one batch of the executed model; never fit or repair."""
    model = selection.model
    record = state.current["model"]
    panel = state.get("panel")
    panel_revision = panel.revision if panel is not None else None
    from nof1_causal_lab.models.model_inputs import data_binding_issues, input_fingerprints

    fingerprints = input_fingerprints(model)

    compatible = (
        panel is not None
        and not is_stale(state, "panel")
        and not data_binding_issues(model, read_data_metadata(store, panel.revision))
    )
    law = law_provenance(store, record, model, panel_revision)
    key = scientific_id(
        "check",
        [
            fingerprints["compilation"],
            fingerprints["belief"],
            selection.outcome,
            panel_revision,
            compatible,
            PREDICTIVE_DRAWS,
            PREDICTIVE_SEED,
            law.model_dump(mode="json"),
        ],
    )

    def render() -> ModelPredictiveReport:
        def not_evaluated(
            reason: PredictiveCheckReason, detail: str | None = None
        ) -> ModelPredictiveReport:
            return ModelPredictiveReport(
                draws=PREDICTIVE_DRAWS,
                seed=PREDICTIVE_SEED,
                model_revision=record.revision,
                panel_revision=panel_revision,
                law=law,
                evaluation=UnavailablePredictiveChecks(reason=reason, detail=detail),
            )

        def evaluated(
            findings: tuple[PredictiveAssessment, ...],
            predictive_checks: PosteriorPredictiveChecks | None = None,
        ) -> ModelPredictiveReport:
            return ModelPredictiveReport(
                draws=PREDICTIVE_DRAWS,
                seed=PREDICTIVE_SEED,
                model_revision=record.revision,
                panel_revision=panel_revision,
                law=law,
                evaluation=EvaluatedPredictiveChecks(
                    findings=findings, predictive_checks=predictive_checks
                ),
            )

        from nof1_causal_lab.models.ssm.compile.inputs import IncompleteModel, UnsupportedFit

        compiled = compilation()
        if isinstance(compiled, IncompleteModel):
            return not_evaluated("MODEL_INCOMPLETE")
        if isinstance(compiled, UnsupportedFit):
            return not_evaluated("MODEL_NOT_EXECUTABLE")
        if not compatible:
            return not_evaluated("NO_COMPATIBLE_PANEL")

        from nof1_causal_lab.artifacts.likelihood import DistributionFamily

        if any(
            family != DistributionFamily.GAUSSIAN for family in numeric.diffusion_families(compiled)
        ):
            return not_evaluated(
                "SIMULATION_UNSUPPORTED",
                "Exact forward simulation requires Gaussian process diffusion.",
            )
        from nof1_causal_lab.models.ssm.runtime import PanelPreparationFailure, bind_panel

        assert panel_revision is not None
        data = store.read_parquet_file("panel", panel_revision, "panel.parquet")
        time_origin = read_data_metadata(store, panel_revision).time_origin
        if isinstance(law, (FittedLawProvenance, MixedLawProvenance)):
            from nof1_causal_lab.study.history import StudyRepository

            fit_origin = fitted_law_report(
                store, StudyRepository(store.workspace_id).attempts(), law.fitted_model_revision
            ).time_origin
            if (time_origin is None) != (fit_origin is None):
                return not_evaluated(
                    "NO_COMPATIBLE_PANEL",
                    "Calendar-free laws and calendar-bound observations cannot be aligned",
                )
            time_origin = fit_origin
        bound = bind_panel(data, model=compiled, time_origin=time_origin)
        if isinstance(bound, PanelPreparationFailure):
            return not_evaluated("NO_COMPATIBLE_PANEL", bound.message)
        times = bound.times
        if (
            len(times)
            and isinstance(law, (FittedLawProvenance, MixedLawProvenance))
            and times[0] < model.time_points[0]
        ):
            return not_evaluated(
                "NO_COMPATIBLE_PANEL",
                "The current panel begins before the fit's first retained state",
            )
        times = np.asarray(times)
        if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            return not_evaluated("INSUFFICIENT_OBSERVATION_TIMES")
        from nof1_causal_lab.models.predictive_simulation import PredictiveObservationMeanOverflow
        from nof1_causal_lab.models.ssm.predictive.simulation import (
            generate_simulation_batch,
            measure_simulation_batch,
        )

        try:
            batch = generate_simulation_batch(
                bound,
                start=float(times[0]),
                end=float(times[-1]),
                draws=PREDICTIVE_DRAWS,
                seed=PREDICTIVE_SEED,
            )
        except PredictiveObservationMeanOverflow as exc:
            # This typed scientific failure is raised before an unsafe emission draw.
            # Other generator exceptions still fail the action.
            return evaluated(
                (
                    *(
                        Evaluated(
                            subject=PredictiveSubject(
                                check="C1a finiteness",
                                construct_id=construct.id,
                                target=IndicatorRef(id=indicator.observation.id),
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
                        if indicator.observation.name in exc.bad_manifest_names
                    ),
                    NotEvaluated(
                        subject=PredictiveSubject(
                            check="predictive_measurements", target="whole_model"
                        ),
                        reason="NONFINITE_EMISSION_MEAN",
                        detail="Emission overflow prevented completion of the shared batch.",
                    ),
                ),
            )
        findings, checks = measure_simulation_batch(
            compiled,
            batch,
            groups=("dynamics", "measurement", "data_comparison"),
            clock=time.monotonic,
        )
        return evaluated(findings, checks)

    value, reused = cached_value(
        store.workspace_id, ("predictive", key), TypeAdapter(ModelPredictiveReport), render
    )
    return value.revised(model_revision=record.revision), reused
