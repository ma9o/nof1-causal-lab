"""Inference lineage is a journal query, never a field of the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.predictive_provenance import (
    AuthoredLawProvenance,
    FittedLawProvenance,
    MixedLawProvenance,
    PredictiveLawProvenance,
    UnknownLawProvenance,
)
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.records import ModelFitResult, StudyRevision, inference_record
from nof1_causal_lab.study.state import is_stale
from nof1_causal_lab.study.store import read_model

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReportCore
    from nof1_causal_lab.study.state import ArtifactRecord, StudyState
    from nof1_causal_lab.study.store import ArtifactStore


def inference_report_record[T: StudyRevision](records: Iterable[T], state: StudyState) -> T | None:
    """Find retained numerical evidence for the selected model, ignoring report-only history."""
    model = state.get("model")
    return inference_record(records, model.revision) if model is not None else None


def inference_report_is_current(result: ModelFitResult, state: StudyState) -> bool:
    return (
        state.has("panel")
        and state.current["panel"].revision == result.panel.revision
        and not is_stale(state, "panel")
    )


def read_data_metadata(store: ArtifactStore, revision: GitOid) -> PreparedDataMetadata:
    return TypeAdapter(PreparedDataMetadata).validate_python(
        store.read_json_file("panel", revision, json_filename("panel", "metadata"))
    )


def fitted_law_report(
    store: ArtifactStore, records: Iterable[StudyRevision], revision: GitOid
) -> InferenceReportCore:
    """Read the committed fit that owns inherited laws and their model coordinates."""
    fitted = inference_record(records, revision)
    if fitted is None:
        raise ValueError("Fitted model laws require their committed inference report")
    assert fitted.record.attempt.action == "fit"
    assert fitted.record.attempt.outcome.status == "applied"
    result = fitted.record.attempt.outcome.result
    assert result is not None
    from nof1_causal_lab.actions.fit import read_inference_report

    return read_inference_report(store, revision, result.evidence).core


def law_provenance(
    store: ArtifactStore, record: ArtifactRecord, model: ModelSpec, panel_revision: GitOid | None
) -> PredictiveLawProvenance:
    """Follow authored ancestry; a native law family alone never establishes fitting."""
    laws = model.model_dump(mode="json")["distributions"]
    current = record
    while True:
        if current.produced_by == "fit":
            fitted = read_model(store, current.revision).model_dump(mode="json")["distributions"]
            inherited = {
                key
                for key, value in laws.items()
                if key in model.law_layouts and fitted.get(key) == value
            }
            if inherited:
                fitted_panel = current.derived_from["panel"]
                if inherited != set(laws):
                    return MixedLawProvenance(
                        fitted_panel_revision=fitted_panel,
                        fitted_model_revision=current.revision,
                    )
                return FittedLawProvenance(
                    fitted_panel_revision=fitted_panel,
                    fitted_model_revision=current.revision,
                    interpretation=(
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
    return UnknownLawProvenance() if joint else AuthoredLawProvenance()
