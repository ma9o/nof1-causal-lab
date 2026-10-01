"""Inference lineage is a journal query, never a field of the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.artifacts.predictive_provenance import PredictiveLawProvenance
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.state import is_stale

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.study.records import AttemptRecord
    from nof1_causal_lab.study.state import ArtifactRecord, StudyState
    from nof1_causal_lab.study.store import ArtifactStore


def scientific_inference_report(model: ModelSpec, report: InferenceReport) -> InferenceReport:
    """Join engine diagnostics to the same exact scientific bindings as posterior marginals."""
    from nof1_causal_lab.artifacts.identity import ParameterRef
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    mcmc = report.inference_diagnostics.get("mcmc")
    if not isinstance(mcmc, dict) or "per_parameter" not in mcmc:
        return report
    rows = TypeAdapter(list[JsonObject]).validate_python(mcmc["per_parameter"])
    # Already-referenced scientific reports need no runtime compilation (and
    # may describe archived models whose executable definitions were not retained).
    if all("subject" in row for row in rows):
        return report
    bindings, auxiliary = parameter_bindings(model)
    subjects = {
        coordinate: ParameterRef(parameter_id=binding.parameter_id, element_id=element)
        for binding in bindings
        for element, coordinate in binding.coordinates.items()
    }
    labels = {
        (binding.parameter_id, element): label
        for binding in bindings
        for element, label in binding.elements.items()
    }
    referenced: list[JsonObject] = []
    for row in rows:
        if "coordinate" in row:
            coordinate = ParameterCoordinate.model_validate(row["coordinate"])
            if coordinate in auxiliary:
                continue
            subject = subjects[coordinate]
        else:
            subject = ParameterRef.model_validate(row["subject"])
        referenced.append(
            {
                **row,
                "subject": subject.model_dump(mode="json"),
                "parameter": labels[(subject.parameter_id, subject.element_id)],
            }
        )
    return type(report).model_validate(
        {
            **report.model_dump(),
            "inference_diagnostics": {
                **report.inference_diagnostics,
                "mcmc": {**mcmc, "per_parameter": referenced},
            },
        }
    )


def inference_record[T: AttemptRecord](records: Iterable[T], model_revision: GitOid) -> T | None:
    """Find the committed inference operation that produced this exact model value."""
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and record.action == "fit"
            and any(
                info.artifact_id == "model" and info.revision == model_revision
                for info in record.produced
            )
        ),
        None,
    )


def inference_is_current(state: StudyState) -> bool:
    """Check the fitted revision's inputs; the model itself has no fitted-status flag."""
    info = state.get("model")
    return (
        info is not None
        and info.produced_by == "fit"
        and state.matches_inputs("model", "panel")
        and not is_stale(state, "panel")
    )


def _input_pins(record: AttemptRecord) -> dict[ArtifactId, GitOid]:
    return TypeAdapter(dict[ArtifactId, GitOid]).validate_python(record.diagnostics["input_pins"])


def inference_report_record[T: AttemptRecord](records: Iterable[T], state: StudyState) -> T | None:
    """Reports can survive in history even when a numerical value was not retained."""
    model = state.get("model")
    if model is None:
        return None
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and record.action == "fit"
            and (
                inference_record([record], model.revision) is not None
                or (
                    record.diagnostics.get("retention") == "report_only"
                    and _input_pins(record)["model"] == model.revision
                )
            )
        ),
        None,
    )


def inference_report_is_current(record: AttemptRecord, state: StudyState) -> bool:
    pins = _input_pins(record)
    return (
        state.has("panel")
        and state.current["panel"].revision == pins["panel"]
        and not is_stale(state, "panel")
    )


def read_data_metadata(store: ArtifactStore, revision: GitOid) -> PreparedDataMetadata:
    return PreparedDataMetadata.model_validate(
        store.read_json_file("panel", revision, json_filename("panel", "metadata"))
    )


def fitted_law_report(records: Iterable[AttemptRecord], revision: GitOid) -> InferenceReport:
    """Read the committed fit that owns inherited laws and their model coordinates."""
    from nof1_causal_lab.artifacts.posterior import InferenceReport

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
        if current.produced_by == "fit":
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
