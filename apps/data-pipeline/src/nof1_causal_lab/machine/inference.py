"""Inference lineage is a journal query, never a field of the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.machine.execution import is_stale

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord


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
    return report.model_copy(
        update={
            "inference_diagnostics": {
                **report.inference_diagnostics,
                "mcmc": {**mcmc, "per_parameter": referenced},
            }
        }
    )


def inference_record[T: TransitionRecord](records: Iterable[T], model_revision: GitOid) -> T | None:
    """Find the committed inference operation that produced this exact model value."""
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and record.operation_id == "posterior"
            and any(
                info.artifact_id == "model" and info.revision == model_revision
                for info in record.produced
            )
        ),
        None,
    )


def inference_is_current(state: EpisodeState) -> bool:
    """Check the fitted revision's inputs; the model itself has no fitted-status flag."""
    info = state.get("model")
    return (
        info is not None
        and info.produced_by == "run:posterior"
        and state.matches_inputs("model", "panel")
        and not is_stale(state, "panel")
    )


def inference_report_record[T: TransitionRecord](
    records: Iterable[T], state: EpisodeState
) -> T | None:
    """Reports can survive in history even when a numerical value was not retained."""
    model = state.get("model")
    if model is None:
        return None
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and record.operation_id == "posterior"
            and (
                inference_record([record], model.revision) is not None
                or (
                    record.diagnostics.get("retention") == "report_only"
                    and record.diagnostics["input_pins"]["model"] == model.revision
                )
            )
        ),
        None,
    )


def inference_report_is_current(record: TransitionRecord, state: EpisodeState) -> bool:
    pins = record.diagnostics["input_pins"]
    return (
        state.has("panel")
        and state.current["panel"].revision == pins["panel"]
        and not is_stale(state, "panel")
    )


def inference_input_revision(store: ArtifactStore, model_revision: GitOid) -> GitOid:
    """Refitting restarts from the input law, so observations are counted exactly once."""
    info = store.read_meta("model", model_revision)
    while info.produced_by == "run:posterior" or "model" in info.derived_from:
        parent = info.derived_from["model"]
        previous = store.read_meta("model", parent)
        if (
            info.produced_by != "run:posterior"
            and info.model_inputs["belief"] != previous.model_inputs["belief"]
        ):
            break
        info = previous
    return info.revision
