"""Read-only selection and comparison contracts for immutable scientific inputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.artifacts.checks import SpecificationReport  # noqa: TC001
from nof1_causal_lab.artifacts.identity import ModelRevision, ParameterId
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec  # noqa: TC001
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo  # noqa: TC001

if TYPE_CHECKING:
    from nof1_causal_lab.machine.store import ArtifactStore


class RevisionCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[ArtifactVersionInfo]
    raw_data: list[ArtifactVersionInfo]
    panels: list[ArtifactVersionInfo]


class ParameterChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameter_id: ParameterId
    before: ParameterSpec | None
    after: ParameterSpec | None
    change: str


class ModelComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before: ModelRevision
    after: ModelRevision
    parameters: list[ParameterChange]
    changed_inputs: list[str]
    before_checks: SpecificationReport
    after_checks: SpecificationReport
    before_fit: InferenceReport | None
    after_fit: InferenceReport | None
    before_simulation: SimulationReport | None
    after_simulation: SimulationReport | None


def compare_models(store: ArtifactStore, before: int, after: int) -> ModelComparison:
    from nof1_causal_lab.actions.checks import check_specification
    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.machine.inference import inference_record
    from nof1_causal_lab.machine.store import EpisodeJournal
    from nof1_causal_lab.models.model_inputs import input_fingerprints

    records = EpisodeJournal(store.workspace_id).read_all()

    def fit_report(version):
        record = inference_record(records, version)
        return InferenceReport.model_validate(record.diagnostics["report"]) if record else None

    def simulation_report(version):
        for record in reversed(records):
            if (
                record.status == "applied"
                and record.move.kind == "run"
                and record.move.operation_id == "simulate"
            ):
                report = SimulationReport.model_validate(record.diagnostics["report"])
                if report.model.version == version:
                    return report
        return None

    left, right = read_model(store, before), read_model(store, after)
    old, new = {p.id: p for p in left.parameters}, {p.id: p for p in right.parameters}
    changes = []
    for identity in sorted(old.keys() | new.keys()):
        a, b = old.get(identity), new.get(identity)
        if a == b and (
            a is None
            or a.distribution is None
            or json.dumps(
                left.model_dump(mode="json")["distributions"][a.distribution], sort_keys=True
            )
            == json.dumps(
                right.model_dump(mode="json")["distributions"][a.distribution], sort_keys=True
            )
        ):
            continue
        change = (
            "added"
            if a is None
            else "removed"
            if b is None
            else "released"
            if a.value is not None and b.value is None
            else "pinned"
            if a.value is None and b.value is not None
            else "revised"
        )
        changes.append(ParameterChange(parameter_id=identity, before=a, after=b, change=change))
    fingerprints = input_fingerprints(left)
    return ModelComparison(
        before=ModelRevision(workspace_id=store.workspace_id, version=before),
        after=ModelRevision(workspace_id=store.workspace_id, version=after),
        parameters=changes,
        before_checks=check_specification(left),
        after_checks=check_specification(right),
        before_fit=fit_report(before),
        after_fit=fit_report(after),
        before_simulation=simulation_report(before),
        after_simulation=simulation_report(after),
        changed_inputs=[
            key for key, value in input_fingerprints(right).items() if value != fingerprints[key]
        ],
    )
