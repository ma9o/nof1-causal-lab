"""Read-only selection and comparison contracts for immutable scientific inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identity import (
    GitOid,
    GitRef,
)
from nof1_causal_lab.models.model_structure import (
    compare_model_graph,
    compare_parameters,
    model_graph_entities,
)
from nof1_causal_lab.study.view_models import (
    ModelDiffReport,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReportCore
    from nof1_causal_lab.artifacts.simulation import SimulationReport


def _model_revision(
    workspace_id: str, revision: GitOid
) -> tuple[ModelSpec, GitRef, InferenceReportCore | None, SimulationReport | None]:
    """Select an exact model tree or the model and recorded evidence at a Git commit."""
    import pygit2

    from nof1_causal_lab.study.errors import StudyLookupError
    from nof1_causal_lab.study.snapshots import ModelReader
    from nof1_causal_lab.study.store import ArtifactStore, read_model

    store = ArtifactStore(workspace_id)
    oid = pygit2.Oid(hex=revision)
    if oid not in store.repo:
        raise StudyLookupError(f"Unknown model revision {revision}")
    obj = store.repo[oid]
    if isinstance(obj, pygit2.Tree):
        if "model.json" not in obj:
            raise StudyLookupError("The selected tree is not a model artifact")
        model = read_model(store, revision)
        return (
            model,
            GitRef(workspace_id=workspace_id, revision=revision, path="model.json"),
            None,
            None,
        )
    if obj.type != pygit2.GIT_OBJECT_COMMIT:
        raise StudyLookupError("Select a model artifact tree or a study commit")
    reader = ModelReader(workspace_id, at=revision)
    if reader.model is None:
        raise StudyLookupError("The selected checkpoint contains no model")
    fit, simulation = reader.inference_report, reader.simulation()
    return (
        reader.model,
        GitRef(workspace_id=workspace_id, revision=revision, path="artifacts/model/model.json"),
        fit.value.core if fit is not None and fit.source.validity == "fresh" else None,
        simulation.value
        if simulation is not None and simulation.source.validity == "fresh"
        else None,
    )


def model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffReport:
    """Inspect scientific definition changes and evidence without fitting or simulation."""
    from nof1_causal_lab.actions.checks import check_specification
    from nof1_causal_lab.models.model_inputs import input_fingerprints
    from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model
    from nof1_causal_lab.study.snapshots import ModelReader

    left, before, before_fit, before_simulation = _model_revision(workspace_id, before_id)
    right, after, after_fit, after_simulation = _model_revision(workspace_id, after_id)
    # A study has one question, so its outcome scopes both revisions alike.
    reader = ModelReader(workspace_id)
    scoped = reader.scoped(left), reader.scoped(right)
    checks = []
    for selection in scoped:
        compiled = compile_model(selection)
        checks.append(check_specification(compiled, compile_fit_inputs(compiled, selection)))
    changes = compare_parameters(left, right)
    fingerprints = input_fingerprints(left)
    constructs, edges = compare_model_graph(*scoped)
    graphs = tuple(model_graph_entities(selection) for selection in scoped)
    return ModelDiffReport(
        before=before,
        after=after,
        parameters=tuple(changes),
        constructs=constructs,
        edges=edges,
        before_dispositions=scoped[0].structural_dispositions
        if left.measurement_clock is not None and left.indicators
        else (),
        after_dispositions=scoped[1].structural_dispositions
        if right.measurement_clock is not None and right.indicators
        else (),
        before_dynamic_construct_ids=tuple(item.id for item in graphs[0][0] if item.is_dynamic),
        after_dynamic_construct_ids=tuple(item.id for item in graphs[1][0] if item.is_dynamic),
        before_checks=checks[0],
        after_checks=checks[1],
        before_fit=before_fit,
        after_fit=after_fit,
        before_simulation=before_simulation,
        after_simulation=after_simulation,
        changed_inputs=tuple(
            key for key, value in input_fingerprints(right).items() if value != fingerprints[key]
        ),
    )
