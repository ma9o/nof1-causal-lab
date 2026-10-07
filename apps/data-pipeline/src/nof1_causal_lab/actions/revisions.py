"""Read-only selection and comparison contracts for immutable scientific inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.actions.io import ModelDiffOutput
from nof1_causal_lab.models.model_structure import (
    compare_model_graph,
    compare_parameters,
    model_graph_entities,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import GitOid, GitRef
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReportCore
    from nof1_causal_lab.artifacts.simulation import SimulationReport


def _model_revision(
    workspace_id: str, revision: GitOid
) -> tuple[ModelSpec | None, GitRef | None, InferenceReportCore | None, SimulationReport | None]:
    """Select an exact model tree or the model and recorded evidence at a Git commit."""
    import pygit2

    from nof1_causal_lab.study.errors import StudyLookupError
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import Applied
    from nof1_causal_lab.study.snapshots import ModelReader
    from nof1_causal_lab.study.store import ArtifactStore, read_model

    store = ArtifactStore(workspace_id)
    oid = pygit2.Oid(hex=revision)
    if oid not in store.repo:
        raise StudyLookupError(f"Unknown model revision {revision}")
    obj = store.repo[oid]
    if isinstance(obj, pygit2.Tree):
        ref = store.model_ref(revision)
        model = read_model(store, revision)
        return (
            model,
            ref,
            None,
            None,
        )
    if obj.type != pygit2.GIT_OBJECT_COMMIT:
        raise StudyLookupError("Select a model artifact tree or a study commit")
    repository = StudyRepository(workspace_id)
    if "logs" in obj.peel(pygit2.Commit).tree:
        selected = repository.record(revision)
        if not isinstance(selected.record.attempt.outcome, Applied):
            # Failure leaves record no scientific change; compare their exact execution parent.
            revision = selected.parent_ids[0]
    reader = ModelReader(workspace_id, at=revision)
    if reader.model is None:
        return None, None, None, None
    fit, simulation = reader.inference_report, reader.simulation()
    return (
        reader.model,
        store.model_ref(reader.state.current["model"].revision),
        fit.core if fit is not None else None,
        simulation
        if simulation is not None
        and simulation.evidence.model.revision == reader.state.current["model"].revision
        else None,
    )


def model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffOutput:
    """Inspect scientific definition changes and evidence without fitting or simulation."""
    from nof1_causal_lab.actions.checks import check_specification
    from nof1_causal_lab.models.model_inputs import input_fingerprints
    from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model

    left, before, before_fit, before_simulation = _model_revision(workspace_id, before_id)
    right, after, after_fit, after_simulation = _model_revision(workspace_id, after_id)
    # A study has one question, so its outcome scopes both revisions alike.
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.store import ArtifactStore, read_question

    scoped: tuple[StructuralSelection | None, StructuralSelection | None] = (None, None)
    if left is not None or right is not None:
        question = read_question(
            ArtifactStore(workspace_id), StudyRepository(workspace_id).question().revision
        )
        scoped = (
            StructuralSelection.for_question(left, question) if left is not None else None,
            StructuralSelection.for_question(right, question) if right is not None else None,
        )
    checks = []
    for selection in scoped:
        if selection is None:
            checks.append(())
        else:
            compiled = compile_model(selection)
            checks.append(check_specification(compiled, compile_fit_inputs(compiled, selection)))
    changes = compare_parameters(left, right)
    before_inputs = input_fingerprints(left) if left is not None else {}
    after_inputs = input_fingerprints(right) if right is not None else {}
    constructs, edges = compare_model_graph(*scoped)
    graphs = tuple(
        model_graph_entities(selection) if selection is not None else ((), ())
        for selection in scoped
    )
    from nof1_causal_lab.study.action_arrays import array_references, result_arrays
    from nof1_causal_lab.study.store import ArtifactStore

    arrays = result_arrays(
        ArtifactStore(workspace_id),
        (
            *(array_references(left.model_dump(mode="json")) if left is not None else ()),
            *(array_references(right.model_dump(mode="json")) if right is not None else ()),
        ),
    )
    return ModelDiffOutput(
        arrays=arrays,
        before=before,
        after=after,
        before_model=left,
        after_model=right,
        parameters=tuple(changes),
        constructs=constructs,
        edges=edges,
        before_dispositions=scoped[0].structural_dispositions
        if scoped[0] is not None
        and scoped[0].model.measurement_clock is not None
        and scoped[0].model.indicators
        else (),
        after_dispositions=scoped[1].structural_dispositions
        if scoped[1] is not None
        and scoped[1].model.measurement_clock is not None
        and scoped[1].model.indicators
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
            key
            for key in sorted(before_inputs.keys() | after_inputs.keys())
            if before_inputs.get(key) != after_inputs.get(key)
        ),
    )


def read_model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffOutput:
    """Compute the comparison once during its owning action."""
    return model_diff(workspace_id, before_id, after_id)
