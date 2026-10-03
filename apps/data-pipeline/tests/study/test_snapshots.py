"""History and accessors read one canonical scientific definition with exact sources."""

import time
from datetime import UTC, date, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.availability import NotApplicable
from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec, replace_constructs
from nof1_causal_lab.artifacts.execution import StructuralItemDisposition
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import ConstructId, GitRef, IndicatorId
from nof1_causal_lab.artifacts.likelihood import (
    ObservationLawSpec,
)
from nof1_causal_lab.artifacts.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec
from nof1_causal_lab.models.model_structure import StructuralSelection, selected_state_ids
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from nof1_causal_lab.numpyro_json import empirical_atoms, empirical_distribution
from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    Applied,
    AttemptRecord,
    EditAttempt,
    ModelSimulationResult,
    Raised,
)
from nof1_causal_lab.study.snapshot_models import ModelSnapshot
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record, question_root
from tests.git_fixtures import artifact_revision, commit_id, git_oid
from tests.helpers import graph_constructs, write_question
from tests.model_fixtures import compile_model_fixture

pytestmark = pytest.mark.contract


def _present[T](value: T | None) -> T:
    assert value is not None
    return value


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "SNAPSHOT"


def _model():
    constructs = {
        key: {
            "id": f"construct:{key}",
            "name": key.upper(),
            "description": key,
            "role": "endogenous",
            "temporal_status": "time_varying",
            "indicators": [
                {
                    "observation": {
                        "id": f"indicator:{key}",
                        "name": f"{key.upper()}_obs",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    "construct_polarity": "positive",
                }
            ],
        }
        for key in ("x", "y")
    }
    return ModelSpec.model_validate(
        {
            "measurement_clock": "1d",
            "edges": [
                {
                    "id": "edge:xy",
                    "cause": constructs["x"],
                    "effect": constructs["y"],
                    "description": "effect",
                }
            ],
        }
    )


def _drop_x(model):
    return model.revised(
        edges=(
            CausalEdgeSpec(
                id="edge:yz",
                cause=model.get_construct("construct:y"),
                effect=ConstructSpec(
                    id="construct:z",
                    name="Z",
                    description="Downstream response",
                    role="endogenous",
                    temporal_status="time_varying",
                ),
                description="Y affects Z",
            ),
        )
    )


def test_predictive_findings_use_entity_ids_and_keep_served_reasons(monkeypatch):
    from nof1_causal_lab.artifacts.checks import NumericCriterionEvidence
    from nof1_causal_lab.models.ssm.predictive import simulation
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from nof1_causal_lab.models.ssm.reachability import CheckResult
    from nof1_causal_lab.models.ssm.simulation_checks import DesignInfo

    model = ModelSpec.model_validate_json(
        (Path(__file__).resolve().parents[1] / "fixtures/models/common/x_y_model.json").read_text()
    )
    edge = model.edges[0]

    def measured(_model, _prediction, _design, target, **_kwargs):
        if target.construct.id != edge.effect.id:
            return [], []
        return [
            CheckResult.measured(
                "measured",
                target,
                "1",
                "0",
                "Served reason.",
                outcome="failed",
                measurements=(NumericCriterionEvidence(criterion="measured", value=1, upper=0),),
            )
            for target in (
                edge.effect.name,
                edge.effect.indicators[0].observation.id,
                f"{edge.cause.name}->{edge.effect.name}",
            )
        ], []

    monkeypatch.setattr(simulation, "measure_construct_simulation", measured)
    paths = jnp.zeros((1, 2, 2))
    prediction = PredictiveDraws(
        {}, PredictiveTrajectory(paths, paths, paths, jnp.ones_like(paths, dtype=bool), paths)
    )
    batch = simulation.SimulationBatch(
        (0.0, 1.0), prediction, None, DesignInfo(jnp.array([0.0, 1.0]), (), {}, {})
    )
    findings, _ = simulation.measure_simulation_batch(
        compile_model_fixture(model), batch, groups=("measurement",), clock=time.monotonic
    )
    targets = [finding.subject.target for finding in findings]
    assert all(not isinstance(target, str) for target in targets)
    assert [target.id for target in targets if not isinstance(target, str)] == [
        edge.effect.id,
        edge.effect.indicators[0].observation.id,
        edge.id,
    ]
    for finding in findings:
        assert finding.kind == "evaluated"
        assert finding.evidence[0].note == "Served reason."


def _commit(workspace, artifact_id, payload, *, pins=None, retracted=()):
    journal = StudyRepository(workspace)
    store = ArtifactStore(workspace)
    info = store.write_artifact(
        artifact_id,
        derived_from=pins or {},
        produced_by=None,
        json_files={json_filename(artifact_id, artifact_id): payload},
    )
    # Every lineage carries its question; the first commit adds it alongside.
    rooted = journal.state(journal.head()).has("question")
    journal.append(
        applied_record(
            Applied(
                result=None,
                effects=ActionEffects(
                    produced=[info] if rooted else [write_question(store), info],
                    retracted=list(retracted),
                ),
            ),
            seq=journal.latest_seq() + 1,
            ts="2026-09-12T12:00:00Z",
            trace_ids=[],
        )
    )
    return info


def _measured(workspace):
    _commit(workspace, "model", _model().model_dump(mode="json"))


def _identity_owners(snapshot):
    if not snapshot.model:
        return ()
    model = _present(snapshot.model).value
    return (
        *model.constructs,
        *model.edges,
        *(item.observation for item in model.indicators),
        *model.parameters,
    )


def test_fitted_snapshot_keeps_joint_arrays_lazy_and_workspace_bound(workspace, monkeypatch):
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/models"
            / "snapshots/fitted_snapshot_keeps_joint_arrays_lazy_and_workspace_bound_complete_test_model.json"
        ).read_text()
    )
    bindings, _ = parameter_bindings(compile_model_fixture(model))
    layout = JointLawLayout.from_bindings(
        bindings,
        parameters=[p.id for p in model.parameters],
        constructs=selected_state_ids(StructuralSelection(model, None)),
        time_points=(0, 1),
    )
    store = ArtifactStore(workspace)
    atoms = np.ones((2, layout.width))
    law = empirical_distribution(atoms, array_writer=store.write_array)
    conditioned = model.revised(
        parameters=tuple(
            p.revised(distribution=layout.distribution_id, transform={"kind": "identity"})
            for p in model.parameters
        ),
        edges=replace_constructs(
            model.edges,
            tuple(c.revised(distribution=layout.distribution_id) for c in model.constructs),
        ),
        distributions={layout.distribution_id: law},
        time_points=(0, 1),
    )
    _commit(workspace, "model", conditioned.model_dump(mode="json"))
    read_array = ArtifactStore.read_array
    reads = []

    def tracked_read(self, identity):
        reads.append(identity)
        return read_array(self, identity)

    monkeypatch.setattr(ArtifactStore, "read_array", tracked_read)
    reader = ModelReader(workspace)
    snapshot = reader.snapshot()
    assert snapshot.authoring_prior_densities == {}
    assert _present(snapshot.model).value.model_dump(mode="json") == conditioned.model_dump(
        mode="json"
    )
    client = TestClient(create_read_facade_app())
    for path in ("", "/definition", "/parameters"):
        response = client.get(f"/api/studies/{workspace}/model{path}")
        assert response.status_code == 200, response.text
    assert reads == []
    # A consumer that needs the draws can still resolve them in this workspace.
    restored = _present(reader.model).distributions[layout.distribution_id]
    np.testing.assert_array_equal(empirical_atoms(restored), atoms)
    assert len(reads) == 2  # the draws and their weights, each read once


def test_checkpoint_comparison_uses_evidence_from_each_selected_journal_prefix(workspace):
    _measured(workspace)
    for seq in (2, 3):
        report = SimulationReport(
            causal=NotApplicable(reason="No intervention was requested."),
            model=GitRef(
                workspace_id=workspace,
                revision=artifact_revision(workspace, "model", 1),
                path="model.json",
            ),
            design=SimulationSpec(start=date(2026, 1, 1), horizon="1d"),
            time_origin=datetime(2026, 1, 1, tzinfo=UTC),
            assignments=(),
            fit_reliability="not_fitted",
            times=(0, 1),
            draws=1,
            seed=seq,
            state_ids=(),
            observation_layout={
                "variables": [
                    {
                        "id": "indicator:y",
                        "name": "y",
                        "measurement_dtype": "continuous",
                        "aggregation": "last",
                        "observation_window": "1d",
                    }
                ],
                "support_start_times": "starts",
                "support_end_times": "ends",
                "mask": "mask",
            },
            parameter_draws={},
            latent_paths=f"simulation-{seq}/latent",
            observations=f"simulation-{seq}/observations",
        )
        StudyRepository(workspace).append(
            applied_record(
                Applied(result=ModelSimulationResult(report=report), effects=ActionEffects()),
                seq=seq,
                ts="2026-09-12T12:00:00Z",
                trace_ids=[],
            )
        )
    client = TestClient(create_read_facade_app())
    response = client.get(
        f"/api/studies/{workspace}/model-diff",
        params={"before": commit_id(workspace, 1), "after": commit_id(workspace, 2)},
    )
    assert response.status_code == 200
    comparison = response.json()
    assert comparison["before"]["revision"] == commit_id(workspace, 1)
    assert comparison["after"]["revision"] == commit_id(workspace, 2)
    assert comparison["before_simulation"] is None
    assert comparison["after_simulation"]["seed"] == 2
    response = client.get(
        f"/api/studies/{workspace}/model-diff",
        params={"before": commit_id(workspace, 3), "after": commit_id(workspace, 2)},
    )
    assert response.status_code == 200
    comparison = response.json()
    assert comparison["before_simulation"]["seed"] == 3
    assert comparison["after_simulation"]["seed"] == 2
    assert all(item["kind"] == "unchanged" for item in comparison["constructs"])
    assert (
        client.get(
            f"/api/studies/{workspace}/model-diff",
            params={"before": git_oid(4), "after": commit_id(workspace, 2)},
        ).status_code
        == 404
    )

    # Exact model artifacts are also valid comparison inputs; no run is inferred.
    model_revision = artifact_revision(workspace, "model", 1)
    response = client.get(
        f"/api/studies/{workspace}/model-diff",
        params={"before": model_revision, "after": commit_id(workspace, 2)},
    )
    assert response.status_code == 200, response.text
    comparison = response.json()
    assert comparison["before"]["path"] == "model.json"
    assert comparison["before_simulation"] is None
    assert comparison["after_simulation"]["seed"] == 2


def test_snapshot_exists_before_any_compilation(workspace):
    empty = ModelReader(workspace).snapshot()
    assert empty.selected_seq == 0
    assert _identity_owners(empty) == ()
    assert empty.model is None
    assert empty.question is None
    question_root(workspace, QuestionSpec(text="Does X change Y?"))
    _commit(workspace, "model", {"measurement_clock": "1d"})
    snapshot = ModelReader(workspace).snapshot()
    assert _present(snapshot.question).value.text == "Does X change Y?"
    assert _present(snapshot.model).value == ModelSpec(measurement_clock="1d")
    assert _present(snapshot.model).source.ref.model_dump() == {
        "workspace_id": workspace,
        "revision": artifact_revision(workspace, "model", 1),
        "path": "model.json",
    }
    assert not _identity_owners(snapshot)


def test_ownership_and_sources_are_explicit_from_first_structure(workspace):
    _measured(workspace)
    snapshot = ModelReader(workspace).snapshot()
    assert {entity.id for entity in _identity_owners(snapshot)} == {
        "construct:x",
        "construct:y",
        "edge:xy",
        "indicator:x",
        "indicator:y",
    }
    model = _present(snapshot.model).value
    assert (
        model.indicator(IndicatorId("indicator:x"))
        is model.get_construct(ConstructId("construct:x")).indicators[0]
    )
    assert _present(snapshot.model).source.pointer == ""
    assert _present(snapshot.model).source.validity == "fresh"
    assert _present(snapshot.dispositions).source.ref == _present(snapshot.model).source.ref
    assert ModelSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_rename_preserves_identity_and_historical_content(workspace):
    _measured(workspace)
    before = ModelReader(workspace).snapshot()
    assert before.workspace_id == workspace
    assert _present(before.model).source.ref == GitRef(
        workspace_id=workspace, revision=artifact_revision(workspace, "model", 1), path="model.json"
    )
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Treatment"
    _commit(workspace, "model", payload, pins={"model": artifact_revision(workspace, "model", 1)})
    after = ModelReader(workspace).snapshot()
    assert [entity.id for entity in _identity_owners(before)] == [
        entity.id for entity in _identity_owners(after)
    ]
    assert ModelReader(workspace, at=before.commit_id).snapshot() == before
    assert _present(after.model).value.get_construct(ConstructId("construct:x")).name == "Treatment"
    assert (
        _present(after.model).value.indicator_owner(IndicatorId("indicator:x")).id == "construct:x"
    )
    assert (
        "construct_id"
        not in _present(after.model).value.indicator(IndicatorId("indicator:x")).model_dump()
    )
    assert _present(after.model).source.validity == "fresh"
    assert _present(after.model).source.ref.model_dump() == {
        "workspace_id": workspace,
        "revision": artifact_revision(workspace, "model", 2),
        "path": "model.json",
    }


def test_planning_preserves_ids_across_name_and_role_edits(workspace):
    original = _model()
    payload = _model().model_dump(mode="json")
    given = graph_constructs(payload)[0]
    given.update(name="Renamed", role="exogenous", coefficients=[], dynamics=[], distribution=None)
    for indicator in given["indicators"]:
        indicator["likelihood"] = {
            "law": {
                "distribution": "Delta",
                "v": {"kind": "state", "construct_id": given["id"]},
            },
            "standardized": False,
            "reasoning": "The renamed input is given exactly.",
            "sources": [],
        }
    revised = ModelSpec.model_validate(payload)
    assert set(selected_state_ids(StructuralSelection(original, None))) == set(
        selected_state_ids(StructuralSelection(revised, None))
    )
    assert original.edges[0].id == revised.edges[0].id == "edge:xy"
    assert "semantics" not in original.model_dump()
    _commit(workspace, "model", revised.model_dump(mode="json"))
    graph = ModelReader(workspace).snapshot().graph
    assert set(graph.dynamic_construct_ids) == set(
        selected_state_ids(StructuralSelection(revised, None))
    )
    from nof1_causal_lab.utils.identifiability import unroll_temporal_dag

    dag = unroll_temporal_dag(revised.constructs, revised.edges, {"Renamed", "Y"})
    for construct_id in graph.dynamic_construct_ids:
        name = revised.get_construct(construct_id).name
        assert dag.has_edge(f"{name}_{{t-1}}", f"{name}_t")
    assert dag.has_edge("Renamed_{t-1}", "Y_t")


def test_snapshot_derives_dispositions_from_its_model_revision(workspace):
    _measured(workspace)
    planned = ModelReader(workspace).snapshot()
    assert {item.target.id for item in _present(planned.dispositions).value} == {
        item.id for item in _identity_owners(planned)
    }
    _commit(
        workspace,
        "model",
        _model().model_dump(mode="json"),
        pins={"model": artifact_revision(workspace, "model", 1)},
    )
    assert _present(ModelReader(workspace).snapshot().dispositions).source.ref.model_dump() == {
        "workspace_id": workspace,
        "revision": artifact_revision(workspace, "model", 2),
        "path": "model.json",
    }
    assert ModelReader(workspace, at=planned.commit_id).snapshot() == planned


def test_removed_construct_removes_its_owned_indicators(workspace):
    _measured(workspace)
    _commit(
        workspace,
        "model",
        _drop_x(_model()).model_dump(mode="json"),
    )
    assert {entity.id for entity in _identity_owners(ModelReader(workspace).snapshot())} == {
        "construct:y",
        "indicator:y",
        "construct:z",
        "edge:yz",
    }
    assert len(_identity_owners(ModelReader(workspace, at=commit_id(workspace, 1)).snapshot())) == 5


def test_uncommitted_versions_and_failed_attempts_never_become_snapshots(workspace):
    _measured(workspace)
    before = ModelReader(workspace).snapshot()
    ArtifactStore(workspace).write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": {"measurement_clock": "2d"}},
    )
    StudyRepository(workspace).append(
        AttemptRecord(
            seq=3,
            ts="2026-09-12T13:00:00Z",
            trace_ids=[],
            attempt=EditAttempt(
                action="edit_model",
                request=None,
                outcome=Raised(error_type="SavedError", error_message="failed"),
            ),
        )
    )
    assert ModelReader(workspace).snapshot() == before
    with pytest.raises(StudyLookupError):
        ModelReader(workspace, at=commit_id(workspace, 3))
    client = TestClient(create_read_facade_app())
    url = f"/api/studies/{workspace}/model"
    assert client.get(url).json()["selected_seq"] == 1
    assert client.get(url + f"?at={commit_id(workspace, 3)}").status_code == 404
    assert client.get(url + "?at=-1").status_code == 422
    assert client.get(url + f"?at={commit_id(workspace, 0)}").json()["model"] is None


@pytest.mark.parametrize("violation", ["owner", "revision", "validity", "identity"])
def test_snapshot_rejects_inconsistent_facts(workspace, violation):
    _measured(workspace)
    payload = ModelReader(workspace).snapshot().model_dump(mode="json")
    if violation == "owner":
        payload["model"]["value"]["edges"][0]["effect"] = {
            "kind": "construct",
            "id": "construct:missing",
        }
    elif violation == "revision":
        payload["model"]["source"]["ref"]["revision"] = git_oid(2)
    elif violation == "validity":
        payload["model"]["source"]["validity"] = "stale"
    else:
        graph_constructs(payload["model"]["value"])[0]["indicators"][0]["observation"]["id"] = (
            "indicator:y"
        )
    with pytest.raises(ValidationError):
        ModelSnapshot.model_validate(payload)


def test_owned_likelihood_survives_reused_names(workspace, monkeypatch):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[1]["indicators"][0]["likelihood"] = {
        "law": TypeAdapter(ObservationLawSpec)
        .validate_json(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures/models"
                / "common/y_gaussian_observation_law.json"
            ).read_text()
        )
        .model_dump(mode="json"),
        "reasoning": "Test",
    }
    _commit(workspace, "model", payload)
    before = ModelReader(workspace).snapshot()
    graph_constructs(payload)[0]["indicators"][0]["observation"]["name"] = "Y_obs"
    graph_constructs(payload)[1]["indicators"][0]["observation"]["name"] = "X_obs"
    _commit(workspace, "model", payload)

    def reject_catalog(*_args):
        raise AssertionError("Ownership cannot be recovered from an old semantic catalog")

    monkeypatch.setattr(ArtifactStore, "list_revisions", reject_catalog)
    after = ModelReader(workspace).snapshot()
    assert _present(after.model).value.indicator(IndicatorId("indicator:x")).likelihood is None
    assert (
        _present(after.model).value.indicator(IndicatorId("indicator:y")).likelihood
        == _present(before.model).value.indicator(IndicatorId("indicator:y")).likelihood
    )
    assert (
        _present(after.model).value.indicator(IndicatorId("indicator:y")).observation.name
        == "X_obs"
    )
    assert ModelReader(workspace, at=before.commit_id).snapshot() == before


@pytest.mark.parametrize("owner", ["constructs", "edges"])
def test_owned_mechanisms_survive_rename_and_disappear_with_owner(workspace, owner):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    field = "dynamics" if owner == "constructs" else "mechanisms"
    from nof1_causal_lab.artifacts.expressions import hill, restoring_force, state

    entity = (graph_constructs(payload) if owner == "constructs" else payload["edges"])[0]
    mechanism = DriftMechanismSpec(
        id="mechanism:owned-term",
        expression=restoring_force(entity["id"], center=0, stiffness=1, quartic=0)
        if owner == "constructs"
        else hill(state(entity["cause"]["id"]), emax=2, ec50=1, n=2),
    ).model_dump(mode="json")
    entity[field] = [mechanism]
    _commit(workspace, "model", payload)
    before = ModelReader(workspace).snapshot()
    graph_constructs(payload)[0]["name"] = "Renamed X"
    _commit(workspace, "model", payload)
    after = ModelReader(workspace).snapshot()
    encoded = _present(after.model).value.model_dump(mode="json")
    assert (graph_constructs(encoded) if owner == "constructs" else encoded["edges"])[0][field] == [
        mechanism
    ]
    payload = _drop_x(ModelSpec.model_validate(payload)).model_dump(mode="json")
    _commit(workspace, "model", payload)
    assert list(_present(ModelReader(workspace).model).iter_mechanisms()) == []
    assert ModelReader(workspace, at=before.commit_id).snapshot() == before


def test_rename_changes_only_construct_label():
    payload = _model().model_dump(mode="json")
    original = ModelSpec.model_validate(payload)
    graph_constructs(payload)[0]["name"] = "Renamed"
    renamed = ModelSpec.model_validate(payload)
    assert original.indicators == renamed.indicators
    assert original.edges[0].id == renamed.edges[0].id
    assert original.edges[0].effect == renamed.edges[0].effect
    assert original.edges[0].cause.revised(name="Renamed") == renamed.edges[0].cause
    assert selected_state_ids(StructuralSelection(original, None)) == selected_state_ids(
        StructuralSelection(renamed, None)
    )


def _identification():
    return {
        "outcome": "construct:y",
        "treatments": {
            "construct:x": {
                "status": "identified",
                "method": "do_calculus",
                "estimand": "P(Y | do(X))",
            },
        },
    }


def test_identification_is_independent_and_keeps_original_pin_after_rename(workspace):
    _measured(workspace)
    _commit(
        workspace,
        "identification_report",
        _identification(),
        pins={"model": artifact_revision(workspace, "model", 1)},
    )
    before = ModelReader(workspace).snapshot()
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Treatment"
    _commit(workspace, "model", payload, pins={"model": artifact_revision(workspace, "model", 1)})
    after = ModelReader(workspace).snapshot()
    assert _present(after.identification).value == _present(before.identification).value
    assert _present(after.identification).source.validity == "stale"
    assert after.state.current["identification_report"].derived_from == {
        "model": artifact_revision(workspace, "model", 1)
    }
    assert "construct:x" in _present(after.identification).value.estimable_treatments


@pytest.mark.parametrize(
    "field", ["outcome", "treatment", "marginalized_confounders", "instruments", "confounders"]
)
def test_identification_rejects_unresolved_construct_ids(field):
    payload = _identification()
    if field == "outcome":
        payload["outcome"] = "construct:missing"
    elif field == "treatment":
        payload["treatments"]["construct:missing"] = {
            "status": "identified",
            "method": "do_calculus",
            "estimand": "test",
        }
    elif field == "confounders":
        payload["treatments"]["construct:y"] = {
            "status": "not_identified",
            "confounders": ["construct:missing"],
        }
    else:
        payload["treatments"]["construct:x"][field] = ["construct:missing"]
    with pytest.raises(ValueError, match="unknown construct IDs"):
        IdentificationReport.model_validate(payload).validate_model(_model())


def test_identification_round_trip_preserves_tagged_evidence():
    payload = _identification()
    payload["treatments"]["construct:y"] = {
        "status": "not_identified",
        "confounders": ["construct:x"],
        "notes": "Blocking confounding",
    }
    report = IdentificationReport.model_validate(payload)
    restored = IdentificationReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.estimable_treatments == ("construct:x",)
    identified = restored.treatments[ConstructId("construct:x")]
    assert identified.status == "identified"
    assert identified.estimand == "P(Y | do(X))"
    assert restored.non_identifiable[ConstructId("construct:y")].confounders == ("construct:x",)
    assert restored.non_identifiable[ConstructId("construct:y")].notes == "Blocking confounding"

    payload["treatments"]["construct:y"]["estimand"] = "Contradictory positive evidence"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        IdentificationReport.model_validate(payload)


def test_collection_accessors_share_canonical_objects_and_one_revision(workspace):
    _measured(workspace)
    reader = ModelReader(workspace)
    assert type(reader.constructs()[0]) is ConstructSpec
    assert reader.constructs()[0] is _present(reader.model).constructs[0]
    assert reader.edges()[0] is _present(reader.model).edges[0]
    assert reader.indicators()[0] is _present(reader.model).constructs[0].indicators[0]
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Changed after opening"
    _commit(workspace, "model", payload)
    assert reader.constructs()[0].name == "X"
    assert reader.snapshot().selected_seq == 1
    assert ModelReader(workspace).constructs()[0].name == "Changed after opening"


@pytest.mark.parametrize(
    "accessor", ["constructs", "edges", "indicators", "parameters", "inference_report"]
)
def test_accessors_do_not_materialize_other_views(workspace, monkeypatch, accessor):
    _measured(workspace)
    reader = ModelReader(workspace)
    expected = (
        reader.inference_report if accessor == "inference_report" else getattr(reader, accessor)()
    )
    collection = isinstance(expected, tuple)
    payload = [item.model_dump(mode="json") for item in expected] if collection else None

    def reject_batch(*_args):
        raise AssertionError("An accessor must not load unrelated views")

    monkeypatch.setattr(ModelReader, "snapshot", reject_batch)
    client = TestClient(create_read_facade_app())
    path = f"/api/studies/{workspace}/model/{accessor.replace('_', '-')}"
    response = client.get(path + f"?at={commit_id(workspace, 1)}")
    assert response.status_code == 200
    assert response.json() == payload
    assert client.get(path + f"?at={git_oid(99)}").status_code == 404
    assert client.get(path + "?at=-1").status_code == 422
    assert client.get(path + f"?at={commit_id(workspace, 0)}").json() == (
        [] if collection else None
    )


@pytest.mark.parametrize(
    ("target", "disposition", "error"),
    [
        ({"kind": "construct", "id": "construct:x"}, "retained_edge", "does not apply"),
        ({"kind": "indicator", "id": "indicator:x"}, "retained_edge", "does not apply"),
        ({"kind": "construct", "id": "indicator:x"}, "retained_state", "pattern"),
        ({"kind": "mechanism", "id": "mechanism:x"}, "retained_state", "union_tag_invalid"),
    ],
)
def test_dispositions_validate_target_identity_and_compatible_decision(target, disposition, error):
    with pytest.raises(ValidationError, match=error):
        StructuralItemDisposition.model_validate(
            {"target": target, "disposition": disposition, "reason": "Test"}
        )
