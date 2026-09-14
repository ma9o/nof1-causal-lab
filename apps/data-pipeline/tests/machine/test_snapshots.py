"""History and accessors read one canonical scientific definition with exact sources."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nof1_causal_lab.artifacts.construct import CausalEdge, Construct
from nof1_causal_lab.artifacts.execution import StructuralItemDisposition
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.moves import WriteArtifact
from nof1_causal_lab.machine.snapshot_models import ModelSnapshot
from nof1_causal_lab.machine.snapshots import ModelReader, SnapshotRevisionNotFound
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.read_facade import create_read_facade_app
from tests.helpers import graph_constructs


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
                    "id": f"indicator:{key}",
                    "name": f"{key.upper()}_obs",
                    "how_to_measure": key,
                    "construct_polarity": "positive",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
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
                    "lagged": True,
                }
            ],
        }
    )


def _drop_x(model):
    return model.revised(
        edges=(
            CausalEdge(
                id="edge:yz",
                cause=model.get_construct("construct:y"),
                effect=Construct(
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


def _commit(workspace, artifact_id, payload, *, pins=None, retracted=()):
    journal = EpisodeJournal(workspace)
    info = ArtifactStore(workspace).write_version(
        artifact_id,
        provenance="human",
        derived_from=pins or {},
        produced_by=None,
        json_files={json_filename(artifact_id, artifact_id): payload},
    )
    journal.append(
        TransitionRecord(
            seq=journal.latest_seq() + 1,
            ts="2026-09-12T12:00:00Z",
            move=WriteArtifact(
                artifact_id=artifact_id,
                expected_model_version=info.version - 1 if artifact_id == "model" else None,
            ),
            status="applied",
            produced=[info],
            retracted=list(retracted),
            trace_ids=[],
            resume=None,
        )
    )
    return info


def _measured(workspace):
    _commit(workspace, "question", {"text": "Does X change Y?"})
    _commit(workspace, "model", _model().model_dump(mode="json"))


def _definitions(snapshot):
    if not snapshot.model:
        return ()
    model = _present(snapshot.model).value
    return (*model.constructs, *model.edges, *model.indicators, *model.parameters)


def test_snapshot_exists_before_any_compilation(workspace):
    empty = ModelReader(workspace).snapshot()
    assert empty.context.seq == 0
    assert _definitions(empty) == ()
    assert empty.data.question is None
    _commit(workspace, "question", {"text": "Does X change Y?"})
    snapshot = ModelReader(workspace).snapshot()
    assert _present(snapshot.data.question).value.text == "Does X change Y?"
    assert _present(snapshot.data.question).source.ref.model_dump() == {
        "artifact_id": "question",
        "version": 1,
    }
    assert not _definitions(snapshot)


def test_ownership_and_sources_are_explicit_from_first_structure(workspace):
    _measured(workspace)
    snapshot = ModelReader(workspace).snapshot()
    assert {entity.id for entity in _definitions(snapshot)} == {
        "construct:x",
        "construct:y",
        "edge:xy",
        "indicator:x",
        "indicator:y",
    }
    model = _present(snapshot.model).value
    assert model.indicator("indicator:x") is model.get_construct("construct:x").indicators[0]
    assert _present(snapshot.model).source.pointer == ""
    assert _present(snapshot.model).source.validity == "fresh"
    assert (
        _present(snapshot.findings.dispositions).source.ref == _present(snapshot.model).source.ref
    )
    assert ModelSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_rename_preserves_identity_and_historical_content(workspace):
    _measured(workspace)
    before = ModelReader(workspace).snapshot()
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Treatment"
    _commit(workspace, "model", payload, pins={"model": 1})
    after = ModelReader(workspace).snapshot()
    assert [entity.id for entity in _definitions(before)] == [
        entity.id for entity in _definitions(after)
    ]
    assert ModelReader(workspace, at_seq=before.context.seq).snapshot() == before
    assert _present(after.model).value.get_construct("construct:x").name == "Treatment"
    assert _present(after.model).value.indicator_owner("indicator:x").id == "construct:x"
    assert "construct_id" not in _present(after.model).value.indicator("indicator:x").model_dump()
    assert _present(after.model).source.validity == "fresh"
    assert _present(after.model).source.ref.model_dump() == {"artifact_id": "model", "version": 2}


def test_planning_preserves_ids_across_name_and_lag_edits():
    original = _model()
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Renamed"
    graph_constructs(payload)[0]["role"] = "exogenous"
    payload["edges"][0]["lagged"] = False
    revised = ModelSpec.model_validate(payload)
    assert set(original.state_order) == set(revised.state_order)
    assert original.edges[0].id == revised.edges[0].id == "edge:xy"
    assert "semantics" not in original.model_dump()


def test_snapshot_derives_dispositions_from_its_model_revision(workspace):
    _measured(workspace)
    planned = ModelReader(workspace).snapshot()
    assert {item.source_id for item in _present(planned.findings.dispositions).value} == {
        item.id for item in _definitions(planned)
    }
    _commit(
        workspace,
        "model",
        _model().model_dump(mode="json"),
        pins={"model": 1},
    )
    assert _present(
        ModelReader(workspace).snapshot().findings.dispositions
    ).source.ref.model_dump() == {"artifact_id": "model", "version": 2}
    assert ModelReader(workspace, at_seq=planned.context.seq).snapshot() == planned


def test_removed_construct_removes_its_owned_indicators(workspace):
    _measured(workspace)
    _commit(
        workspace,
        "model",
        _drop_x(_model()).model_dump(mode="json"),
    )
    assert {entity.id for entity in _definitions(ModelReader(workspace).snapshot())} == {
        "construct:y",
        "indicator:y",
        "construct:z",
        "edge:yz",
    }
    assert len(_definitions(ModelReader(workspace, at_seq=2).snapshot())) == 5


def test_uncommitted_versions_and_failed_attempts_never_become_snapshots(workspace):
    _measured(workspace)
    before = ModelReader(workspace).snapshot()
    ArtifactStore(workspace).write_version(
        "question",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"question.json": {"text": "Uncommitted"}},
    )
    EpisodeJournal(workspace).append(
        TransitionRecord(
            seq=3,
            ts="2026-09-12T13:00:00Z",
            move=WriteArtifact(artifact_id="question"),
            status="raised",
            trace_ids=[],
            resume=None,
        )
    )
    assert ModelReader(workspace).snapshot() == before
    with pytest.raises(SnapshotRevisionNotFound):
        ModelReader(workspace, at_seq=3)
    client = TestClient(create_read_facade_app())
    url = f"/api/episodes/{workspace}/model"
    assert client.get(url).json()["context"]["seq"] == 2
    assert client.get(url + "?at_seq=3").status_code == 404
    assert client.get(url + "?at_seq=-1").status_code == 422
    assert client.get(url + "?at_seq=0").json()["model"] is None


@pytest.mark.parametrize("violation", ["owner", "version", "validity", "identity"])
def test_snapshot_rejects_inconsistent_facts(workspace, violation):
    _measured(workspace)
    payload = ModelReader(workspace).snapshot().model_dump(mode="json")
    if violation == "owner":
        payload["model"]["value"]["edges"][0]["effect"] = {
            "kind": "construct",
            "id": "construct:missing",
        }
    elif violation == "version":
        payload["model"]["source"]["ref"]["version"] = 2
    elif violation == "validity":
        payload["model"]["source"]["validity"] = "stale"
    else:
        graph_constructs(payload["model"]["value"])[0]["indicators"][0]["id"] = "indicator:y"
    with pytest.raises(ValidationError):
        ModelSnapshot.model_validate(payload)


def test_owned_likelihood_survives_reused_names(workspace, monkeypatch):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[1]["indicators"][0]["likelihood"] = {
        "law": observation_law("construct:y", "gaussian", "identity").model_dump(mode="json"),
        "reasoning": "Test",
    }
    _commit(workspace, "model", payload)
    before = ModelReader(workspace).snapshot()
    graph_constructs(payload)[0]["indicators"][0]["name"] = "Y_obs"
    graph_constructs(payload)[1]["indicators"][0]["name"] = "X_obs"
    _commit(workspace, "model", payload)

    def reject_catalog(*_args):
        raise AssertionError("Ownership cannot be recovered from an old semantic catalog")

    monkeypatch.setattr(ArtifactStore, "read_meta", reject_catalog)
    after = ModelReader(workspace).snapshot()
    assert _present(after.model).value.indicator("indicator:x").likelihood is None
    assert (
        _present(after.model).value.indicator("indicator:y").likelihood
        == _present(before.model).value.indicator("indicator:y").likelihood
    )
    assert _present(after.model).value.indicator("indicator:y").name == "X_obs"
    assert ModelReader(workspace, at_seq=before.context.seq).snapshot() == before


@pytest.mark.parametrize("owner", ["constructs", "edges"])
def test_owned_mechanisms_survive_rename_and_disappear_with_owner(workspace, owner):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    field = "dynamics" if owner == "constructs" else "mechanisms"
    from nof1_causal_lab.artifacts.coefficient import FixedCoefficient
    from nof1_causal_lab.artifacts.expressions import hill, restoring_force, state
    from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism

    entity = (graph_constructs(payload) if owner == "constructs" else payload["edges"])[0]
    mechanism = DynamicsMechanism(
        id="mechanism:owned-term",
        expression=restoring_force(
            entity["id"],
            center=FixedCoefficient(value=0),
            stiffness=FixedCoefficient(value=1),
            quartic=FixedCoefficient(value=0),
        )
        if owner == "constructs"
        else hill(
            state(entity["cause"]["id"]),
            emax=FixedCoefficient(value=2),
            ec50=FixedCoefficient(value=1),
            n=FixedCoefficient(value=2),
        ),
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
    assert ModelReader(workspace, at_seq=before.context.seq).snapshot() == before


@pytest.mark.parametrize(
    "usage",
    [
        {"kind": "known_input", "source_indicator_id": "indicator:x"},
        {"kind": "scientific_only", "reason": "Scientific context"},
    ],
)
def test_rename_changes_only_construct_label(usage):
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["usage"] = usage
    original = ModelSpec.model_validate(payload)
    graph_constructs(payload)[0]["name"] = "Renamed"
    renamed = ModelSpec.model_validate(payload)
    assert original.indicators == renamed.indicators
    assert original.edges[0].id == renamed.edges[0].id
    assert original.edges[0].effect == renamed.edges[0].effect
    assert original.edges[0].cause.model_copy(update={"name": "Renamed"}) == renamed.edges[0].cause
    assert original.constructs[0].usage == renamed.constructs[0].usage
    assert original.state_order == renamed.state_order


def _identification():
    return {
        "outcome": "construct:y",
        "status": {
            "identifiable_treatments": {
                "construct:x": {"method": "do_calculus", "estimand": "P(Y | do(X))"}
            },
            "non_identifiable_treatments": {},
        },
    }


def test_identification_is_independent_and_keeps_original_pin_after_rename(workspace):
    _measured(workspace)
    _commit(workspace, "identification_report", _identification(), pins={"model": 1})
    before = ModelReader(workspace).snapshot()
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Treatment"
    _commit(workspace, "model", payload, pins={"model": 1})
    after = ModelReader(workspace).snapshot()
    assert (
        _present(after.findings.identification).value
        == _present(before.findings.identification).value
    )
    assert _present(after.findings.identification).source.validity == "stale"
    assert after.context.state.current["identification_report"].derived_from == {"model": 1}
    assert (
        "construct:x"
        in _present(after.findings.identification).value.status.identifiable_treatments
    )


@pytest.mark.parametrize(
    "field", ["outcome", "treatment", "marginalized_confounders", "instruments", "confounders"]
)
def test_identification_rejects_unresolved_construct_ids(field):
    payload = _identification()
    if field == "outcome":
        payload["outcome"] = "construct:missing"
    elif field == "treatment":
        payload["status"]["identifiable_treatments"]["construct:missing"] = {
            "method": "do_calculus",
            "estimand": "test",
        }
    elif field == "confounders":
        payload["status"]["non_identifiable_treatments"]["construct:y"] = {
            "confounders": ["construct:missing"]
        }
    else:
        payload["status"]["identifiable_treatments"]["construct:x"][field] = ["construct:missing"]
    with pytest.raises(ValueError, match="unknown construct IDs"):
        IdentificationReport.model_validate(payload).validate_model(_model())


def test_identification_rejects_conflicting_results():
    payload = _identification()
    payload["status"]["non_identifiable_treatments"]["construct:x"] = {"confounders": []}
    with pytest.raises(ValidationError, match="both identified and non-identifiable"):
        IdentificationReport.model_validate(payload)


def test_collection_accessors_share_canonical_objects_and_one_revision(workspace):
    _measured(workspace)
    reader = ModelReader(workspace)
    assert type(reader.constructs()[0]) is Construct
    assert reader.constructs()[0] is _present(reader.model).constructs[0]
    assert reader.edges()[0] is _present(reader.model).edges[0]
    assert reader.indicators()[0] is _present(reader.model).constructs[0].indicators[0]
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Changed after opening"
    _commit(workspace, "model", payload)
    assert reader.constructs()[0].name == "X"
    assert reader.snapshot().context.seq == 2
    assert ModelReader(workspace).constructs()[0].name == "Changed after opening"


@pytest.mark.parametrize(
    "accessor", ["constructs", "edges", "indicators", "parameters", "inference_report"]
)
def test_accessors_do_not_materialize_other_views(workspace, monkeypatch, accessor):
    _measured(workspace)
    expected = getattr(ModelReader(workspace), accessor)()
    collection = isinstance(expected, tuple)
    payload = [item.model_dump(mode="json") for item in expected] if collection else None

    def reject_batch(*_args):
        raise AssertionError("An accessor must not load unrelated views")

    monkeypatch.setattr(ModelReader, "snapshot", reject_batch)
    client = TestClient(create_read_facade_app())
    path = f"/api/episodes/{workspace}/model/{accessor.replace('_', '-')}"
    response = client.get(path + "?at_seq=2")
    assert response.status_code == 200
    assert response.json() == payload
    assert client.get(path + "?at_seq=99").status_code == 404
    assert client.get(path + "?at_seq=-1").status_code == 422
    assert client.get(path + "?at_seq=0").json() == ([] if collection else None)


@pytest.mark.parametrize("kind", ["construct", "indicator"])
def test_dispositions_reject_incompatible_owner_kind(kind):
    with pytest.raises(ValidationError, match="does not apply to its owner kind"):
        StructuralItemDisposition(
            source_id=f"{kind}:x", source_kind=kind, disposition="retained_edge", reason="Test"
        )
