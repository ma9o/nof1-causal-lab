"""Semantic ownership, persistent identity, and exact historical snapshot reads."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.identity import ArtifactId
from nof1_causal_lab.artifacts.latent_structure import Construct, LatentStructure
from nof1_causal_lab.artifacts.structural_plan import StructuralItemDisposition, StructuralPlan
from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.moves import RetractedArtifact, WriteArtifact
from nof1_causal_lab.machine.snapshot_models import ModelSnapshot
from nof1_causal_lab.machine.snapshots import (
    ModelReader,
    SnapshotRevisionNotFound,
    read_model_snapshot,
)
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
from nof1_causal_lab.models.structural import build_structural_plan
from nof1_causal_lab.read_facade import create_read_facade_app


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "MODEL-SNAPSHOT"


def _definitions(snapshot):
    latent = snapshot.latent_structure.value if snapshot.latent_structure else None
    measurement = snapshot.measurement_structure.value if snapshot.measurement_structure else None
    return (
        *(latent.constructs if latent else []),
        *(latent.edges if latent else []),
        *(measurement.measurement_structure.indicators if measurement else []),
    )


def _latent():
    return {
        "default_outcome": {"kind": "construct", "id": "construct:y"},
        "constructs": [
            {
                "id": "construct:x",
                "name": "X",
                "description": "Treatment",
                "role": "exogenous",
                "temporal_status": "time_varying",
            },
            {
                "id": "construct:y",
                "name": "Y",
                "description": "Outcome",
                "role": "endogenous",
                "temporal_status": "time_varying",
            },
        ],
        "edges": [
            {
                "id": "edge:xy",
                "cause_id": "construct:x",
                "effect_id": "construct:y",
                "description": "Treatment effect",
                "lagged": True,
            }
        ],
    }


def _measurement():
    return {
        "measurement_structure": {
            "model_clock": "1d",
            "indicators": [
                {
                    "id": f"indicator:{name.lower()}",
                    "construct_id": f"construct:{name.lower()}",
                    "name": f"{name}_obs",
                    "how_to_measure": f"Observe {name}",
                    "construct_polarity": "positive",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                }
                for name in ("X", "Y")
            ],
        },
        "known_inputs": [],
        "scientific_only_constructs": [],
    }


def _commit(workspace, artifact_id: ArtifactId, payload, *, pins=None, retracted=()):
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
            move=WriteArtifact(artifact_id=artifact_id),
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
    _commit(workspace, "latent_structure", {"latent_structure": _latent()}, pins={"question": 1})
    _commit(workspace, "measurement_structure", _measurement(), pins={"latent_structure": 1})


def test_snapshot_exists_before_any_compilation(workspace):
    empty = read_model_snapshot(workspace)
    assert empty.seq == 0
    assert _definitions(empty) == ()
    assert empty.question is None
    _commit(workspace, "question", {"text": "Does X change Y?"})
    snapshot = read_model_snapshot(workspace)
    assert snapshot.question is not None
    assert snapshot.question.value.text == "Does X change Y?"
    assert snapshot.question.source.artifact.version == 1
    assert _definitions(snapshot) == ()


def test_ownership_and_sources_are_explicit_from_first_structure(workspace):
    _measured(workspace)
    snapshot = read_model_snapshot(workspace)
    assert snapshot.measurement_structure is not None
    assert snapshot.latent_structure is not None
    assert {entity.id for entity in _definitions(snapshot)} == {
        "construct:x",
        "construct:y",
        "edge:xy",
        "indicator:x",
        "indicator:y",
    }
    indicator = snapshot.measurement_structure.value.measurement_structure.indicators[0]
    assert indicator.construct_id == "construct:x"
    assert indicator.construct_id in {
        entity.id for entity in snapshot.latent_structure.value.constructs
    }
    assert snapshot.measurement_structure.source.pointer == ""
    assert snapshot.measurement_structure.source.validity == "fresh"
    assert not snapshot.dispositions
    assert ModelSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_rename_preserves_identity_and_historical_content(workspace):
    _measured(workspace)
    before = read_model_snapshot(workspace)
    renamed = _latent()
    renamed["constructs"][0]["name"] = "Treatment"
    _commit(workspace, "latent_structure", {"latent_structure": renamed}, pins={"question": 1})
    after = read_model_snapshot(workspace)
    assert [entity.id for entity in _definitions(before)] == [
        entity.id for entity in _definitions(after)
    ]
    assert read_model_snapshot(workspace, at_seq=before.seq) == before
    construct = next(entity for entity in _definitions(after) if entity.id == "construct:x")
    assert construct.name == "Treatment"
    indicator = next(entity for entity in _definitions(after) if entity.id == "indicator:x")
    assert indicator.construct_id == construct.id
    assert "construct_name" not in indicator.model_dump()
    assert after.measurement_structure is not None
    assert after.measurement_structure.source.validity == "stale"
    assert after.measurement_structure.source.artifact.version == 1


def test_compilation_preserves_ids_across_name_and_lag_edits():
    measurement = _measurement()
    design = CausalDesign.model_validate(
        {"latent": _latent(), "measurement": measurement["measurement_structure"]}
    )
    original = build_structural_plan(design)
    edited = design.model_dump(mode="json")
    edited["latent"]["constructs"][0]["name"] = "Renamed"
    edited["latent"]["edges"][0]["lagged"] = False
    revised = build_structural_plan(CausalDesign.model_validate(edited))
    assert set(original.semantics.constructs) == set(revised.semantics.constructs)
    assert set(original.semantics.edges) == set(revised.semantics.edges)
    assert original.edges[0].source_id == revised.edges[0].source_id == "edge:xy"


def test_snapshot_serves_dispositions_only_while_the_plan_exists(workspace):
    _measured(workspace)
    design = CausalDesign.model_validate(
        {"latent": _latent(), "measurement": _measurement()["measurement_structure"]}
    )
    plan = build_structural_plan(design)
    _commit(workspace, "structural_plan", {"structural_plan": plan.model_dump(mode="json")})
    planned = read_model_snapshot(workspace)
    assert planned.dispositions is not None
    assert {item.source_id for item in planned.dispositions.value} == {
        item.id for item in _definitions(planned)
    }
    _commit(
        workspace,
        "measurement_structure",
        _measurement(),
        pins={"latent_structure": 1},
        retracted=(RetractedArtifact(artifact_id="structural_plan", reason_ref="test.retracted"),),
    )
    assert not read_model_snapshot(workspace).dispositions
    assert read_model_snapshot(workspace, at_seq=planned.seq) == planned


def test_removed_owner_does_not_leave_an_orphan_stale_indicator(workspace):
    _measured(workspace)
    revised = _latent()
    revised["constructs"][0].update(id="construct:z", name="Z")
    revised["edges"][0].update(id="edge:zy", cause_id="construct:z")
    _commit(workspace, "latent_structure", {"latent_structure": revised}, pins={"question": 1})
    assert {entity.id for entity in _definitions(read_model_snapshot(workspace))} == {
        "construct:z",
        "construct:y",
        "edge:zy",
        "indicator:y",
    }
    assert len(_definitions(read_model_snapshot(workspace, at_seq=3))) == 5


def test_uncommitted_versions_and_failed_attempts_never_become_snapshots(workspace):
    _measured(workspace)
    before = read_model_snapshot(workspace)
    ArtifactStore(workspace).write_version(
        "question",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"question.json": {"text": "Uncommitted"}},
    )
    EpisodeJournal(workspace).append(
        TransitionRecord(
            seq=4,
            ts="2026-09-12T13:00:00Z",
            move=WriteArtifact(artifact_id="question"),
            status="raised",
            trace_ids=[],
            resume=None,
        )
    )
    assert read_model_snapshot(workspace) == before
    with pytest.raises(SnapshotRevisionNotFound):
        read_model_snapshot(workspace, at_seq=4)
    client = TestClient(create_read_facade_app())
    assert client.get(f"/api/episodes/{workspace}/model").json()["seq"] == 3
    assert client.get(f"/api/episodes/{workspace}/model?at_seq=4").status_code == 404
    assert client.get(f"/api/episodes/{workspace}/model?at_seq=-1").status_code == 422
    assert (
        client.get(f"/api/episodes/{workspace}/model?at_seq=0").json()["latent_structure"] is None
    )


@pytest.mark.parametrize("violation", ["owner", "version", "validity", "identity"])
def test_snapshot_rejects_inconsistent_facts(workspace, violation):
    _measured(workspace)
    payload = read_model_snapshot(workspace).model_dump(mode="json")
    indicator = payload["measurement_structure"]["value"]["measurement_structure"]["indicators"][0]
    if violation == "owner":
        indicator["construct_id"] = "construct:missing"
    elif violation == "version":
        payload["measurement_structure"]["source"]["artifact"]["version"] = 2
    elif violation == "validity":
        payload["measurement_structure"]["source"]["validity"] = "stale"
    else:
        indicator["id"] = "indicator:y"
    with pytest.raises(ValidationError):
        ModelSnapshot.model_validate(payload)


def test_authored_ids_are_required_unique_and_match_references():
    payload = _latent()
    del payload["constructs"][0]["id"]
    with pytest.raises(ValidationError, match="id"):
        LatentStructure.model_validate(payload)
    payload = _latent()
    payload["constructs"][1]["id"] = payload["constructs"][0]["id"]
    with pytest.raises(ValidationError, match="Duplicate construct IDs"):
        LatentStructure.model_validate(payload)
    payload = _latent()
    payload["edges"][0]["effect_id"] = "construct:missing"
    with pytest.raises(ValidationError, match="Edge effect"):
        LatentStructure.model_validate(payload)


@pytest.mark.parametrize("violation", ["identity", "endpoint", "owner"])
def test_catalog_preserves_authored_identity_and_references(violation):
    design = CausalDesign.model_validate(
        {"latent": _latent(), "measurement": _measurement()["measurement_structure"]}
    )
    payload = deepcopy(build_structural_plan(design).model_dump(mode="json"))
    if violation == "identity":
        payload["semantics"]["constructs"]["construct:x"]["id"] = "construct:other"
    elif violation == "endpoint":
        payload["semantics"]["edges"]["edge:xy"]["cause_id"] = "construct:missing"
    else:
        payload["semantics"]["indicators"]["indicator:x"]["construct_id"] = "construct:missing"
    with pytest.raises(ValidationError, match="Semantic"):
        StructuralPlan.model_validate(payload)


def test_indicator_findings_follow_pinned_ids_when_names_are_reused(workspace, monkeypatch):
    _measured(workspace)
    measurement = _measurement()
    design = CausalDesign.model_validate(
        {"latent": _latent(), "measurement": measurement["measurement_structure"]}
    )
    _commit(
        workspace,
        "causal_design",
        {"causal_design": design.model_dump(mode="json")},
        pins={"latent_structure": 1, "measurement_structure": 1},
    )
    _commit(
        workspace,
        "structural_plan",
        {"structural_plan": build_structural_plan(design).model_dump(mode="json")},
        pins={"causal_design": 1},
    )
    _commit(
        workspace,
        "statistical_model_spec",
        {
            "statistical_model_spec": {
                "mechanisms": [],
                "likelihoods": [
                    {
                        "indicator_id": "indicator:y",
                        "distribution": "gaussian",
                        "link": "identity",
                        "reasoning": "Test",
                    }
                ],
                "parameters": [],
            },
        },
        pins={"structural_plan": 1},
    )
    before = read_model_snapshot(workspace)
    measurement["measurement_structure"]["indicators"][0]["name"] = "Y_obs"
    measurement["measurement_structure"]["indicators"][1]["name"] = "X_obs"
    _commit(workspace, "measurement_structure", measurement, pins={"latent_structure": 1})

    def reject_catalog_resolution(*args, **kwargs):
        raise AssertionError(
            "Reading a model must not recover scientific ownership from historical catalogs"
        )

    monkeypatch.setattr(ArtifactStore, "read_meta", reject_catalog_resolution)
    after = read_model_snapshot(workspace)
    assert after.measurement_structure is not None
    assert after.specification is not None
    indicator_x = next(
        entity
        for entity in after.measurement_structure.value.measurement_structure.indicators
        if entity.id == "indicator:x"
    )
    indicator_y = next(
        entity
        for entity in after.measurement_structure.value.measurement_structure.indicators
        if entity.id == "indicator:y"
    )
    likelihoods = {
        item.indicator_id: item
        for item in after.specification.value.statistical_model_spec.likelihoods
    }
    assert indicator_x.id not in likelihoods
    assert indicator_y.name == "X_obs"
    assert likelihoods[indicator_y.id] is not None
    assert likelihoods[indicator_y.id].indicator_id == "indicator:y"
    assert after.specification.source.validity == "stale"
    assert read_model_snapshot(workspace, at_seq=before.seq) == before
    assert ModelSnapshot.model_validate_json(after.model_dump_json()) == after


@pytest.mark.parametrize(
    ("mechanism", "owner_field"),
    [
        (
            {
                "kind": "node_potential",
                "target_id": "construct:x",
                "center": {"kind": "fixed", "value": 0},
                "stiffness": {"kind": "fixed", "value": 1},
                "quartic": {"kind": "fixed", "value": 0},
            },
            "target_id",
        ),
        (
            {
                "kind": "hill",
                "edge_id": "edge:xy",
                "emax": {"kind": "fixed", "value": 2},
                "ec50": {"kind": "fixed", "value": 1},
                "n": {"kind": "fixed", "value": 2},
            },
            "edge_id",
        ),
    ],
)
def test_mechanisms_follow_ids_across_renames_and_replacements(workspace, mechanism, owner_field):
    _measured(workspace)
    _commit(
        workspace,
        "statistical_model_spec",
        {
            "statistical_model_spec": {
                "mechanisms": [mechanism],
                "likelihoods": [],
                "parameters": [],
            },
        },
        pins={"measurement_structure": 1},
    )
    before = read_model_snapshot(workspace)
    assert before.specification is not None
    original = before.specification.value.statistical_model_spec.mechanisms
    latent = _latent()
    latent["constructs"][0]["name"] = "Renamed X"
    _commit(workspace, "latent_structure", {"latent_structure": latent}, pins={"question": 1})
    renamed = read_model_snapshot(workspace)
    assert renamed.specification is not None
    assert renamed.specification.value.statistical_model_spec.mechanisms == original
    invalid = renamed.model_dump(mode="json")
    invalid["specification"]["value"]["statistical_model_spec"]["mechanisms"][0][owner_field] = (
        "construct:missing" if owner_field == "target_id" else "edge:missing"
    )
    with pytest.raises(ValidationError, match="Mechanism"):
        ModelSnapshot.model_validate(invalid)

    latent["constructs"][0]["id"] = "construct:replacement"
    latent["edges"][0]["id"] = "edge:replacement"
    latent["edges"][0]["cause_id"] = "construct:replacement"
    _commit(workspace, "latent_structure", {"latent_structure": latent}, pins={"question": 1})
    replaced = read_model_snapshot(workspace)
    assert replaced.specification is not None
    assert replaced.specification.value.statistical_model_spec.mechanisms == []
    assert replaced.specification.source.validity == "stale"
    assert read_model_snapshot(workspace, at_seq=before.seq) == before


@pytest.mark.parametrize("declaration", ["known_inputs", "scientific_only_constructs"])
def test_rename_changes_only_the_construct_definition(declaration):
    measurement = _measurement()
    payload = {
        "latent": _latent(),
        "measurement": measurement["measurement_structure"],
        declaration: [
            {"construct_id": "construct:x", "source_indicator_id": "indicator:x"}
            if declaration == "known_inputs"
            else {"construct_id": "construct:x", "reason": "Scientific context"}
        ],
    }
    original = CausalDesign.model_validate(payload)
    payload["latent"]["constructs"][0]["name"] = "Renamed"
    renamed = CausalDesign.model_validate(payload)
    assert original.measurement == renamed.measurement
    assert original.latent.edges == renamed.latent.edges
    assert original.known_inputs == renamed.known_inputs
    assert original.scientific_only_constructs == renamed.scientific_only_constructs
    assert build_structural_plan(original).state_order == build_structural_plan(renamed).state_order


def test_identification_is_shared_and_follows_ids_across_renames(workspace):
    _measured(workspace)
    finding = {
        "status": "identified",
        "method": "do_calculus",
        "estimand": "P(Y | do(X))",
        "marginalized_confounders": [],
        "instruments": [],
    }
    design = CausalDesign.model_validate(
        {
            "latent": _latent(),
            "measurement": _measurement()["measurement_structure"],
            "identifiability": {"identifiable_treatments": {"construct:x": finding}},
        }
    )
    _commit(
        workspace,
        "causal_design",
        {"causal_design": design.model_dump(mode="json")},
        pins={"latent_structure": 1, "measurement_structure": 1},
    )
    before = read_model_snapshot(workspace)
    assert before.latent_structure is not None
    assert before.identification is not None
    construct = next(
        entity for entity in before.latent_structure.value.constructs if entity.id == "construct:x"
    )
    assert before.identification.value.identifiable_treatments[construct.id] is not None
    assert design.identifiability is not None
    assert (
        before.identification.value.identifiable_treatments[construct.id]
        == design.identifiability.identifiable_treatments["construct:x"]
    )
    assert before.identification.source.pointer == "/causal_design/identifiability"
    renamed = _latent()
    renamed["constructs"][0]["name"] = "Renamed treatment"
    _commit(workspace, "latent_structure", {"latent_structure": renamed})
    renamed_design = design.model_dump(mode="json")
    renamed_design["latent"] = renamed
    _commit(
        workspace,
        "causal_design",
        {"causal_design": renamed_design},
        pins={"latent_structure": 2, "measurement_structure": 1},
    )
    after = read_model_snapshot(workspace)
    assert after.latent_structure is not None
    assert after.identification is not None
    construct = next(
        entity for entity in after.latent_structure.value.constructs if entity.id == "construct:x"
    )
    assert after.identification.value.identifiable_treatments[construct.id] is not None
    assert construct.name == "Renamed treatment"
    assert (
        after.identification.value.identifiable_treatments[construct.id].model_dump(mode="json")
        == finding
    )
    assert after.identification.source.validity == "fresh"
    assert after.identification.source.artifact.version == 2
    assert read_model_snapshot(workspace, at_seq=before.seq) == before


@pytest.mark.parametrize(
    "field", ["treatment", "marginalized_confounders", "instruments", "confounders"]
)
def test_identification_rejects_unresolved_construct_ids(field):
    identified = {"method": "do_calculus", "estimand": "P(Y | do(X))"}
    results = {
        "identifiable_treatments": {"construct:x": identified},
        "non_identifiable_treatments": {},
    }
    if field == "treatment":
        results["identifiable_treatments"] = {"construct:missing": identified}
    elif field == "confounders":
        results["non_identifiable_treatments"] = {
            "construct:y": {"confounders": ["construct:missing"]}
        }
    else:
        identified[field] = ["construct:missing"]
    with pytest.raises(ValidationError, match="unknown construct IDs"):
        CausalDesign.model_validate(
            {
                "latent": _latent(),
                "measurement": _measurement()["measurement_structure"],
                "identifiability": results,
            }
        )


def test_identification_rejects_conflicting_results():
    with pytest.raises(ValidationError, match="both identified and non-identifiable"):
        CausalDesign.model_validate(
            {
                "latent": _latent(),
                "measurement": _measurement()["measurement_structure"],
                "identifiability": {
                    "identifiable_treatments": {
                        "construct:x": {"method": "do_calculus", "estimand": "P(Y | do(X))"}
                    },
                    "non_identifiable_treatments": {"construct:x": {"confounders": []}},
                },
            }
        )


def test_collection_accessors_share_canonical_objects_and_one_revision(workspace):
    _measured(workspace)
    reader = ModelReader(workspace)
    latent = reader.selected("latent_structure")
    assert type(reader.constructs()[0]) is Construct
    assert reader.constructs()[0] is latent.latent_structure.constructs[0]
    latent_read = reader.latent_structure()
    measurement_read = reader.measurement_structure()
    assert latent_read is not None
    assert measurement_read is not None
    assert latent_read.value is latent.latent_structure
    assert measurement_read.value is reader.selected("measurement_structure")
    assert reader.edges()[0] is latent.latent_structure.edges[0]
    renamed = _latent()
    renamed["constructs"][0]["name"] = "Changed after opening the reader"
    _commit(workspace, "latent_structure", {"latent_structure": renamed})
    assert reader.constructs()[0].name == "X"
    assert reader.snapshot().seq == 3
    assert ModelReader(workspace).constructs()[0].name == "Changed after opening the reader"


@pytest.mark.parametrize(
    "accessor",
    [
        "constructs",
        "edges",
        "indicators",
        "parameters",
        "latent_structure",
        "measurement_structure",
        "specification",
        "posterior",
    ],
)
def test_model_accessors_do_not_materialize_other_views(workspace, monkeypatch, accessor):
    _measured(workspace)
    expected = getattr(ModelReader(workspace), accessor)()
    collection = isinstance(expected, tuple)
    payload = (
        [item.model_dump(mode="json") for item in expected]
        if collection
        else expected.model_dump(mode="json")
        if expected
        else None
    )

    def reject_batch(*args, **kwargs):
        raise AssertionError("A collection read must not materialize panel, fit, or view data")

    monkeypatch.setattr(ModelReader, "snapshot", reject_batch)
    client = TestClient(create_read_facade_app())
    path = f"/api/episodes/{workspace}/model/{accessor.replace('_', '-')}"
    response = client.get(path + "?at_seq=3")
    assert response.status_code == 200
    assert response.json() == payload
    assert client.get(path + "?at_seq=99").status_code == 404
    assert client.get(path + "?at_seq=-1").status_code == 422
    assert client.get(path + "?at_seq=0").json() == ([] if collection else None)


@pytest.mark.parametrize("kind", ["construct", "indicator"])
def test_canonical_dispositions_reject_an_incompatible_owner_kind(kind):
    with pytest.raises(ValidationError, match="does not apply to its owner kind"):
        StructuralItemDisposition(
            source_id=f"{kind}:x", source_kind=kind, disposition="retained_edge", reason="Test"
        )


def test_collection_accessor_rejects_a_fresh_orphan_indicator(workspace):
    _commit(workspace, "measurement_structure", _measurement())
    with pytest.raises(ValueError, match="Indicator owner does not exist"):
        ModelReader(workspace).indicators()
