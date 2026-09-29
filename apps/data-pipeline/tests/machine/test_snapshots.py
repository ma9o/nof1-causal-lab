"""History and accessors read one canonical scientific definition with exact sources."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec, replace_constructs
from nof1_causal_lab.artifacts.execution import StructuralItemDisposition
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import ConstructId, GitRef, IndicatorId
from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.predictive_provenance import PredictiveLawProvenance
from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec
from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.snapshot_models import ModelSnapshot
from nof1_causal_lab.machine.snapshots import ModelReader, SnapshotRevisionNotFound
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from nof1_causal_lab.numpyro_json import empirical_atoms, empirical_distribution
from nof1_causal_lab.read_facade import create_read_facade_app
from tests.git_fixtures import artifact_revision, commit_id, git_oid
from tests.helpers import complete_test_model, graph_constructs

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
                    "id": f"indicator:{key}",
                    "name": f"{key.upper()}_obs",
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
            "question": "Does X change Y?",
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


def _commit(workspace, artifact_id, payload, *, pins=None, retracted=()):
    journal = StudyRepository(workspace)
    info = ArtifactStore(workspace).write_artifact(
        artifact_id,
        derived_from=pins or {},
        produced_by=None,
        json_files={json_filename(artifact_id, artifact_id): payload},
    )
    journal.append(
        TransitionRecord(
            seq=journal.latest_seq() + 1,
            ts="2026-09-12T12:00:00Z",
            action="edit_model",
            inputs={
                "expected_revision": journal.state(journal.head()).current["model"].revision
                if journal.state(journal.head()).has("model")
                else None
            },
            status="applied",
            produced=[info],
            retracted=list(retracted),
            trace_ids=[],
            resume=None,
        )
    )
    return info


def _measured(workspace):
    _commit(workspace, "model", _model().model_dump(mode="json"))


def _definitions(snapshot):
    if not snapshot.model:
        return ()
    model = _present(snapshot.model).value
    return (*model.constructs, *model.edges, *model.indicators, *model.parameters)


def test_fitted_snapshot_keeps_joint_arrays_lazy_and_workspace_bound(workspace, monkeypatch):
    model = complete_test_model(_model())
    bindings, _ = parameter_bindings(model)
    layout = JointLawLayout.from_bindings(
        bindings,
        parameters=[p.id for p in model.parameters if p.value is None],
        constructs=model.state_order,
        time_points=(0, 1),
    )
    store = ArtifactStore(workspace)
    atoms = np.ones((2, layout.width))
    law = empirical_distribution(atoms, array_writer=store.write_array)
    conditioned = model.revised(
        parameters=tuple(
            p.model_copy(
                update={
                    "distribution": layout.distribution_id,
                    "distribution_transform": "identity",
                    "reference_interval_days": None,
                }
            )
            if p.value is None
            else p
            for p in model.parameters
        ),
        edges=replace_constructs(
            model.edges,
            tuple(
                c.model_copy(update={"distribution": layout.distribution_id})
                for c in model.constructs
            ),
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
    assert _present(snapshot.findings.diagnostics).prior_densities == {}
    assert _present(snapshot.model).value.model_dump(mode="json") == conditioned.model_dump(
        mode="json"
    )
    client = TestClient(create_read_facade_app())
    for path in ("", "/definition", "/views/model_diagnostics", "/parameters"):
        response = client.get(f"/api/episodes/{workspace}/model{path}")
        assert response.status_code == 200, response.text
    assert reads == []
    # A consumer that needs the draws can still resolve them in this workspace.
    restored = _present(reader.model).distributions[layout.distribution_id]
    np.testing.assert_array_equal(empirical_atoms(restored), atoms)
    assert len(reads) == 2  # the draws and their weights, each read once


@pytest.mark.parametrize("paired", [False, True])
def test_simulation_bands_read_saved_draws_at_selected_version(workspace, monkeypatch, paired):
    model = _model().revised(default_outcome=ConstructId("construct:y"))
    info = _commit(workspace, "model", model.model_dump(mode="json"))
    store = ArtifactStore(workspace)
    values = np.array([[[90, 1], [91, 2]], [[92, 5], [93, 10]]], dtype=float)
    observations = values + 100
    mask = np.ones_like(values, dtype=bool)
    mask[:, 0, 1] = False
    observations[:, 0, 1] = np.nan
    report = SimulationReport(
        model=GitRef(workspace_id=workspace, revision=info.revision, path="model.json"),
        design=SimulationSpec(
            start=0,
            end=1,
            interventions=[{"target": "construct:x", "time": 0, "value": 0}] if paired else [],
        ),
        times=(0, 1),
        draws=2,
        seed=1,
        state_ids=tuple(construct.id for construct in model.constructs),
        observation_layout={
            "variables": [
                {
                    "id": indicator.id,
                    "name": indicator.name,
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "observation_window": "1d",
                }
                for indicator in model.indicators
            ],
            "support_start_times": "unused",
            "support_end_times": "unused",
            "mask": store.write_array(mask),
        },
        parameter_draws={},
        latent_paths=store.write_array(values),
        observations=store.write_array(observations),
        reference_latent_paths=store.write_array(values - 1) if paired else None,
        reference_observations=store.write_array(observations - 1) if paired else None,
    )
    if paired:
        panel = store.write_artifact(
            "panel",
            derived_from={},
            produced_by="run:imported_measurements",
            json_files={
                "metadata.json": {
                    "source": {"file": "observations.parquet"},
                    "variables": [
                        v.model_dump(mode="json") for v in report.observation_layout.variables
                    ],
                }
            },
            parquet_files={
                "panel.parquet": pl.DataFrame(
                    {
                        "indicator_id": ["indicator:x", "indicator:y", "indicator:unused"],
                        "anchor_time": [
                            datetime(2023, 11, 1),
                            datetime(2023, 11, 2),
                            datetime(2000, 1, 1),
                        ],
                    }
                )
            },
        )
        fitted = store.write_artifact(
            "model",
            derived_from={"model": info.revision, "panel": panel.revision},
            produced_by="run:posterior",
            json_files={"model.json": model.model_dump(mode="json")},
        )
        report = report.model_copy(
            update={
                "model": report.model.model_copy(update={"revision": fitted.revision}),
                "law": PredictiveLawProvenance(
                    kind="fitted",
                    fitted_panel_revision=panel.revision,
                    interpretation="posterior_predictive",
                ),
            }
        )
    repository = StudyRepository(workspace)
    repository.append(
        TransitionRecord(
            seq=2,
            ts="2026-09-12T12:00:00Z",
            action="simulate",
            operation_id="simulate",
            status="applied",
            diagnostics={"report": report.model_dump(mode="json")},
            trace_ids=[],
            resume=None,
        )
    )
    saved = repository.head()
    # A later edit changes the selected outcome; the projection must still use
    # the simulation's pinned model and preserve its freshness provenance.
    _commit(
        workspace,
        "model",
        model.revised(default_outcome=ConstructId("construct:x")).model_dump(mode="json"),
    )
    head = repository.head()
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")

    def no_write(*args, **kwargs):
        pytest.fail("A trajectory read must not write arrays or artifacts")

    monkeypatch.setattr(ArtifactStore, "write_array", no_write)
    monkeypatch.setattr(ArtifactStore, "write_artifact", no_write)
    client = TestClient(create_read_facade_app())
    url = f"/api/episodes/{workspace}/model/simulation-trajectories"
    assert client.get(url, params={"at": commit_id(workspace, 1)}).json() is None
    response = client.get(url, params={"at": saved})
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"]["ref"]["revision"] == saved
    # The paired report pins a fitted value outside the selected fixture's model
    # head; its evidence and calendar origin are still readable with stale provenance.
    assert payload["source"]["validity"] == ("stale" if paired else "fresh")
    view = payload["value"]
    assert view["outcome"] == "construct:y"
    assert view["interval_mass"] == 0.95
    assert view["times"] == [0, 1]
    assert view["time_origin"] == ("2023-11-01T00:00:00Z" if paired else None)
    assert view["outcome_state"]["action"] == {
        "mean": [3, 6],
        "lower": [1.1, 2.2],
        "upper": [4.9, 9.8],
        "n_draws": [2, 2],
    }
    assert list(view["indicators"]) == ["indicator:y"]
    assert view["indicators"]["indicator:y"]["action"] == {
        "mean": [None, 106],
        "lower": [None, 102.2],
        "upper": [None, 109.8],
        "n_draws": [0, 2],
    }
    if paired:
        assert view["outcome_state"]["reference"]["mean"] == [2, 5]
        assert view["indicators"]["indicator:y"]["reference"]["mean"] == [None, 105]
    else:
        assert view["outcome_state"]["reference"] is None
    current = client.get(url).json()
    assert current["source"]["validity"] == "stale"
    assert current["value"] == view
    assert repository.head() == head
    assert repository.latest_seq() == 3


def test_checkpoint_comparison_uses_evidence_from_each_selected_journal_prefix(workspace):
    _measured(workspace)
    for seq in (2, 3):
        report = SimulationReport(
            model=GitRef(
                workspace_id=workspace,
                revision=artifact_revision(workspace, "model", 1),
                path="model.json",
            ),
            design=SimulationSpec(end=1),
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
            TransitionRecord(
                seq=seq,
                ts="2026-09-12T12:00:00Z",
                action="simulate",
                operation_id="simulate",
                inputs={"model": artifact_revision(workspace, "model", 1)},
                status="applied",
                diagnostics={"report": report.model_dump(mode="json")},
                trace_ids=[],
                resume=None,
            )
        )
    client = TestClient(create_read_facade_app())
    response = client.get(
        f"/api/episodes/{workspace}/model-diff",
        params={"before": commit_id(workspace, 1), "after": commit_id(workspace, 2)},
    )
    assert response.status_code == 200
    comparison = response.json()
    assert comparison["before"]["revision"] == commit_id(workspace, 1)
    assert comparison["after"]["revision"] == commit_id(workspace, 2)
    assert comparison["before_simulation"] is None
    assert comparison["after_simulation"]["seed"] == 2
    response = client.get(
        f"/api/episodes/{workspace}/model-diff",
        params={"before": commit_id(workspace, 3), "after": commit_id(workspace, 2)},
    )
    assert response.status_code == 200
    comparison = response.json()
    assert comparison["before_simulation"]["seed"] == 3
    assert comparison["after_simulation"]["seed"] == 2
    assert all(item["change"] == "unchanged" for item in comparison["graph"]["constructs"])
    assert (
        client.get(
            f"/api/episodes/{workspace}/model-diff",
            params={"before": git_oid(4), "after": commit_id(workspace, 2)},
        ).status_code
        == 404
    )

    # Exact model artifacts are also valid comparison inputs; no run is inferred.
    model_revision = artifact_revision(workspace, "model", 1)
    response = client.get(
        f"/api/episodes/{workspace}/model-diff",
        params={"before": model_revision, "after": commit_id(workspace, 2)},
    )
    assert response.status_code == 200, response.text
    comparison = response.json()
    assert comparison["before"]["path"] == "model.json"
    assert comparison["definition_changes"] == []
    assert comparison["before_simulation"] is None
    assert comparison["after_simulation"]["seed"] == 2


def test_snapshot_exists_before_any_compilation(workspace):
    empty = ModelReader(workspace).snapshot()
    assert empty.context.seq == 0
    assert _definitions(empty) == ()
    assert empty.model is None
    _commit(workspace, "model", {"question": "Does X change Y?"})
    snapshot = ModelReader(workspace).snapshot()
    assert _present(snapshot.model).value.question == "Does X change Y?"
    assert _present(snapshot.model).source.ref.model_dump() == {
        "workspace_id": workspace,
        "revision": artifact_revision(workspace, "model", 1),
        "path": "model.json",
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
    assert (
        model.indicator(IndicatorId("indicator:x"))
        is model.get_construct(ConstructId("construct:x")).indicators[0]
    )
    assert _present(snapshot.model).source.pointer == ""
    assert _present(snapshot.model).source.validity == "fresh"
    assert (
        _present(snapshot.findings.dispositions).source.ref == _present(snapshot.model).source.ref
    )
    assert ModelSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_rename_preserves_identity_and_historical_content(workspace):
    _measured(workspace)
    before = ModelReader(workspace).snapshot()
    assert before.context.workspace_id == workspace
    assert _present(before.model).source.ref == GitRef(
        workspace_id=workspace, revision=artifact_revision(workspace, "model", 1), path="model.json"
    )
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Treatment"
    _commit(workspace, "model", payload, pins={"model": artifact_revision(workspace, "model", 1)})
    after = ModelReader(workspace).snapshot()
    assert [entity.id for entity in _definitions(before)] == [
        entity.id for entity in _definitions(after)
    ]
    assert ModelReader(workspace, at=before.context.commit_id).snapshot() == before
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
    assert {item.target.id for item in _present(planned.findings.dispositions).value} == {
        item.id for item in _definitions(planned)
    }
    _commit(
        workspace,
        "model",
        _model().model_dump(mode="json"),
        pins={"model": artifact_revision(workspace, "model", 1)},
    )
    assert _present(
        ModelReader(workspace).snapshot().findings.dispositions
    ).source.ref.model_dump() == {
        "workspace_id": workspace,
        "revision": artifact_revision(workspace, "model", 2),
        "path": "model.json",
    }
    assert ModelReader(workspace, at=planned.context.commit_id).snapshot() == planned


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
    assert len(_definitions(ModelReader(workspace, at=commit_id(workspace, 1)).snapshot())) == 5


def test_uncommitted_versions_and_failed_attempts_never_become_snapshots(workspace):
    _measured(workspace)
    before = ModelReader(workspace).snapshot()
    ArtifactStore(workspace).write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": {"question": "Uncommitted"}},
    )
    StudyRepository(workspace).append(
        TransitionRecord(
            seq=3,
            ts="2026-09-12T13:00:00Z",
            action="edit_model",
            inputs={"expected_revision": artifact_revision(workspace, "model", 1)},
            status="raised",
            trace_ids=[],
            resume=None,
        )
    )
    assert ModelReader(workspace).snapshot() == before
    with pytest.raises(SnapshotRevisionNotFound):
        ModelReader(workspace, at=commit_id(workspace, 3))
    client = TestClient(create_read_facade_app())
    url = f"/api/episodes/{workspace}/model"
    assert client.get(url).json()["context"]["seq"] == 1
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
        graph_constructs(payload["model"]["value"])[0]["indicators"][0]["id"] = "indicator:y"
    with pytest.raises(ValidationError):
        ModelSnapshot.model_validate(payload)


def test_owned_likelihood_survives_reused_names(workspace, monkeypatch):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[1]["indicators"][0]["likelihood"] = {
        "law": observation_law(
            ConstructId("construct:y"), DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY
        ).model_dump(mode="json"),
        "reasoning": "Test",
    }
    _commit(workspace, "model", payload)
    before = ModelReader(workspace).snapshot()
    graph_constructs(payload)[0]["indicators"][0]["name"] = "Y_obs"
    graph_constructs(payload)[1]["indicators"][0]["name"] = "X_obs"
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
    assert _present(after.model).value.indicator(IndicatorId("indicator:y")).name == "X_obs"
    assert ModelReader(workspace, at=before.context.commit_id).snapshot() == before


@pytest.mark.parametrize("owner", ["constructs", "edges"])
def test_owned_mechanisms_survive_rename_and_disappear_with_owner(workspace, owner):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    field = "dynamics" if owner == "constructs" else "mechanisms"
    from nof1_causal_lab.artifacts.expressions import hill, restoring_force, state
    from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec

    entity = (graph_constructs(payload) if owner == "constructs" else payload["edges"])[0]
    mechanism = DynamicsMechanismSpec(
        id="mechanism:owned-term",
        expression=restoring_force(
            entity["id"],
            center=0,
            stiffness=1,
            quartic=0,
        )
        if owner == "constructs"
        else hill(
            state(entity["cause"]["id"]),
            emax=2,
            ec50=1,
            n=2,
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
    assert ModelReader(workspace, at=before.context.commit_id).snapshot() == before


def test_rename_changes_only_construct_label():
    payload = _model().model_dump(mode="json")
    original = ModelSpec.model_validate(payload)
    graph_constructs(payload)[0]["name"] = "Renamed"
    renamed = ModelSpec.model_validate(payload)
    assert original.indicators == renamed.indicators
    assert original.edges[0].id == renamed.edges[0].id
    assert original.edges[0].effect == renamed.edges[0].effect
    assert original.edges[0].cause.model_copy(update={"name": "Renamed"}) == renamed.edges[0].cause
    assert original.state_order == renamed.state_order


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
    assert (
        _present(after.findings.identification).value
        == _present(before.findings.identification).value
    )
    assert _present(after.findings.identification).source.validity == "stale"
    assert after.context.state.current["identification_report"].derived_from == {
        "model": artifact_revision(workspace, "model", 1)
    }
    assert "construct:x" in _present(after.findings.identification).value.estimable_treatments


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
    assert restored.estimable_treatments == ["construct:x"]
    identified = restored.treatments[ConstructId("construct:x")]
    assert identified.status == "identified"
    assert identified.estimand == "P(Y | do(X))"
    assert restored.non_identifiable[ConstructId("construct:y")].confounders == ["construct:x"]
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
    assert reader.snapshot().context.seq == 1
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
    path = f"/api/episodes/{workspace}/model/{accessor.replace('_', '-')}"
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
