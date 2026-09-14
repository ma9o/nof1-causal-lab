"""Offline history conversion preserves source pins, findings and numerical payloads."""

import hashlib
import json
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from scripts.migrate_additive_model import (
    merge_scientific_values,
    migrate_fitted,
    migrate_workspace,
    validate_retired_compiler,
)
from scripts.migrate_mechanism_identity import remap_references, retired_identity_map

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.moves import is_stale
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, replay_state
from nof1_causal_lab.models.ssm import numerics as numeric
from tests.helpers import complete_test_model, make_model


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _version(source, aid, version, filename, payload, pins):
    info = {
        "artifact_id": aid,
        "version": version,
        "provenance": "computed",
        "derived_from": pins,
        "produced_by": f"run:{aid}",
        "created_at": f"2026-01-{version:02d}T00:00:00+00:00",
    }
    directory = source / "store" / aid / f"v{version}"
    _write(directory / "meta.json", info)
    if filename:
        _write(directory / filename, payload)
    return info


def _record(source, seq, operation, produced=(), retracted=(), status="applied", resume=None):
    _write(
        source / "episode/journal" / f"{seq:06d}.json",
        {
            "seq": seq,
            "ts": f"2026-01-{seq:02d}T00:00:00+00:00",
            "move": {"kind": "run", "artifact_id": operation},
            "status": status,
            "reason": "not ready" if status == "rejected" else None,
            "error_type": "TestFailure" if status == "raised" else None,
            "error_message": "retained error" if status == "raised" else None,
            "diagnostics": {},
            "produced": list(produced),
            "retracted": list(retracted),
            "trace_ids": ["original-trace"],
            "resume": resume,
        },
    )


def _legacy_layers(model):
    """A deliberately retired fixture, confined to migration tests."""

    inverse = {new: old for old, new in retired_identity_map(model).items()}
    from tests.slot_fixtures import flat_catalogue_payload

    value = remap_references(flat_catalogue_payload(model), inverse)
    latent = {key: value[key] for key in ("constructs", "edges", "default_outcome")}
    if latent["default_outcome"] is not None:
        latent["default_outcome"] = {"kind": "construct", "id": latent["default_outcome"]}
    measurement = {
        "measurement_structure": {"model_clock": value["measurement_clock"], "indicators": []},
        "known_inputs": [],
        "scientific_only_constructs": [],
    }
    statistical = {
        "likelihoods": [],
        "mechanisms": [],
        "parameters": value["parameters"],
        "initialization_policy": value["policies"]["initialization"],
        "observation_intercept_policy": value["policies"]["observation_intercept"],
    }
    for owner in latent["constructs"]:
        for indicator in owner.pop("indicators"):
            likelihood = indicator.pop("likelihood")
            if likelihood is not None:
                statistical["likelihoods"].append({"indicator_id": indicator["id"], **likelihood})
            measurement["measurement_structure"]["indicators"].append(
                {"construct_id": owner["id"], **indicator}
            )
        for term in owner.pop("dynamics"):
            term.pop("id")
            statistical["mechanisms"].append({"target_id": owner["id"], **term})
    for owner in latent["edges"]:
        for term in owner.pop("mechanisms"):
            term.pop("id")
            statistical["mechanisms"].append({"edge_id": owner["id"], **term})
    for parameter in statistical["parameters"]:
        reference = parameter.pop("distribution")
        parameter["prior"] = (
            model.model_dump(mode="json")["distributions"][reference]
            if reference is not None
            else None
        )
        parameter["prior_transform"] = parameter.pop("distribution_transform")
        parameter["prior_reasoning"] = "Original proposal"
        parameter["prior_sources"] = []
        parameter["owners"] = [
            owner for owner in parameter["owners"] if owner["kind"] != "mechanism"
        ]
        parameter.update(
            role="test retired vocabulary", constraint="test retired vocabulary", elements=[]
        )
    return latent, measurement, statistical


@pytest.fixture
def historical(tmp_path):
    source = tmp_path / "original" / "HISTORY"
    model = make_model(["Treatment", "Outcome"], [("Treatment", "Outcome")])
    model = model.revised(default_outcome=model.constructs[1].id)
    latent, measurement, _ = _legacy_layers(model)
    q = _version(source, "question", 1, "question.json", {"text": "Question"}, {})
    l1 = _version(
        source,
        "latent_structure",
        1,
        "latent-structure.json",
        {"latent_structure": latent},
        {"question": 1},
    )
    m1 = _version(
        source,
        "measurement_structure",
        1,
        "measurement_structure.json",
        measurement,
        {"latent_structure": 1},
    )
    status = {
        "identifiable_treatments": {},
        "non_identifiable_treatments": {
            model.constructs[0].id: {"confounders": [], "notes": "Original negative finding"},
        },
    }
    c1 = _version(
        source,
        "causal_design",
        1,
        "causal_design.json",
        {"causal_design": {"latent": latent, "identifiability": status}},
        {"latent_structure": 1, "measurement_structure": 1},
    )
    panel = _version(source, "panel", 5, None, None, {"measurement_structure": 1})
    pl.DataFrame({"value": [1.25, 2.5]}).write_parquet(source / "store/panel/v5/table.parquet")
    latent2 = deepcopy(latent)
    latent2["constructs"][0]["name"] = "Renamed treatment"
    l2 = _version(
        source,
        "latent_structure",
        2,
        "latent-structure.json",
        {"latent_structure": latent2},
        {"question": 1},
    )
    measurement = deepcopy(measurement)
    measurement["measurement_structure"]["indicators"][0]["how_to_measure"] = (
        "Revised extraction rule"
    )
    m2 = _version(
        source,
        "measurement_structure",
        2,
        "measurement_structure.json",
        measurement,
        {"latent_structure": 2},
    )
    positive = {
        "identifiable_treatments": {
            model.constructs[0].id: {"method": "do_calculus", "estimand": "retained estimand"}
        },
        "non_identifiable_treatments": {},
    }
    c2 = _version(
        source,
        "causal_design",
        2,
        "causal_design.json",
        {"causal_design": {"latent": latent2, "identifiability": positive}},
        {"latent_structure": 2, "measurement_structure": 2},
    )
    report = _version(
        source,
        "identification_report",
        1,
        "identification_report.json",
        {"estimable_treatments": [model.constructs[0].id]},
        {"causal_design": 2},
    )
    _record(source, 1, "raw_data", [q])
    _record(source, 2, "latent_structure", [l1])
    _record(source, 3, "measurement_structure", [m1, c1])
    workers = _version(
        source,
        "measurements",
        1,
        "measurements.json",
        {
            "workers": [
                {
                    "worker_id": 0,
                    "status": "failed",
                    "n_extractions": 0,
                    "n_windows": 3,
                    "error": "Retained worker failure",
                }
            ]
        },
        {"measurement_structure": 1},
    )
    _record(source, 4, "measurements", [workers, panel])
    _record(source, 5, "posterior", status="rejected")
    _record(
        source,
        6,
        "statistical_model_spec",
        status="raised",
        resume={"kind": "model_spec", "run_id": "seq-000006", "checkpoint_id": "old"},
    )
    _record(source, 7, "latent_structure", [l2])
    _record(source, 8, "measurement_structure", [m2, c2, report])
    _write(source / "episode/traces/000003/original-trace.json", {"trace": "original"})
    return source, tmp_path / "converted" / "HISTORY"


def _activate(monkeypatch, destination):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(destination.parent))
    return ArtifactStore(destination.name), EpisodeJournal(destination.name)


def test_history_keeps_original_pins_failures_negative_findings_and_source_bytes(
    historical, monkeypatch
):
    source, destination = historical
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    manifest = migrate_workspace(source, destination)
    store, journal = _activate(monkeypatch, destination)
    records = journal.read_all()
    assert [r.seq for r in records] == list(range(1, 9))
    assert records[3].diagnostics["workers"][0]["error"] == "Retained worker failure"
    assert (
        records[3].diagnostics["model_input"]
        == records[3].produced[0].consumed_model_inputs["extraction"]
    )
    assert not (destination / "store/measurements").exists()
    assert records[4].status == "rejected"
    assert records[5].error_message == "retained error"
    assert records[5].resume is None
    assert all(r.trace_ids == ["original-trace"] for r in records)
    restored = manifest["restored_identification_reports"]["1"]
    assert restored == 2  # The original positive report remains version 1.
    assert any(
        i.artifact_id == "identification_report" and i.version == restored
        for i in records[2].produced
    )
    negative = store.read_json_file("identification_report", restored, "identification_report.json")
    assert next(iter(negative["treatments"].values()))["notes"] == "Original negative finding"
    assert store.read_meta("panel", 5).derived_from == {
        "model": manifest["model_revisions"]["measurement_structure/v1"]
    }
    assert is_stale(replay_state(journal.read_all()), "panel")
    assert (destination / "store/panel/v5/table.parquet").read_bytes() == before[
        "store/panel/v5/table.parquet"
    ]
    assert {key: hashlib.sha256(value).hexdigest() for key, value in before.items()} == manifest[
        "source_sha256"
    ]
    assert {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    } == before


def test_retracting_details_restores_entities_and_removes_their_findings(historical, monkeypatch):
    source, destination = historical
    _record(
        source,
        9,
        "measurement_structure",
        retracted=[
            {"artifact_id": aid, "reason_ref": "original.reason"}
            for aid in ("measurement_structure", "causal_design", "identification_report")
        ],
    )
    manifest = migrate_workspace(source, destination)
    store, journal = _activate(monkeypatch, destination)
    state = replay_state(journal.read_all())
    assert state.current["model"].version == manifest["model_revisions"]["latent_structure/v2"]
    assert not state.has("identification_report")
    model = ModelSpec.model_validate(
        store.read_json_file("model", state.current["model"].version, "model.json")
    )
    assert model.constructs[0].name == "Renamed treatment"
    assert model.indicators == ()
    assert len(journal.read_all()[-1].retracted) == 1


def test_conflicting_historical_pins_fail_without_leaving_a_destination(historical):
    source, destination = historical
    path = source / "store/causal_design/v2/meta.json"
    info = json.loads(path.read_text())
    info["derived_from"]["latent_structure"] = 1
    _write(path, info)
    with pytest.raises(ValueError, match="Inconsistent scientific source pins"):
        migrate_workspace(source, destination)
    assert not destination.exists()
    assert not list(destination.parent.glob(".additive-migration-*"))


def test_destination_preserves_the_workspace_identity(historical):
    source, destination = historical
    with pytest.raises(ValueError, match="preserve the workspace ID"):
        migrate_workspace(source, destination.with_name("DIFFERENT"))


def _retired_compiler(model, compiled, statistical):
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    old_ids = {new: old for old, new in retired_identity_map(model).items()}
    bindings = remap_references(
        [binding.model_dump(mode="json") for binding in parameter_bindings(model)[0]], old_ids
    )
    definitions = {p["id"]: p for p in statistical["parameters"]}
    for binding in bindings:
        definitions[binding["parameter_id"]]["elements"] = binding.pop("elements")
    return {
        "schema_version": 2,
        "parameters": list(definitions.values()),
        "parameter_bindings": bindings,
        "structure": {
            "spec": {"latent_ids": numeric.state_ids(model)},
            "anchor_certificates": [item.model_dump(mode="json") for item in compiled],
        },
        "compile_diagnostics": [],
    }


def test_retired_compiler_matches_checks_derived_from_the_model():
    model = complete_test_model(make_model(["X"]))
    compiled = model.check_execution()
    latent, measurement, statistical = _legacy_layers(model)
    old = _retired_compiler(model, compiled, statistical)
    converted_model = merge_scientific_values(latent, measurement, statistical)
    from tests.slot_fixtures import fixture_parameter_id

    ids = {
        parameter.id: fixture_parameter_id(
            model.parameter_context(parameter.id).quantity,
            model.parameter_context(parameter.id).owners,
        )
        for parameter in model.parameters
        if any(owner.kind == "mechanism" for owner in model.parameter_context(parameter.id).owners)
    }
    expected = ModelSpec.model_validate(remap_references(model.model_dump(mode="json"), ids))
    from nof1_causal_lab.models.model_distributions import with_parameter_distributions

    expected = with_parameter_distributions(
        expected.revised(
            parameters=tuple(
                p.model_copy(update={"distribution": None}) for p in expected.parameters
            ),
            distributions={},
        ),
        {
            p.id: law
            for p in expected.parameters
            if (law := expected.distribution_for(p.id)) is not None
        },
    )
    assert converted_model == expected
    validate_retired_compiler(old, converted_model)
    assert converted_model.check_execution() == compiled
    old["structure"]["anchor_certificates"] = []
    with pytest.raises(ValueError, match="anchors disagree"):
        validate_retired_compiler(old, converted_model)


def test_fitted_conversion_keeps_exact_draws_times_and_scientific_identities():
    from types import SimpleNamespace

    import cloudpickle

    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
        ParticleMCMCPosterior,
    )
    from nof1_causal_lab.models.ssm.parameterization import build_site_registry

    model = complete_test_model(make_model(["X"]))
    _, _, statistical = _legacy_layers(model)
    retired = _retired_compiler(model, model.check_execution(), statistical)
    native = {
        site.name: jnp.arange(2 * max(1, int(np.prod(site.shape))), dtype=float).reshape(
            (2, *site.shape)
        )
        for site in build_site_registry(model)
    }
    original = SimpleNamespace(
        result=ParticleMCMCPosterior(
            draws=JointPosteriorDraws(
                parameters=native, latent_paths=jnp.arange(4.0).reshape(2, 2, 1)
            )
        ),
        spec=SimpleNamespace(retired_numeric_definition=True),
        times=jnp.array([0.0, 2.0]),
        observation_support=None,
    )
    from nof1_causal_lab.models.ssm.inference.persistence import model_draws

    converted = migrate_fitted(cloudpickle.dumps(original), model, retired)
    restored = model_draws(converted)
    for name, values in native.items():
        np.testing.assert_array_equal(restored.parameters[name], values)
    np.testing.assert_array_equal(restored.latent_paths, original.result.draws.latent_paths)
    np.testing.assert_array_equal(converted.time_points, original.times)
    assert isinstance(converted, ModelSpec)
    assert not hasattr(converted, "provenance")
