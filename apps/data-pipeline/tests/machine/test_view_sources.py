"""Read findings follow their scientific revision and observational inputs."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.machine.artifact_files import artifact_file_spec
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.inference import scientific_inference_report
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.models.likelihoods import observation_law
from tests.git_fixtures import artifact_revision, commit_id
from tests.helpers import complete_test_model, make_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid

FIXTURE = Path(__file__).resolve().parents[4] / "data/DEMO/fixture/artifacts"


@pytest.mark.contract
def test_runtime_diagnostic_subjects_match_posterior_marginals():
    from nof1_causal_lab.flows.transitions.inference.subjects import reference_posterior_findings
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    bindings, auxiliary = parameter_bindings(model)
    coordinates = [coordinate for b in bindings for coordinate in b.coordinates.values()]
    rows = [
        {"coordinate": c.model_dump(mode="json"), "parameter": c.label, "r_hat": 1.001}
        for c in [*coordinates, *auxiliary]
    ]
    marginals, _ = reference_posterior_findings(
        model,
        [
            {
                "coordinate": c.model_dump(mode="json"),
                "parameter": c.label,
                "mean": 0,
                "sd": 1,
                "lower": -1,
                "upper": 1,
                "interval_kind": "hdi",
                "interval_mass": 0.94,
                "x_values": [],
                "density": [],
            }
            for c in coordinates
        ],
        [],
    )
    report = InferenceReport.model_validate(
        {
            "time_origin": "2024-01-01T00:00:00Z",
            "inference_metadata": {"method": "test", "n_samples": 10, "duration_seconds": 1},
            "inference_diagnostics": {"mcmc": {"per_parameter": rows}},
            "posterior_marginals": marginals,
        }
    )
    view = scientific_inference_report(model, report)
    mcmc = view.inference_diagnostics["mcmc"]
    assert isinstance(mcmc, dict)
    actual = TypeAdapter(list[JsonObject]).validate_python(mcmc["per_parameter"])
    assert [(row["parameter"], row["subject"]) for row in actual] == [
        (row["parameter"], row["subject"]) for row in marginals
    ]
    assert all(row["r_hat"] == 1.001 for row in actual)
    original = report.inference_diagnostics["mcmc"]
    assert isinstance(original, dict)
    assert original["per_parameter"] == rows
    assert scientific_inference_report(model, view) is view


@pytest.mark.contract
def test_inference_log_keeps_findings_across_authoring_log_updates_and_tracks_changed_data(
    monkeypatch, tmp_path
):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("BINDINGS"), StudyRepository("BINDINGS")
    log = json.loads((FIXTURE.parent / "inference.json").read_text())
    log["report"]["inference_diagnostics"]["experimental_kernel"] = {"new_metric": [None, 0.5]}
    artifacts: tuple[tuple[int, ArtifactId, dict[ArtifactId, GitOid]], ...] = (
        (1, "model", {}),
        (4, "panel", {}),
    )
    for seq, aid, pins in artifacts:
        files = (
            {}
            if aid == "panel"
            else {
                next(iter(artifact_file_spec(aid).json.values())): json.loads(
                    (FIXTURE / f"{aid}.json").read_text()
                )
            }
        )
        info = store.write_artifact(
            aid,
            derived_from=pins,
            produced_by=f"run:{aid}",
            json_files=files,
        )
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-07-08T12:00:00Z",
                action="edit_model",
                inputs={"expected_revision": None},
                status="applied",
                produced=[info],
                trace_ids=[],
                resume=None,
            )
        )
    log["input_pins"] = {aid: artifact_revision("BINDINGS", aid, 1) for aid in ("model", "panel")}
    record = TransitionRecord(
        seq=5,
        ts="2026-07-08T12:00:00Z",
        action="fit",
        operation_id="posterior",
        inputs={},
        status="applied",
        produced=[],
        diagnostics=log,
        trace_ids=[],
        resume=None,
    )
    journal.append(record)
    # This log intentionally retains only display findings, never invented joint samples.
    historical = ModelReader("BINDINGS", at=commit_id("BINDINGS", 5)).fit()
    assert historical is not None
    assert historical.value.report.posterior_marginals
    assert historical.source.ref.model_dump() == {
        "workspace_id": "BINDINGS",
        "revision": commit_id("BINDINGS", 5),
        "path": "logs/transition.json",
    }
    assert historical.source.validity == "fresh"
    journal.append(
        record.model_copy(
            update={
                "seq": 6,
                "produced": [],
                "diagnostics": {"search_queries": {"parameter:test": "Research query"}},
                "action": "edit_model",
                "operation_id": "statistical_model_spec",
            }
        )
    )
    republished = ModelReader("BINDINGS").fit()
    assert republished is not None
    assert republished.value.report == historical.value.report
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
    )
    journal.append(
        record.model_copy(
            update={
                "seq": 7,
                "produced": [panel],
                "diagnostics": {},
                "action": "prepare_data",
                "operation_id": "measurements",
            }
        )
    )
    current_reader = ModelReader("BINDINGS")
    current = current_reader.fit()
    assert current is not None
    assert current.source.validity == "stale"
    assert current_reader.artifact_view("inference_report") is current.value.report
    # Findings remain tied to their fit's pinned panel; freshness marks the
    # changed panel without erasing the historical parameter/edge evidence.
    assert current.value.report.posterior_marginals == historical.value.report.posterior_marginals
    assert current.value.edge_estimates == historical.value.edge_estimates
    assert (
        current.value.report.inference_diagnostics == historical.value.report.inference_diagnostics
    )
    assert (
        current.value.report.inference_diagnostics == historical.value.report.inference_diagnostics
    )
    assert current.value.report.loo_diagnostics == historical.value.report.loo_diagnostics
    assert ModelReader("BINDINGS", at=commit_id("BINDINGS", 5)).fit() == historical


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("family", "link", "observed", "prior", "expected_counts", "expected_outside"),
    [
        ("gaussian", "identity", [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0], [0.75, 1.5], 0.25),
        ("bernoulli", "logit", [0.0, 0.0], [0.0, 1.0], [1.0, 1.0], 0.0),
    ],
)
def test_likelihood_plot_preserves_prior_mass_and_requires_its_pinned_panel(
    monkeypatch, tmp_path, family, link, observed, prior, expected_counts, expected_outside
):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("PLOTS")
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
        parquet_files={
            "panel.parquet": pl.DataFrame(
                {
                    "indicator_id": "indicator:8ab0e6245f029d222a9a",
                    "value": observed,
                    "anchor_time": None,
                    "support_start": None,
                    "support_end": None,
                    "support_kind": "point",
                    "summary_operator": "last",
                    "anchor_policy": "support_start",
                    "observation_window": "1d",
                }
            )
        },
    )
    validation = store.write_artifact(
        "validation_report",
        derived_from={},
        produced_by="run:measurements",
        json_files={
            "validation_report.json": {"is_valid": True, "indicators": {}, "dataset_issues": []}
        },
    )
    candidate = make_model(["Y"])
    owner = candidate.constructs[0]
    indicator = owner.indicators[0].model_copy(
        update={
            "id": "indicator:8ab0e6245f029d222a9a",
            "measurement_dtype": "binary" if family == "bernoulli" else "continuous",
            "aggregation": "last" if family == "bernoulli" else "mean",
            "likelihood": LikelihoodSpec(
                law=observation_law(owner.id, family, link), reasoning="Test"
            ),
        }
    )
    candidate = candidate.revised(
        edges=replace_constructs(
            candidate.edges, (owner.model_copy(update={"indicators": (indicator,)}),)
        )
    )
    model = store.write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": candidate.model_dump(mode="json")},
    )
    validation = store.write_artifact(
        "validation_report",
        derived_from={
            "model": artifact_revision("PLOTS", "model", 1),
            "panel": artifact_revision("PLOTS", "panel", 1),
        },
        produced_by="derive:validation_report",
        json_files={
            "validation_report.json": {"is_valid": True, "indicators": {}, "dataset_issues": []}
        },
    )
    StudyRepository("PLOTS").append(
        TransitionRecord(
            seq=1,
            ts="2026-09-14T12:00:00Z",
            action="edit_model",
            operation_id="statistical_model_spec",
            inputs={},
            status="applied",
            produced=[
                model.model_copy(
                    update={"derived_from": {"panel": artifact_revision("PLOTS", "panel", 1)}}
                ),
                panel,
                validation,
            ],
            diagnostics={"prior_predictive": {"samples": {indicator.id: prior}, "diagnostics": []}},
            trace_ids=[],
            resume=None,
        )
    )
    reader = ModelReader("PLOTS")
    view = reader.diagnostics
    assert view is not None
    diagnostic = view.likelihood_diagnostics[indicator.id]
    assert sum(bin.count for bin in diagnostic.histogram) == len(observed)
    assert diagnostic.prior_counts == pytest.approx(expected_counts)
    assert diagnostic.prior_outside_fraction == pytest.approx(expected_outside)
    current_panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
        parquet_files={
            "panel.parquet": pl.DataFrame(
                {
                    "indicator_id": "indicator:8ab0e6245f029d222a9a",
                    "value": [100.0],
                    "anchor_time": None,
                    "support_start": None,
                    "support_end": None,
                    "support_kind": "point",
                    "summary_operator": "last",
                    "anchor_policy": "support_start",
                    "observation_window": "1d",
                }
            )
        },
    )
    StudyRepository("PLOTS").append(
        TransitionRecord(
            seq=2,
            ts="2026-09-14T13:00:00Z",
            action="prepare_data",
            operation_id="measurements",
            inputs={},
            status="applied",
            produced=[current_panel],
            trace_ids=[],
            resume=None,
        )
    )
    revised = ModelReader("PLOTS").diagnostics
    assert revised is not None
    assert revised.likelihood_diagnostics == {}


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_model_view_reads_canonical_science_without_a_compiled_plan(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("DEFINITION")
    model = ModelSpec.model_validate_json((FIXTURE / "model.json").read_text())
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": model.model_dump(mode="json")},
    )
    repository = StudyRepository("DEFINITION")
    repository.append(
        TransitionRecord(
            seq=1,
            ts="2026-09-14T12:00:00Z",
            action="edit_model",
            inputs={"expected_revision": None},
            status="applied",
            produced=[info],
            trace_ids=[],
            resume=None,
        )
    )
    views = ModelReader("DEFINITION")
    assert views.artifact_view("model") is views.model
    assert views.model == model
    assert "diagnostics" not in views.__dict__
    assert views.diagnostics is not None
    assert views.artifact_view("model_diagnostics") is views.diagnostics
    assert any(views.diagnostics.prior_densities.values())
    assert all(
        "prior_density_points" not in parameter.model_dump() for parameter in model.parameters
    )
    changed = model.revised(measurement_clock="2d")
    revision = store.write_artifact(
        "model",
        derived_from={"model": artifact_revision("DEFINITION", "model", 1)},
        produced_by=None,
        json_files={"model.json": changed.model_dump(mode="json")},
    )
    repository.append(
        TransitionRecord(
            seq=2,
            ts="2026-09-14T13:00:00Z",
            action="edit_model",
            inputs={"expected_revision": info.revision},
            status="applied",
            produced=[revision],
            trace_ids=[],
            resume=None,
        )
    )
    assert ModelReader("DEFINITION").model == changed
    assert views.model == model
