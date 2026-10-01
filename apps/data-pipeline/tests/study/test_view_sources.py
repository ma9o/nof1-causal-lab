"""Read findings follow their scientific revision and observational inputs."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLawSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.study.artifact_files import artifact_file_spec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.lineage import scientific_inference_report
from nof1_causal_lab.study.records import AttemptRecord
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from tests.git_fixtures import artifact_revision, commit_id
from tests.helpers import make_model
from tests.model_fixtures import compile_fit_fixture

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid

FIXTURE = Path(__file__).resolve().parents[4] / "data/DEMO/fixture/artifacts"


@pytest.mark.contract
def test_runtime_diagnostic_subjects_match_posterior_marginals():
    from nof1_causal_lab.actions.inference.subjects import reference_posterior_findings
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'view_sources/runtime_diagnostic_subjects_match_posterior_marginals_complete_test_model.json').read_text())
    bindings, auxiliary = parameter_bindings(model)
    coordinates = [coordinate for b in bindings for coordinate in b.coordinates.values()]
    rows = [
        {"coordinate": c.model_dump(mode="json"), "parameter": c.label, "r_hat": 1.001}
        for c in [*coordinates, *auxiliary]
    ]
    marginals, _ = reference_posterior_findings(
        compile_fit_fixture(model),
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
@pytest.mark.parametrize("failed", [False, True])
def test_inference_log_keeps_findings_across_authoring_log_updates_and_tracks_changed_data(
    monkeypatch, tmp_path, failed
):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("BINDINGS"), StudyRepository("BINDINGS")
    log = json.loads((FIXTURE.parent / "inference.json").read_text())
    rows = log["report"]["inference_diagnostics"]["mcmc"]["per_parameter"]
    if failed:
        rows[0].update(r_hat=1.1, ess_bulk=None, ess_tail=1)
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
            AttemptRecord(
                seq=seq,
                ts="2026-07-08T12:00:00Z",
                action="edit_model",
                inputs={"expected_revision": None},
                status="applied",
                produced=[info],
                trace_ids=[],
            )
        )
    log["input_pins"] = {aid: artifact_revision("BINDINGS", aid, 1) for aid in ("model", "panel")}
    record = AttemptRecord(
        seq=5,
        ts="2026-07-08T12:00:00Z",
        action="fit",
        inputs={},
        status="applied",
        produced=[],
        diagnostics=log,
        trace_ids=[],
    )
    journal.append(record)
    # This log intentionally retains only display findings, never invented joint samples.
    historical = ModelReader("BINDINGS", at=commit_id("BINDINGS", 5)).fit()
    assert historical is not None
    assert historical.value.report.posterior_marginals
    convergence = historical.value.convergence
    assert convergence.checked == len(rows)
    assert convergence.passed is not failed
    assert len(convergence.failures) == (3 if failed else 0)
    if failed:
        assert {failure.subject.element_id for failure in convergence.failures} == {
            rows[0]["subject"]["element_id"]
        }
        assert [failure.criterion for failure in convergence.failures] == [
            "R-hat < 1.01",
            "bulk ESS ≥ 400",
            "tail ESS ≥ 400",
        ]
    from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures

    raw = json.loads(json.dumps(log["report"]["inference_diagnostics"]))
    for row in raw["mcmc"]["per_parameter"]:
        del row["subject"]
    assert convergence_failures(raw) == [
        f"{failure.criterion} fails for 1 of {len(rows)} parameters, including {rows[0]['parameter']}"
        for failure in convergence.failures
    ]
    assert historical.source.ref.model_dump() == {
        "workspace_id": "BINDINGS",
        "revision": commit_id("BINDINGS", 5),
        "path": "logs/attempt.json",
    }
    assert historical.source.validity == "fresh"
    journal.append(
        type(record).model_validate(
            {
                **record.model_dump(),
                "seq": 6,
                "produced": [],
                "diagnostics": {"search_queries": {"parameter:test": "Research query"}},
                "action": "edit_model",
            }
        )
    )
    republished = ModelReader("BINDINGS").fit()
    assert republished is not None
    assert republished.value.report == historical.value.report
    assert republished.value.convergence == convergence
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
    )
    journal.append(
        type(record).model_validate(
            {
                **record.model_dump(),
                "seq": 7,
                "produced": [panel],
                "diagnostics": {},
                "action": "prepare_data",
            }
        )
    )
    current_reader = ModelReader("BINDINGS")
    current = current_reader.fit()
    assert current is not None
    assert current.source.validity == "stale"
    assert current_reader.artifact_view("inference_report").summary() == current.value.report
    # Findings remain tied to their fit's pinned panel; freshness marks the
    # changed panel without erasing the historical parameter/edge evidence.
    assert current.value.report.posterior_marginals == historical.value.report.posterior_marginals
    assert current.value.edge_estimates == historical.value.edge_estimates
    assert (
        current.value.report.inference_diagnostics == historical.value.report.inference_diagnostics
    )
    assert current.value.report.loo_diagnostics == historical.value.report.loo_diagnostics
    assert ModelReader("BINDINGS", at=commit_id("BINDINGS", 5)).fit() == historical


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(('family', 'link', 'observed', 'observation_law_payload'), [
    pytest.param('gaussian', 'identity', [0.0, 1.0, 2.0], 'view_sources/likelihood_plot_requires_its_pinned_panel_observation_law_gaussian-identity-observed0.json', id='gaussian-identity-observed0'),
    pytest.param('bernoulli', 'logit', [0.0, 0.0], 'view_sources/likelihood_plot_requires_its_pinned_panel_observation_law_bernoulli-logit-observed1.json', id='bernoulli-logit-observed1'),
])
def test_likelihood_plot_requires_its_pinned_panel(monkeypatch, tmp_path, family, link, observed, observation_law_payload):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("PLOTS")
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
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
    candidate = make_model(["Y"])
    owner = candidate.constructs[0]
    indicator = type(owner.indicators[0]).model_validate(
        {
            **owner.indicators[0].model_dump(),
            "id": "indicator:8ab0e6245f029d222a9a",
            "measurement_dtype": "binary" if family == "bernoulli" else "continuous",
            "aggregation": "last" if family == "bernoulli" else "mean",
            "likelihood": LikelihoodSpec(
                law=ObservationLawSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / observation_law_payload).read_text()), reasoning="Test"
            ),
        }
    )
    candidate = candidate.revised(
        edges=replace_constructs(
            candidate.edges,
            (type(owner).model_validate({**owner.model_dump(), "indicators": (indicator,)}),),
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
        AttemptRecord(
            seq=1,
            ts="2026-09-14T12:00:00Z",
            action="edit_model",
            inputs={},
            status="applied",
            produced=[
                type(model).model_validate(
                    {
                        **model.model_dump(),
                        "derived_from": {"panel": artifact_revision("PLOTS", "panel", 1)},
                    }
                ),
                panel,
                validation,
            ],
            trace_ids=[],
        )
    )
    reader = ModelReader("PLOTS")
    view = reader.diagnostics
    assert view is not None
    diagnostic = view.likelihood_diagnostics[indicator.id]
    assert sum(bin.count for bin in diagnostic.histogram) == len(observed)
    current_panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
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
        AttemptRecord(
            seq=2,
            ts="2026-09-14T13:00:00Z",
            action="prepare_data",
            inputs={},
            status="applied",
            produced=[current_panel],
            trace_ids=[],
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
        AttemptRecord(
            seq=1,
            ts="2026-09-14T12:00:00Z",
            action="edit_model",
            inputs={"expected_revision": None},
            status="applied",
            produced=[info],
            trace_ids=[],
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
        AttemptRecord(
            seq=2,
            ts="2026-09-14T13:00:00Z",
            action="edit_model",
            inputs={"expected_revision": info.revision},
            status="applied",
            produced=[revision],
            trace_ids=[],
        )
    )
    assert ModelReader("DEFINITION").model == changed
    assert views.model == model
