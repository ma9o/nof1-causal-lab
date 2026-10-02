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
from nof1_causal_lab.study.artifact_files import artifact_file_spec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import DataPreparationResult, ModelEditResult, ModelFitResult
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record
from tests.git_fixtures import artifact_revision, commit_id
from tests.helpers import make_model
from tests.model_fixtures import compile_fit_fixture, compile_model_fixture

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid

FIXTURE = Path(__file__).resolve().parents[4] / "data/DEMO/fixture/artifacts"


@pytest.mark.contract
def test_runtime_diagnostic_subjects_match_posterior_marginals():
    from nof1_causal_lab.actions.inference.subjects import parameter_references
    from nof1_causal_lab.artifacts.identity import ParameterRef
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    model = ModelSpec.model_validate_json(
        (Path(__file__).resolve().parents[1] / "fixtures/models/common/x_y_model.json").read_text()
    )
    bindings, auxiliary = parameter_bindings(compile_model_fixture(model))
    references = parameter_references(compile_fit_fixture(model))
    assert set(references) == {c for b in bindings for c in b.coordinates.values()} | set(auxiliary)
    for binding in bindings:
        for element, coordinate in binding.coordinates.items():
            reference = references[coordinate]
            assert reference is not None
            label, subject = reference
            assert subject == ParameterRef(parameter_id=binding.parameter_id, element_id=element)
            assert label == binding.elements[element]
    assert all(references[coordinate] is None for coordinate in auxiliary)


@pytest.mark.contract
@pytest.mark.parametrize("failed", [False, True])
def test_inference_log_keeps_findings_across_authoring_log_updates_and_tracks_changed_data(
    monkeypatch, tmp_path, failed
):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("BINDINGS"), StudyRepository("BINDINGS")
    log = json.loads((FIXTURE.parent / "inference.json").read_text())
    rows = log["report"]["inference_diagnostics"]["per_parameter"]
    if failed:
        rows[0].update(r_hat=1.1, ess_bulk=None, ess_tail=1)
    from nof1_causal_lab.artifacts.posterior_diagnostics import ChainDiagnostics
    from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence

    log["report"]["convergence"] = parameter_convergence(
        ChainDiagnostics.model_validate(log["report"]["inference_diagnostics"])
    ).model_dump(mode="json")
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
            applied_record(
                ModelEditResult(produced=[info]), seq=seq, ts="2026-07-08T12:00:00Z", trace_ids=[]
            )
        )
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.posterior import InferenceReport

    record = applied_record(
        ModelFitResult(
            model=GitRef(
                workspace_id="BINDINGS",
                revision=artifact_revision("BINDINGS", "model", 1),
                path="model.json",
            ),
            panel=GitRef(
                workspace_id="BINDINGS",
                revision=artifact_revision("BINDINGS", "panel", 1),
                path="panel.parquet",
            ),
            report=InferenceReport.model_validate(log["report"]),
            retention="report_only",
        ),
        seq=5,
        ts="2026-07-08T12:00:00Z",
    )
    journal.append(record)
    # This log intentionally retains only display findings, never invented joint samples.
    historical = ModelReader("BINDINGS", at=commit_id("BINDINGS", 5)).fit()
    assert historical is not None
    assert historical.value.report.posterior_marginals
    convergence = historical.value.report.convergence
    assert convergence.checked == len(rows)
    from nof1_causal_lab.artifacts.checks import Evaluated, NotEvaluated

    assert convergence.status == ("failed" if failed else "passed")
    problems = [
        a for a in convergence.assessments if isinstance(a, NotEvaluated) or a.outcome == "failed"
    ]
    assert len(problems) == (3 if failed else 0)
    if failed:
        subjects = []
        for item in problems:
            assert not isinstance(item.subject, str)
            subjects.append(item.subject)
        assert {subject.parameter.element_id for subject in subjects} == {
            rows[0]["subject"]["element_id"]
        }
        assert [subject.criterion for subject in subjects] == ["r_hat", "ess_bulk", "ess_tail"]
        assert isinstance(problems[1], NotEvaluated)
        assert isinstance(problems[0], Evaluated)
    from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures

    assert convergence_failures(convergence) == convergence.messages
    assert historical.source.ref.model_dump() == {
        "workspace_id": "BINDINGS",
        "revision": commit_id("BINDINGS", 5),
        "path": "logs/attempt.json",
    }
    assert historical.source.validity == "fresh"
    journal.append(
        applied_record(ModelEditResult(), seq=6, ts="2026-07-08T12:00:00Z"),
        logs={"research.json": b'{"search_queries":{"parameter:test":"Research query"}}'},
    )
    republished = ModelReader("BINDINGS").fit()
    assert republished is not None
    assert republished.value.report == historical.value.report
    assert republished.value.report.convergence == convergence
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
    )
    journal.append(
        applied_record(DataPreparationResult(produced=(panel,)), seq=7, ts="2026-07-08T12:00:00Z")
    )
    current_reader = ModelReader("BINDINGS")
    current = current_reader.fit()
    assert current is not None
    assert current.source.validity == "stale"
    assert current_reader.artifact_view("inference_report") == current.value.report
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
@pytest.mark.parametrize(
    ("family", "link", "observed", "observation_law_payload"),
    [
        pytest.param(
            "gaussian",
            "identity",
            [0.0, 1.0, 2.0],
            "common/observed_gaussian_observation_law.json",
            id="gaussian-identity-observed0",
        ),
        pytest.param(
            "bernoulli",
            "logit",
            [0.0, 0.0],
            "view_sources/likelihood_plot_requires_its_pinned_panel_observation_law_bernoulli-logit-observed1.json",
            id="bernoulli-logit-observed1",
        ),
    ],
)
def test_likelihood_plot_requires_its_pinned_panel(
    monkeypatch, tmp_path, family, link, observed, observation_law_payload
):
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
    indicator = owner.indicators[0].revised(
        observation=owner.indicators[0].observation.revised(
            id="indicator:8ab0e6245f029d222a9a",
            measurement_dtype="binary" if family == "bernoulli" else "continuous",
            aggregation="last" if family == "bernoulli" else "mean",
        ),
        likelihood=LikelihoodSpec(
            law=TypeAdapter(ObservationLawSpec).validate_json(
                (
                    Path(__file__).resolve().parents[1]
                    / "fixtures/models"
                    / observation_law_payload
                ).read_text()
            ),
            reasoning="Test",
        ),
    )
    candidate = candidate.revised(
        edges=replace_constructs(
            candidate.edges,
            (owner.revised(indicators=(indicator,)),),
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
            "validation_report.json": {
                "is_valid": True,
                "data": {"indicators": {}, "dataset_issues": []},
            }
        },
    )
    StudyRepository("PLOTS").append(
        applied_record(
            ModelEditResult(
                produced=[
                    model.revised(derived_from={"panel": artifact_revision("PLOTS", "panel", 1)}),
                    panel,
                    validation,
                ]
            ),
            seq=1,
            ts="2026-09-14T12:00:00Z",
            trace_ids=[],
        )
    )
    reader = ModelReader("PLOTS")
    view = reader.diagnostics
    assert view is not None
    diagnostic = view.likelihood_diagnostics[indicator.observation.id]
    assert sum(histogram_bin.count for histogram_bin in diagnostic.histogram) == len(observed)
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
        applied_record(
            DataPreparationResult(produced=[current_panel]),
            seq=2,
            ts="2026-09-14T13:00:00Z",
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
        applied_record(
            ModelEditResult(produced=[info]), seq=1, ts="2026-09-14T12:00:00Z", trace_ids=[]
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
        applied_record(
            ModelEditResult(produced=[revision]), seq=2, ts="2026-09-14T13:00:00Z", trace_ids=[]
        )
    )
    assert ModelReader("DEFINITION").model == changed
    assert views.model == model
