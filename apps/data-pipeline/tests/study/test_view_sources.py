"""Read findings follow their scientific revision and observational inputs."""

from nof1_causal_lab.artifacts.data_preparation import FilePreparedDataMetadata

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLawSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.study.artifact_files import artifact_file_spec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    Applied,
    DataPreparationResult,
    ModelFitResult,
)
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record
from tests.git_fixtures import artifact_revision, commit_id
from tests.helpers import make_model, write_question
from tests.inference_fixtures import compile_fit_fixture, compile_model_fixture
from tests.model_fixtures import x_y_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid

FIXTURE = Path(__file__).resolve().parents[4] / "data/DEMO/fixture/artifacts"


@pytest.mark.contract
def test_runtime_diagnostic_subjects_match_posterior_marginals():
    from nof1_causal_lab.actions.inference.subjects import parameter_references
    from nof1_causal_lab.artifacts.identity import ParameterRef
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    model = x_y_model()
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

def test_fit_without_retained_atoms_has_no_report(monkeypatch, tmp_path):
    from nof1_causal_lab.study.records import FitAttempt
    from nof1_causal_lab.utils import data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("ABSENT"), StudyRepository("ABSENT")
    info = store.write_artifact("model", derived_from={}, produced_by="edit_model",
        json_files={"model.json": x_y_model().model_dump(mode="json", round_trip=True)})
    journal.append(applied_record(Applied(result=None, effects=ActionEffects(produced=(write_question(store), info))), seq=1))
    record = applied_record(Applied(result=None, effects=ActionEffects()), seq=2)
    record = record.revised(attempt=FitAttempt(action="fit", request=None, outcome=record.attempt.outcome))
    journal.append(record)
    reader = ModelReader("ABSENT", at=journal.head())
    assert reader.inference_report is None
    assert reader.fit() is None
    assert reader.parameter_draws().kind == "unavailable"


@pytest.mark.contract
def test_joint_reports_and_raw_draws_use_production_labels_without_compiling(monkeypatch, tmp_path):
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.posterior import InferenceEvidence
    from nof1_causal_lab.utils import data
    from tests.model_fixtures import load_model_fixture

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("LABELS"), StudyRepository("LABELS")
    model = load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")
    identity, layout = next(iter(model.law_layouts.items()))
    labels = {element: f"Production label {index}" for index, element in enumerate(layout.labels)}
    model = model.revised(law_layouts={identity: layout.revised(labels=labels)})
    info = store.write_artifact("model", derived_from={}, produced_by="fit",
        json_files={"model.json": model.model_dump(mode="json", round_trip=True)})
    panel = store.write_artifact("panel", derived_from={}, produced_by="prepare_data")
    result = ModelFitResult(model=GitRef(workspace_id="LABELS", revision=info.revision, path="model.json"),
        panel=GitRef(workspace_id="LABELS", revision=panel.revision, path="panel.parquet"),
        evidence=InferenceEvidence(distribution=identity, time_origin=None, duration_seconds=0))
    journal.append(applied_record(Applied(result=result, effects=ActionEffects(produced=(write_question(store), panel, info))), seq=1))
    monkeypatch.setattr("nof1_causal_lab.models.ssm.compile.inputs.compile_model",
        lambda *_args: pytest.fail("Reports and raw atoms do not need the compiler"))
    reader = ModelReader("LABELS", at=journal.head())
    report = reader.inference_report
    assert report is not None
    assert {row.parameter for row in report.value.core.posterior_marginals} == set(labels.values())
    assert report.value.core.inference_diagnostics is None
    assert report.source.pointer == "/attempt/outcome/result/evidence"
    columns = reader.parameter_draws()
    assert columns.kind == "available"
    assert {column.label for column in columns.value} == set(labels.values())
    assert all(len(column.values) == report.value.core.inference_metadata.n_samples for column in columns.value)
    assert set(reader.state.current) == {"question", "model", "panel"}


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
    from nof1_causal_lab.artifacts.data_preparation import (
        DataPreparationSpec,
        DataVariableSpec,
        FileSourceRef,
        PreparedDataMetadata,
        SemanticExtractionSpec,
    )
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("PLOTS")
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
    preparation = DataPreparationSpec(
        default_window="1d",
        variables=(
            DataVariableSpec(
                observation=indicator.observation,
                extraction=SemanticExtractionSpec(how_to_measure="Read Y"),
            ),
        ),
    )
    metadata = FilePreparedDataMetadata(source=FileSourceRef(files=("observations.csv",)), preparation=preparation, time_origin=None)
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": metadata.model_dump(mode="json")},
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
    model = store.write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": candidate.model_dump(mode="json")},
    )
    StudyRepository("PLOTS").append(
        applied_record(
            Applied(
                result=None,
                effects=ActionEffects(
                    produced=[
                        write_question(store),
                        model.revised(
                            derived_from={"panel": artifact_revision("PLOTS", "panel", 1)}
                        ),
                        panel,
                    ]
                ),
            ),
            seq=1,
            ts="2026-09-14T12:00:00Z",
            trace_ids=[],
        )
    )
    reader = ModelReader("PLOTS", at=StudyRepository("PLOTS").head())
    view = reader.snapshot()
    assert view is not None
    diagnostic = view.likelihood_diagnostics[indicator.observation.id]
    assert sum(histogram_bin.count for histogram_bin in diagnostic) == len(observed)
    current_panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": metadata.model_dump(mode="json")},
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
            Applied(
                result=DataPreparationResult(), effects=ActionEffects(produced=[current_panel])
            ),
            seq=2,
            ts="2026-09-14T13:00:00Z",
            trace_ids=[],
        )
    )
    revised = ModelReader("PLOTS", at=StudyRepository("PLOTS").head()).snapshot()
    assert revised is not None
    assert revised.likelihood_diagnostics != view.likelihood_diagnostics
    assert sum(item.count for item in revised.likelihood_diagnostics[indicator.observation.id]) == 1


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
            Applied(
                result=None,
                effects=ActionEffects(produced=[write_question(store), info]),
            ),
            seq=1,
            ts="2026-09-14T12:00:00Z",
            trace_ids=[],
        )
    )
    views = ModelReader("DEFINITION", at=StudyRepository("DEFINITION").head())
    assert views.model == model
    assert any(views.snapshot().authoring_prior_densities.values())
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
            Applied(result=None, effects=ActionEffects(produced=[revision])),
            seq=2,
            ts="2026-09-14T13:00:00Z",
            trace_ids=[],
        )
    )
    assert ModelReader("DEFINITION", at=StudyRepository("DEFINITION").head()).model == changed
    assert views.model == model
