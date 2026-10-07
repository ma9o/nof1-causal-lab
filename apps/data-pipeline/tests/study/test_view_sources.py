"""Read findings follow their scientific revision and observational inputs."""

import pytest

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.posterior import ModelFitResult
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record
from tests.git_fixtures import artifact_revision
from tests.helpers import write_question
from tests.model_fixtures import x_y_model


@pytest.mark.contract
def test_authored_model_has_no_inference_report(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("ABSENT"), StudyRepository("ABSENT")
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": x_y_model().model_dump(mode="json", round_trip=True)},
    )
    journal.append(
        applied_record(
            store.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=(write_question(store), info))),
            seq=1,
        )
    )
    reader = ModelReader("ABSENT", at=journal.head())
    assert reader.inference_report is None
    assert reader.fit() is None


@pytest.mark.inference(concern="sampling")
def test_joint_reports_and_raw_draws_use_production_labels_without_compiling(monkeypatch, tmp_path):
    from nof1_causal_lab.actions.fit import read_inference_report
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.posterior import InferenceEvidence
    from nof1_causal_lab.utils import data
    from tests.inference_fixtures import inference_metadata
    from tests.model_fixtures import load_model_fixture

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("LABELS"), StudyRepository("LABELS")
    model = load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")
    identity, layout = next(iter(model.law_layouts.items()))
    labels = {element: f"Production label {index}" for index, element in enumerate(layout.labels)}
    model = model.with_entities(law_layouts={identity: layout.revised(labels=labels)})
    prior = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={
            "model.json": load_model_fixture("causal_proofs/treatment_outcome.json").model_dump(
                mode="json", round_trip=True
            )
        },
    )
    from tests.integration.runner_fixtures import seed_panel

    panel = seed_panel(store, model_revision=prior.revision)
    info = store.write_artifact(
        "model",
        derived_from={"model": prior.revision, "panel": panel.revision},
        produced_by="fit",
        json_files={"model.json": model.model_dump(mode="json", round_trip=True)},
    )
    result = ModelFitResult(
        model=GitRef(workspace_id="LABELS", revision=prior.revision, path="model.json"),
        data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0),
        evidence=InferenceEvidence(),
    )
    journal.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=result,
                effects=ActionEffects(
                    produced=(write_question(store), panel, info),
                    reports={
                        "inference": store.write_report(
                            read_inference_report(
                                store, info.revision, result, inference_metadata(model)
                            )
                        )
                    },
                ),
            ),
            seq=1,
        ),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.models.ssm.compile.inputs.compile_model",
        lambda *_args: pytest.fail("Reports and raw atoms do not need the compiler"),
    )
    reader = ModelReader("LABELS", at=journal.head())
    report = reader.inference_report
    assert report is not None
    marginals = report.core.posterior_marginals
    assert marginals is not None
    assert {row.parameter for row in marginals} == set(labels.values())
    assert report.core.inference_metadata.num_chains == 1
    assert report.core.inference_diagnostics is not None
    assert reader.fit_result is not None
    assert set(reader.fit_result.model_dump()) == {"model", "checks", "inference", "arrays"}
    from nof1_causal_lab.numpyro_json import empirical_atoms

    np_atoms = empirical_atoms(reader.fit_result.model.distributions[identity])
    assert len(np_atoms) == report.core.inference_metadata.n_samples
    assert {row.subject.element_id for row in marginals} == set(model.law_layouts[identity].labels)
    assert all(row.empirical[-1].probability == 1 for row in marginals)


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_model_view_reads_canonical_science_without_a_compiled_plan(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("DEFINITION")
    model = x_y_model()
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": model.model_dump(mode="json")},
    )
    repository = StudyRepository("DEFINITION")
    repository.append(
        applied_record(
            store.workspace_id,
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
    assert views.snapshot().model == model
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
            store.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=[revision])),
            seq=2,
            ts="2026-09-14T13:00:00Z",
            trace_ids=[],
        )
    )
    assert ModelReader("DEFINITION", at=StudyRepository("DEFINITION").head()).model == changed
    assert views.model == model
