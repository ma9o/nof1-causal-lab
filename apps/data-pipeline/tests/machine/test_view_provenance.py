"""Read findings follow their scientific revision and observational inputs."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.artifact_files import artifact_file_spec
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.moves import RunOperation
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.views import read_artifact_views
from nof1_causal_lab.models.likelihoods import observation_law
from tests.helpers import make_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId

FIXTURE = Path(__file__).resolve().parents[4] / "data/DEMO/fixture/artifacts"


def test_inference_log_keeps_findings_across_admission_republication_and_tracks_changed_data(
    monkeypatch, tmp_path
):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("BINDINGS"), EpisodeJournal("BINDINGS")
    log = json.loads((FIXTURE.parent / "inference.json").read_text())
    log["input_pins"]["model"] = 1
    log["report"]["inference_diagnostics"]["experimental_kernel"] = {"new_metric": [None, 0.5]}
    artifacts: tuple[tuple[int, ArtifactId, dict[ArtifactId, int]], ...] = (
        (1, "model", {}),
        (4, "panel", {"model": 1}),
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
        info = store.write_version(
            aid,
            provenance="computed",
            derived_from=pins,
            produced_by=f"run:{aid}",
            json_files=files,
        )
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-07-08T12:00:00Z",
                move=RunOperation(operation_id="statistical_model_spec"),
                status="applied",
                produced=[info],
                trace_ids=[],
                resume=None,
            )
        )
    record = TransitionRecord(
        seq=5,
        ts="2026-07-08T12:00:00Z",
        move=RunOperation(operation_id="posterior"),
        status="applied",
        produced=[],
        diagnostics=log,
        trace_ids=[],
        resume=None,
    )
    journal.append(record)
    # This log intentionally retains only display findings, never invented joint samples.
    historical = ModelReader("BINDINGS", at_seq=5).fit()
    assert historical is not None
    assert historical.value.report.posterior_marginals
    assert historical.source.ref.model_dump() == {"seq": 5}
    assert historical.source.validity == "fresh"
    admission = store.write_version(
        "admission_report",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="run:statistical_model_spec",
        json_files={"admission_report.json": {}},
    )
    journal.append(
        record.model_copy(
            update={
                "seq": 6,
                "produced": [admission],
                "diagnostics": {},
                "move": RunOperation(operation_id="statistical_model_spec"),
            }
        )
    )
    republished = ModelReader("BINDINGS").fit()
    assert republished is not None
    assert republished.value.report == historical.value.report
    panel = store.write_version(
        "panel", provenance="computed", derived_from={"model": 1}, produced_by="run:measurements"
    )
    journal.append(
        record.model_copy(
            update={
                "seq": 7,
                "produced": [panel],
                "diagnostics": {},
                "move": RunOperation(operation_id="measurements"),
            }
        )
    )
    current = ModelReader("BINDINGS").fit()
    assert current is not None
    assert current.source.validity == "stale"
    assert not current.value.report.posterior_marginals
    assert not current.value.edge_estimates
    assert (
        current.value.report.inference_diagnostics == historical.value.report.inference_diagnostics
    )
    assert current.value.report.assessment == historical.value.report.assessment
    assert ModelReader("BINDINGS", at_seq=5).fit() == historical


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
    panel = store.write_version(
        "panel",
        provenance="computed",
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
    validation = store.write_version(
        "validation_report",
        provenance="computed",
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
    model = store.write_version(
        "model",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"model.json": candidate.model_dump(mode="json")},
    )
    validation = store.write_version(
        "validation_report",
        provenance="computed",
        derived_from={"model": 1, "panel": 1},
        produced_by="derive:validation_report",
        json_files={
            "validation_report.json": {"is_valid": True, "indicators": {}, "dataset_issues": []}
        },
    )
    report = store.write_version(
        "admission_report",
        provenance="computed",
        derived_from={"model": 1, "panel": 1},
        produced_by="run:statistical_model_spec",
        json_files={"admission_report.json": {"prior_predictive_samples": {indicator.id: prior}}},
    )
    state = EpisodeState().with_versions([panel, validation, model, report])
    view = read_artifact_views(store, state).model_diagnostics
    assert view is not None
    diagnostic = view.likelihood_diagnostics["indicator:8ab0e6245f029d222a9a"]
    assert sum(bin.count for bin in diagnostic.histogram) == len(observed)
    assert diagnostic.prior_counts == pytest.approx(expected_counts)
    assert diagnostic.prior_outside_fraction == pytest.approx(expected_outside)
    current_panel = store.write_version(
        "panel",
        provenance="computed",
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
    revised = read_artifact_views(store, state.with_versions([current_panel])).model_diagnostics
    assert revised is not None
    assert revised.likelihood_diagnostics == {}


def test_model_view_reads_canonical_science_without_a_compiled_plan(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("DEFINITION")
    model = ModelSpec.model_validate_json((FIXTURE / "model.json").read_text())
    info = store.write_version(
        "model",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"model.json": model.model_dump(mode="json")},
    )
    state = EpisodeState().with_versions([info])
    views = read_artifact_views(store, state)
    assert views.model == model
    assert views.model_diagnostics is not None
    assert any(views.model_diagnostics.prior_densities.values())
    assert all(
        "prior_density_points" not in parameter.model_dump() for parameter in model.parameters
    )
    changed = model.revised(measurement_clock="2d")
    revision = store.write_version(
        "model",
        provenance="human",
        derived_from={"model": 1},
        produced_by=None,
        json_files={"model.json": changed.model_dump(mode="json")},
    )
    assert read_artifact_views(store, state.with_versions([revision])).model == changed
    assert read_artifact_views(store, state).model == model
