"""Posterior joins require the selected compiler version, not coincident parameter labels."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nof1_causal_lab.machine.artifact_files import artifact_file_spec
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.moves import RunArtifact
from nof1_causal_lab.machine.snapshots import read_model_snapshot
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.views import read_artifact_views

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId

FIXTURE = Path(__file__).resolve().parents[4] / "data/DEMO/fixture/artifacts"


def test_posterior_coordinates_never_join_across_compiler_versions(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("BINDINGS")
    journal = EpisodeJournal("BINDINGS")
    revisions: tuple[tuple[int, ArtifactId, dict[ArtifactId, int]], ...] = (
        (1, "compiled_ssm", {}),
        (2, "posterior", {"compiled_ssm": 1}),
        (3, "compiled_ssm", {}),
    )
    for seq, aid, pins in revisions:
        payload = json.loads((FIXTURE / f"{aid}.json").read_text())
        info = store.write_version(
            aid,
            provenance="computed",
            derived_from=pins,
            produced_by=f"run:{aid}",
            json_files={next(iter(artifact_file_spec(aid).json.values())): payload},
        )
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-07-08T12:00:00Z",
                move=RunArtifact(artifact_id=aid),
                status="applied",
                produced=[info],
                trace_ids=[],
                resume=None,
            )
        )
    historical = read_model_snapshot("BINDINGS", at_seq=2)
    current = read_model_snapshot("BINDINGS")
    assert historical.fit is not None
    assert current.fit is not None
    assert current.fit.value.posterior.assessment.mcmc_diagnostics is not None
    assert historical.fit.value.posterior.posterior_marginals
    assert historical.fit.source is not None
    assert historical.fit.source.artifact.version == 1
    assert not current.fit.value.posterior.posterior_marginals
    assert not current.fit.value.posterior.assessment.mcmc_diagnostics.per_parameter
    assert not current.fit.value.edge_estimates
    assert not current.fit.value.decay_estimates
    assert current.fit.source.validity == "stale"
    assert current.fit.value.posterior.draws == historical.fit.value.posterior.draws
    assert (
        current.fit.value.posterior.assessment.ppc == historical.fit.value.posterior.assessment.ppc
    )
    assert next(status for status in current.artifacts if status.artifact_id == "posterior").stale
    assert read_model_snapshot("BINDINGS", at_seq=2) == historical


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
                {"indicator_id": "indicator:8ab0e6245f029d222a9a", "value": observed}
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
    spec = store.write_version(
        "statistical_model_spec",
        provenance="computed",
        derived_from={"panel": 1, "validation_report": 1},
        produced_by="run:statistical_model_spec",
        json_files={
            "statistical_model_spec.json": {
                "statistical_model_spec": {
                    "mechanisms": [],
                    "likelihoods": [
                        {
                            "indicator_id": "indicator:8ab0e6245f029d222a9a",
                            "distribution": family,
                            "link": link,
                            "reasoning": "Test",
                        }
                    ],
                    "parameters": [],
                },
                "prior_predictive_samples": {"indicator:8ab0e6245f029d222a9a": prior},
            }
        },
    )
    state = EpisodeState().with_versions([panel, validation, spec])
    view = read_artifact_views(store, state, {}).statistical_model_spec
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
                {"indicator_id": "indicator:8ab0e6245f029d222a9a", "value": [100.0]}
            )
        },
    )
    revised = read_artifact_views(
        store, state.with_versions([current_panel]), {}
    ).statistical_model_spec
    assert revised is not None
    assert revised.likelihood_diagnostics == {}


@pytest.mark.parametrize("changed_artifact", ["measurement_structure", "causal_design"])
def test_measurement_view_reads_one_design_and_requires_compatible_versions(
    monkeypatch, tmp_path, changed_artifact
):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("MEASUREMENT")
    dependencies: dict[ArtifactId, dict[ArtifactId, int]] = {
        "measurement_structure": {},
        "causal_design": {"measurement_structure": 1},
        "structural_plan": {"causal_design": 1},
    }
    versions = []
    for aid, pins in dependencies.items():
        versions.append(
            store.write_version(
                aid,
                provenance="computed",
                derived_from=pins,
                produced_by=f"run:{aid}",
                json_files={
                    next(iter(artifact_file_spec(aid).json.values())): json.loads(
                        (FIXTURE / f"{aid}.json").read_text()
                    )
                },
            )
        )
    state = EpisodeState().with_versions(versions)
    view = read_artifact_views(store, state, {}).measurement_structure
    assert view is not None
    assert set(view.model_dump()) == {"causal_design", "structural_plan"}
    measurement = json.loads((FIXTURE / "measurement_structure.json").read_text())
    assert (
        view.causal_design.measurement.model_dump(mode="json")
        == measurement["measurement_structure"]
    )

    changed = store.write_version(
        changed_artifact,
        provenance="computed",
        derived_from=dependencies[changed_artifact],
        produced_by=f"run:{changed_artifact}",
        json_files={
            next(iter(artifact_file_spec(changed_artifact).json.values())): json.loads(
                (FIXTURE / f"{changed_artifact}.json").read_text()
            )
        },
    )
    assert (
        read_artifact_views(store, state.with_versions([changed]), {}).measurement_structure is None
    )
