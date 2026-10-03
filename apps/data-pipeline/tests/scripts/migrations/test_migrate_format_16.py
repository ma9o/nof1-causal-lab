"""Availability conversion preserves stored evidence, ancestry and external bytes."""

import json
from datetime import UTC, date, datetime

import pygit2
import pytest
from scripts.migrations.migrate_format_16 import convert_payload, migrate

from nof1_causal_lab.artifacts.availability import NotApplicable
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.simulation import (
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.study.git_objects import open_repository, read_file, write_tree

pytestmark = pytest.mark.contract


def test_migration_rewrites_a_new_copy_and_keeps_recorded_histories(tmp_path):
    source, destination = tmp_path / "source/STUDY", tmp_path / "converted/STUDY"
    repo = open_repository("STUDY", source / "study/history.git")
    repo.config["nof1.format"] = 15
    report = SimulationReport(
        model=GitRef(workspace_id="STUDY", revision="a" * 40, path="model.json"),
        design=SimulationSpec(start=date(2026, 1, 1), horizon="1d"),
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        assignments=(),
        times=(0, 1),
        draws=2,
        seed=0,
        state_ids=(),
        parameter_draws={},
        latent_paths="action.npy",
        observations="observations.npy",
        observation_layout=SimulationObservationLayout(
            variables=(),
            support_start_times="starts.npy",
            support_end_times="ends.npy",
            mask="mask.npy",
        ),
        fit_reliability="not_fitted",
        causal=NotApplicable(reason="No intervention was requested."),
    )
    historical = report.model_dump(mode="json")
    historical.pop("causal")
    historical.update(causal_result=None, causal_unavailable_reason=None)
    root = repo.head.target
    signature = pygit2.Signature("Test", "test@example.com", 1, 0)
    commit = repo.create_commit(
        "refs/heads/main",
        signature,
        signature,
        "Retained simulation",
        write_tree(repo, {"logs/simulation.json": json.dumps(historical).encode()}),
        [root],
    )
    repo.references.create("refs/heads/review", commit)
    array = source / "store/arrays/action.npy"
    array.parent.mkdir(parents=True)
    array.write_bytes(b"retained draw bytes")
    mapping = migrate(source, destination)
    converted = pygit2.Repository(str(destination / "study/history.git"))
    moved = mapping[str(commit)]
    restored = SimulationReport.model_validate_json(
        read_file(converted, moved, "logs/simulation.json")
    )
    assert restored == report
    assert converted.config.get_int("nof1.format") == 16
    assert str(converted.references["refs/heads/review"].target) == moved
    assert converted[pygit2.Oid(hex=moved)].peel(pygit2.Commit).parent_ids == [
        pygit2.Oid(hex=mapping[str(root)])
    ]
    assert (destination / "store/arrays/action.npy").read_bytes() == array.read_bytes()
    assert repo.config.get_int("nof1.format") == 15
    assert repo.head.target == commit
    assert json.loads(read_file(repo, str(commit), "logs/simulation.json")) == historical


@pytest.mark.parametrize(
    "payload",
    [
        {
            "design": {"interventions": [{}]},
            "latent_paths": "draws",
            "causal_result": {},
            "causal_unavailable_reason": "Unavailable",
        },
        {"columns": [{}], "unavailable_reason": "Unavailable"},
        {
            "indicator_id": "indicator:x",
            "left": [],
            "right": [],
            "reference_side": None,
            "predictive_checks": {},
            "predictive_unavailable_reason": None,
        },
    ],
)
def test_migration_refuses_contradictory_retained_availability(payload):
    with pytest.raises(ValueError, match=r"cannot|require"):
        convert_payload(payload)
