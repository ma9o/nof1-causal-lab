"""Versioned artifact store + transition log."""

import polars as pl
import pytest

from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import AttemptRecord
from nof1_causal_lab.study.state import RetractedArtifact
from nof1_causal_lab.study.store import ArtifactStore, read_current_state
from tests.git_fixtures import git_oid
from tests.helpers import make_model

pytestmark = pytest.mark.contract


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


class TestArtifactStore:
    def test_artifacts_are_immutable_git_trees(self, workspace):
        store = ArtifactStore(workspace)
        first = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"question": "does exercise improve sleep?"}},
        )
        second = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"question": "does caffeine harm sleep?"}},
        )
        assert first.revision != second.revision
        assert all(len(info.revision) == 40 for info in (first, second))
        assert not list(__import__("pathlib").Path(store._root).glob("*/v*"))
        assert store.list_revisions("model") == [first.revision, second.revision]
        # Old revision stays readable — nothing is overwritten.
        assert store.read_json_file("model", first.revision, "model.json")["question"].startswith(
            "does exercise"
        )
        assert store.read_json_file("model", second.revision, "model.json")["question"].startswith(
            "does caffeine"
        )

    def test_meta_roundtrip(self, workspace):
        store = ArtifactStore(workspace)
        info = store.write_artifact(
            "model",
            derived_from={"raw_data": git_oid(10)},
            produced_by="edit_model",
            json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
        )
        loaded = store.read_meta("model", info.revision)
        assert loaded == info
        assert loaded.derived_from["raw_data"] == git_oid(10)
        assert loaded.created_at

    def test_parquet_payload(self, workspace):
        store = ArtifactStore(workspace)
        df = pl.DataFrame({"indicator": ["mood"], "value": [3.5]})
        info = store.write_artifact(
            "panel",
            derived_from={},
            produced_by="prepare_data",
            parquet_files={"panel.parquet": df},
        )
        loaded_df = store.read_parquet_file("panel", info.revision, "panel.parquet")
        assert loaded_df.equals(df)

    def test_empty_artifact_has_no_versions(self, workspace):
        store = ArtifactStore(workspace)
        assert store.list_revisions("model") == []


class TestStudyRepository:
    def _record(self, seq, action, status="applied", **kwargs):
        kwargs.setdefault("trace_ids", [])
        return AttemptRecord(
            seq=seq,
            ts="2026-07-03T00:00:00+00:00",
            action=action,
            status=status,
            **kwargs,
        )

    def test_append_and_read_back_in_order(self, workspace):
        journal = StudyRepository(workspace)
        journal.append(self._record(1, "edit_model"))
        journal.append(
            self._record(
                2,
                "edit_model",
                status="rejected",
                reason=(
                    "measurement_structure requires artifacts that do not exist: "
                    "raw_data, latent_structure"
                ),
            )
        )
        journal.append(
            self._record(
                3,
                "fit",
                status="raised",
                error_type="ModelFitError",
                error_message="sampler diverged",
                diagnostics={"rhat_max": 2.4},
            )
        )
        records = journal.attempts()
        assert [r.seq for r in records] == [1, 2, 3]
        assert records[1].status == "rejected"
        assert records[1].reason is not None
        assert "raw_data" in records[1].reason
        assert records[2].error_type == "ModelFitError"
        assert records[2].diagnostics["rhat_max"] == 2.4
        # Scientific action identifiers round-trip.
        assert records[0].action == "edit_model"
        assert records[2].action == "fit"

    def test_identical_duplicate_seq_is_idempotent(self, workspace):
        journal = StudyRepository(workspace)
        record = self._record(1, "edit_model")
        journal.append(record)
        journal.append(record)
        assert [
            item.model_dump(exclude={"commit_id", "parent_ids"}) for item in journal.attempts()
        ] == [record.model_dump()]

    def test_different_duplicate_seq_refused(self, workspace):
        journal = StudyRepository(workspace)
        journal.append(self._record(1, "edit_model"))
        with pytest.raises(FileExistsError):
            journal.append(self._record(1, "prepare_data"))

    def test_latest_seq_reads_max_entry_without_state_manifest(self, workspace):
        journal = StudyRepository(workspace)
        assert journal.latest_seq() == 0
        journal.append(self._record(3, "edit_model"))
        assert journal.latest_seq() == 3


class TestDerivedCurrentState:
    def _append(self, workspace, seq, action, *, produced=None, retracted=None, status="applied"):
        StudyRepository(workspace).append(
            AttemptRecord(
                seq=seq,
                ts="2026-07-03T00:00:00+00:00",
                action=action,
                status=status,
                produced=produced or [],
                retracted=retracted or [],
                trace_ids=[],
            )
        )

    def test_current_state_replays_only_applied_versions(self, workspace):
        store = ArtifactStore(workspace)
        first = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"question": "first"}},
        )
        self._append(
            workspace,
            1,
            "edit_model",
            produced=[first],
        )
        second = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"question": "second"}},
        )

        # Persisting a revision is not the commit boundary. Until an applied
        # transition records it, readers continue to see the prior state.
        assert read_current_state(workspace).get("model") == first

        self._append(
            workspace,
            2,
            "edit_model",
            produced=[second],
        )

        assert read_current_state(workspace).get("model") == second

    def test_rejected_and_raised_effects_are_not_current(self, workspace):
        store = ArtifactStore(workspace)
        rejected = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"question": "rejected"}},
        )
        raised = store.write_artifact(
            "raw_data",
            derived_from={},
            produced_by="prepare_data",
        )
        self._append(
            workspace,
            1,
            "edit_model",
            produced=[rejected],
            status="rejected",
        )
        self._append(
            workspace,
            2,
            "prepare_data",
            produced=[raised],
            status="raised",
        )

        assert read_current_state(workspace).current == {}

    def test_applied_retraction_removes_optional_output(self, workspace):
        store = ArtifactStore(workspace)
        panel_v1 = store.write_artifact(
            "panel",
            derived_from={},
            produced_by="prepare_data",
            parquet_files={"panel.parquet": pl.DataFrame({"indicator": ["m"], "value": [1.0]})},
        )
        self._append(
            workspace,
            1,
            "prepare_data",
            produced=[panel_v1],
        )

        state_with_panel = read_current_state(workspace)
        assert state_with_panel.get("panel") == panel_v1

        self._append(
            workspace,
            2,
            "prepare_data",
            produced=[],
            retracted=[
                RetractedArtifact(
                    artifact_id="panel",
                    reason_ref="measurements.produces_optional.panel",
                )
            ],
        )

        state_without_panel = read_current_state(workspace)
        assert state_without_panel.get("panel") is None
