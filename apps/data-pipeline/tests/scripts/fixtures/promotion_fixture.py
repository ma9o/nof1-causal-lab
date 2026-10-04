"""Small facts-only Git study for fixture promotion; no numerical execution."""

import json
import sys
from pathlib import Path

import polars as pl

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, DataPreparationResult
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage
from tests.action_fixtures import applied_record, question_root
from tests.data_fixtures import metadata_for_model
from tests.helpers import fixture_entity_id, make_model
from tests.integration.runner_fixtures import panel_frame


def seed(root, workspace, options):
    data_module._DATA_URI = root
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    stamp = "2026-08-07T00:00:00Z"
    directory = Path(root) / workspace
    (directory / "access.json").write_text('{"version": 1}')
    for name in ("input/bundle.zip", "scratch/discard.json", "cache/discard.json"):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture input")
    question_root(workspace)
    # Incomplete authored equations keep this projection contract non-numerical.
    model = make_model(["stress_score", "sleep_score"], [("stress_score", "sleep_score")])
    frame = panel_frame(n_days=2).with_columns(
        pl.col("indicator_id").replace_strict(
            {
                fixture_entity_id("indicator", name): next(
                    indicator.observation.id
                    for indicator in model.indicators
                    if indicator.observation.name == name + "_obs"
                )
                for name in ("stress_score", "sleep_score")
            }
        )
    )
    old_raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"raw.parquet": frame},
    )
    raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"raw.parquet": frame.with_columns((frame["value"] + 1).alias("value"))},
    )
    definition = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json", round_trip=True)},
    )
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={"raw_data": old_raw.revision if options.get("stalePanel") else raw.revision},
        json_files={
            "metadata.json": metadata_for_model(model).model_dump(mode="json", round_trip=True)
        },
        parquet_files={"panel.parquet": frame},
    )

    def append(operation, produced, trace=None):
        applied = Applied(
            result=DataPreparationResult() if operation in {"raw_data", "measurements"} else None,
            effects=ActionEffects(produced=tuple(produced)),
        )
        repository.append(
            applied_record(
                applied,
                seq=repository.latest_seq() + 1,
                ts=stamp,
                trace_ids=[trace] if trace else [],
            ),
            logs={
                f"traces/{trace}.json": LLMTrace(
                    model="fixture",
                    messages=(TraceMessage(role="assistant", content=f"{operation}: {trace}"),),
                )
                .model_dump_json(round_trip=True)
                .encode()
            }
            if trace
            else {},
        )

    for artifact, operation, trace in (
        (raw, "raw_data", "raw-data"),
        (definition, "statistical_model_spec", None),
        (panel, "measurements", None),
    ):
        if artifact.artifact_id != options.get("omit"):
            append(operation, (artifact,), trace)
    for operation, trace in (
        ("latent_structure", "latent-structure"),
        ("measurement_structure", "measurement-structure"),
        ("measurements", "measurement-chunk-000000-attempt-001"),
        ("statistical_model_spec", "model-spec-sleep-attempt-001"),
    ):
        append(operation, (), trace)


if __name__ == "__main__":
    seed(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
