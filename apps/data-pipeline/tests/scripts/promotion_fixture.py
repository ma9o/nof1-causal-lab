"""Small native Git study for the fixture promotion contract tests; no inference."""

import json
import sys
from pathlib import Path

import pyarrow as pa

from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.utils import data as data_module


def seed(root, workspace, options):
    data_module._DATA_URI = root
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    stamp = "2026-08-07T00:00:00Z"
    directory = Path(root) / workspace
    (directory / "access.json").write_text('{"version": 1}')
    for name in ["input/bundle.zip", "scratch/discard.json", "cache/discard.json"]:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture input")

    def write(aid, pins=None, producer=None, **files):
        return store.write_artifact(aid, derived_from=pins or {}, produced_by=producer, **files)

    panel = write("panel", parquet_files={"panel.parquet": pa.table({"value": [1]})})
    old_panel = write("panel", parquet_files={"panel.parquet": pa.table({"value": [2]})})
    model = write(
        "model",
        {"panel": panel.revision},
        "run:posterior",
        json_files={"model.json": {"question": "Fixture question"}},
    )
    artifacts = [
        write(
            "raw_data",
            producer="run:raw_data",
            parquet_files={"raw.parquet": pa.table({"value": [1]})},
        ),
        model,
        write(
            "identification_report",
            {"model": model.revision},
            json_files={"identification_report.json": {}},
        ),
        panel,
        write(
            "validation_report",
            {
                "model": model.revision,
                "panel": old_panel.revision if options.get("staleValidation") else panel.revision,
            },
            json_files={"validation_report.json": {}},
        ),
    ]

    def append(seq, operation, produced, trace=None, diagnostics=None):
        repository.append(
            TransitionRecord(
                seq=seq,
                ts=stamp,
                action="fit"
                if operation == "posterior"
                else "prepare_data"
                if operation in {"raw_data", "measurements"}
                else "edit_model",
                operation_id=operation,
                inputs={},
                status="applied",
                produced=produced,
                diagnostics=diagnostics or {},
                trace_ids=[trace] if trace else [],
                resume=None,
            ),
            logs={
                f"traces/{trace}.json": json.dumps({"artifact": operation, "trace": trace}).encode()
            }
            if trace
            else {},
        )

    for seq, artifact in enumerate(artifacts, 1):
        if artifact.artifact_id == options.get("omit"):
            continue
        operation = (
            "posterior"
            if artifact.artifact_id == "model"
            else "raw_data"
            if artifact.artifact_id == "raw_data"
            else "measurements"
        )
        append(
            seq,
            operation,
            [artifact],
            "raw-data" if operation == "raw_data" else None,
            {"report": {"inference_metadata": {"method": "test"}}}
            if operation == "posterior"
            else None,
        )
    for seq, (operation, trace) in enumerate(
        [
            ("latent_structure", "latent-structure"),
            ("measurement_structure", "measurement-structure"),
            ("measurements", "measurement-chunk-000000-attempt-001"),
            ("statistical_model_spec", "model-spec-sleep-attempt-001"),
        ],
        101,
    ):
        append(
            seq,
            operation,
            [],
            trace,
            {
                "search_queries": {"parameter:test": "prior study"},
                "validation_diagnostics": [],
                "prior_predictive": {"samples": {"indicator:test": [0.5]}, "diagnostics": []},
            }
            if operation == "statistical_model_spec"
            else None,
        )


if __name__ == "__main__":
    seed(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
