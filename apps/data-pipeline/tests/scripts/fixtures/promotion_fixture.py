"""Small native Git study for the fixture promotion contract tests; no inference."""

import json
import sys
from pathlib import Path

import pyarrow as pa

from nof1_causal_lab.artifacts.checks import NotEvaluated
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.posterior import (
    InferenceMetadata,
    InferenceReport,
    InferenceReportDetail,
)
from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import DataPreparationResult, ModelEditResult, ModelFitResult
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage
from tests.action_fixtures import applied_record


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
        "fit",
        json_files={"model.json": {"question": "Fixture question"}},
    )
    artifacts = [
        write(
            "raw_data",
            producer="prepare_data",
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

    report = InferenceReport(
        time_origin=None,
        inference_metadata=InferenceMetadata(method="test", n_samples=0, duration_seconds=0),
        engine=NotEvaluated(
            subject="production_engine",
            reason="ARCHIVED_ENGINE_NOT_RETAINED",
            detail="No inference in this fixture",
        ),
        inference_diagnostics=None,
        sampler_diagnostics=None,
        convergence=parameter_convergence(None),
        detail=InferenceReportDetail(),
    )

    def append(seq, operation, produced, trace=None):
        result = (
            ModelFitResult(
                produced=tuple(produced),
                model=GitRef(workspace_id=workspace, revision=model.revision, path="model.json"),
                panel=GitRef(workspace_id=workspace, revision=panel.revision, path="panel.parquet"),
                report=report,
            )
            if operation == "posterior"
            else DataPreparationResult(produced=tuple(produced))
            if operation in {"raw_data", "measurements"}
            else ModelEditResult(produced=tuple(produced))
        )
        repository.append(
            applied_record(result, seq=seq, ts=stamp, trace_ids=[trace] if trace else []),
            logs={
                f"traces/{trace}.json": LLMTrace(
                    model="fixture",
                    messages=(TraceMessage(role="assistant", content=f"{operation}: {trace}"),),
                )
                .model_dump_json()
                .encode()
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
        )


if __name__ == "__main__":
    seed(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
