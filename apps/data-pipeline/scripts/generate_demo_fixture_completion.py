"""Compose DEMO read fixtures from retained canonical artifacts.

The illustrative inference report is a fixed fixture input.
The original joint samples were not retained; the report stays in the log, and
this fixture deliberately does not invent a conditioned ModelSpec. This
command validates their scientific references.
It regenerates presentation reads through
the production reader, without fitting or generating new inference results. Prior plot viewports use
a small deterministic native draw; curve densities come from NumPyro.

Run ``bun run fixture:demo`` or ``bun run fixture:demo:check`` from the repo root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pyarrow.parquet as pq

from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.artifact_files import artifact_file_spec
from nof1_causal_lab.machine.moves import Move, RunOperation, WriteArtifact
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.views import read_artifact_views
from nof1_causal_lab.utils import data as data_module

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId

DEMO_ROOT = Path(__file__).resolve().parents[3] / "data" / "DEMO"
ARTIFACT_ROOT = DEMO_ROOT / "fixture" / "artifacts"
STAMP = "2026-07-08T12:00:00Z"


def build_outputs():
    payloads = {
        cast("ArtifactId", path.stem): json.loads(path.read_text())
        for path in ARTIFACT_ROOT.glob("*.json")
    }
    model = ModelSpec.model_validate(payloads["model"])
    measured = model.revised(
        edges=replace_constructs(
            tuple(edge.model_copy(update={"mechanisms": ()}) for edge in model.edges),
            tuple(
                construct.model_copy(
                    update={
                        "dynamics": (),
                        "coefficients": (),
                        "indicators": tuple(
                            indicator.model_copy(update={"likelihood": None})
                            for indicator in construct.indicators
                        ),
                    }
                )
                for construct in model.constructs
            ),
        ),
        parameters=(),
        distributions={},
    )
    proposed = measured.revised(
        edges=replace_constructs(
            measured.edges,
            tuple(
                construct.model_copy(update={"indicators": ()}) for construct in measured.constructs
            ),
        ),
        measurement_clock=None,
    )
    for aid, value in payloads.items():
        ARTIFACT_CONTRACTS[aid].model_validate(value)
    model.require_measurements()
    tables = {
        "raw_data": pq.read_table(DEMO_ROOT / "store/raw_data/v1/raw.parquet"),
        "panel": pq.read_table(DEMO_ROOT / "store/panel/v1/panel.parquet"),
    }
    # Each operation commits an enriched version of the same ModelSpec. Execution
    # artifacts record findings; posterior draws retain scientific identities.
    groups: list[
        tuple[int, Move, list[tuple[ArtifactId, dict[ArtifactId, int], ModelSpec | None]]]
    ] = [
        (1, RunOperation(operation_id="raw_data"), [("raw_data", {}, None)]),
        (
            2,
            WriteArtifact(artifact_id="model", expected_model_version=0),
            [("model", {}, ModelSpec(question=model.question))],
        ),
        (3, RunOperation(operation_id="latent_structure"), [("model", {"model": 1}, proposed)]),
        (
            4,
            RunOperation(operation_id="measurement_structure"),
            [
                ("model", {"raw_data": 1, "model": 2}, measured),
                ("identification_report", {"model": 3}, None),
            ],
        ),
        (
            5,
            RunOperation(operation_id="measurements"),
            [
                ("panel", {"raw_data": 1, "model": 3}, None),
                ("validation_report", {"model": 3, "panel": 1}, None),
            ],
        ),
        (
            7,
            RunOperation(operation_id="statistical_model_spec"),
            [
                (
                    "model",
                    {"model": 3, "panel": 1, "validation_report": 1},
                    model,
                ),
            ],
        ),
        (
            8,
            RunOperation(operation_id="posterior"),
            [],
        ),
    ]
    with (
        TemporaryDirectory(prefix="nof1-model-fixture-") as directory,
        patch.object(data_module, "_DATA_URI", directory),
    ):
        store, journal = ArtifactStore("DEMO"), EpisodeJournal("DEMO")
        for seq, move, entries in groups:
            produced = []
            for aid, pins, definition in entries:
                files = artifact_file_spec(aid)
                value = (
                    definition.model_dump(mode="json")
                    if definition is not None
                    else payloads.get(aid)
                )
                info = store.write_version(
                    aid,
                    provenance="computed",
                    produced_by=f"run:{move.operation_id}"
                    if isinstance(move, RunOperation)
                    else None,
                    derived_from=pins,
                    json_files={next(iter(files.json.values())): value} if files.json else None,
                    parquet_files={next(iter(files.parquet.values())): tables[aid]}
                    if files.parquet
                    else None,
                )
                produced.append(info.model_copy(update={"created_at": STAMP}))
            journal.append(
                TransitionRecord(
                    seq=seq,
                    ts=STAMP,
                    move=move,
                    status="applied",
                    diagnostics={
                        **json.loads((DEMO_ROOT / "episode/journal/000005.json").read_text())[
                            "diagnostics"
                        ],
                        "input_pins": produced[0].derived_from,
                        "model_input": produced[0].consumed_model_inputs["extraction"],
                    }
                    if isinstance(move, RunOperation) and move.operation_id == "measurements"
                    else json.loads((DEMO_ROOT / "fixture/inference.json").read_text())
                    if isinstance(move, RunOperation) and move.operation_id == "posterior"
                    else json.loads((DEMO_ROOT / "fixture/model_authoring.json").read_text())
                    if isinstance(move, RunOperation)
                    and move.operation_id == "statistical_model_spec"
                    else {},
                    produced=produced,
                    trace_ids=[],
                    resume=None,
                )
            )
        reader = ModelReader("DEMO")
        return {
            DEMO_ROOT / "fixture/model_snapshot.json": reader.snapshot().model_dump(mode="json"),
            DEMO_ROOT / "fixture/artifact_views.json": read_artifact_views(
                reader.store, reader.state
            ).model_dump(mode="json"),
            DEMO_ROOT / "fixture/model_history.json": {
                str(seq): ModelReader("DEMO", at_seq=seq).snapshot().model_dump(mode="json")
                for seq in [0, *(seq for seq, _, _ in groups)]
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatches = []
    for path, value in build_outputs().items():
        rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.check:
            if path.read_text() != rendered:
                mismatches.append(str(path))
        else:
            path.write_text(rendered)
    if mismatches:
        raise SystemExit("Read fixtures are stale: " + ", ".join(mismatches))
    print("Validated retained artifacts and composed 3 DEMO read fixtures; no numerical runs")


if __name__ == "__main__":
    main()
