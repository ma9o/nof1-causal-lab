"""Build DEMO and workbench fixtures from the saved bundle, or project a study.

``bun run fixture:build`` regenerates every fixture from the authoritative Git
bundle and blobs; ``bun run fixture:check`` verifies them without writing.
The ``project`` subcommand supports workspace promotion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import patch

from nof1_causal_lab.artifacts.expressions import (
    coefficient,
)
from nof1_causal_lab.artifacts.expressions import state as expr_state
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.models.model_structure import (
    StructuralSelection,
    compare_model_graph,
    compare_parameters,
    model_graph_entities,
)
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.state import is_stale
from nof1_causal_lab.study.store import ArtifactStore, trace_log_path
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.study.state import StudyState

ROOT = Path(__file__).resolve().parents[4]
DEMO_ROOT = ROOT / "data" / "DEMO"
WORKBENCH_OUTPUT = ROOT / "apps/web/src/components/__fixtures__/workbench-comparisons.json"
SIMULATION_REPORTS = ROOT / "apps/web/src/components/dag/__fixtures__/simulation-reports.json"

ARTIFACTS: dict[ArtifactId, str] = {
    "model": "model.json",
}
TRACES = {
    "raw_data": "raw-data",
    "latent_structure": "latent-structure",
    "measurement_structure": "measurement-structure",
    "measurements": "measurement-chunk-",
    "statistical_model_spec": "model-spec-",
}


def read_fixture_files(repository: StudyRepository, state: StudyState) -> dict[str, bytes]:
    """Project retained payloads and logs from the selected Git ancestry."""
    records = repository.records(repository.head())
    store = ArtifactStore(repository.workspace_id)
    files = {}
    for aid, filename in ARTIFACTS.items():
        files[f"artifacts/{aid}.json"] = repository.read_file(state.current[aid].revision, filename)
    for name, prefix in TRACES.items():
        trace, record = next(
            (trace, record)
            for record in reversed(records)
            for trace in sorted(record.record.trace_ids)
            if trace.startswith(prefix)
        )
        raw_trace = repository.read_file(record.commit_id, f"logs/{trace_log_path(trace)}")
        files[f"traces/{name}.json"] = (
            LLMTrace.model_validate_json(raw_trace).model_dump_json(indent=2) + "\n"
        ).encode()
    reader = ModelReader(repository.workspace_id, at=repository.head())
    identification = reader.identification()
    if identification is not None:
        files["artifacts/identification_report.json"] = (identification.value.model_dump_json(indent=2) + "\n").encode()
    if reader.validation_report is not None:
        files["artifacts/validation_report.json"] = (reader.validation_report.value.model_dump_json(indent=2) + "\n").encode()
    report = reader.inference_report
    files["inference.json"] = (report.value.model_dump_json(indent=2) + "\n").encode() if report is not None else b"null\n"
    predictive = reader.checks[0].predictive if reader.checks is not None else None
    checks = predictive.evaluation.predictive_checks if predictive is not None and predictive.evaluation.kind == "evaluated" else None
    files["predictive_checks.json"] = (checks.model_dump_json(indent=2) + "\n").encode() if checks is not None else b"null\n"
    # Check external payload closure as well as the native Git objects.
    for aid, info in state.current.items():
        for filename in store.filenames(aid, info.revision):
            store.file_path(aid, info.revision, filename)
    return files


def project(source: Path, destination: Path | None = None):
    data_module._DATA_URI = str(source.parent)
    repository = StudyRepository(source.name)
    state = repository.state(repository.head())
    required: tuple[ArtifactId, ...] = (
        "raw_data",
        "model",
        "panel",
    )
    missing = [aid for aid in required if not state.has(aid)]
    if missing:
        raise ValueError(
            f"Source workspace is incomplete; missing current artifacts: {', '.join(missing)}."
        )
    stale = [aid for aid in required if is_stale(state, aid)]
    if stale:
        raise ValueError(f"Source workspace has stale current artifacts: {', '.join(stale)}.")
    files = read_fixture_files(repository, state)
    if destination is not None:
        for name, content in files.items():
            path = destination / "fixture" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(
            [
                "git",
                "--git-dir",
                str(destination / "study" / "history.git"),
                "bundle",
                "create",
                str(destination / "study" / "history.bundle"),
                "--all",
            ],
            check=True,
            capture_output=True,
        )
    return {"artifacts": list(ARTIFACTS), "traces": list(TRACES)}


def workbench_comparisons(snapshot, history):
    free = ModelSpec.model_validate(snapshot["model"]["value"])
    parameter = next(
        item
        for item in free.parameters
        if item.name == "beta_perceived_stress_burden_internalizing_symptom_burden"
    )
    edge = next(item for item in free.edges if item.id == "edge:9df1507c29b9de944a33")
    pinned = free.revised(
        edges=tuple(
            item.revised(
                mechanisms=tuple(
                    mechanism.revised(
                        expression=coefficient(0.0, "weight") * expr_state(edge.cause.id)
                    )
                    for mechanism in item.mechanisms
                )
            )
            if item.id == edge.id
            else item
            for item in free.edges
        ),
        parameters=tuple(item for item in free.parameters if item.id != parameter.id),
        distributions={
            key: law for key, law in free.distributions.items() if key != parameter.distribution
        },
    )
    models: dict[str, ModelSpec | None] = {
        history[str(seq)]["state"]["current"]["model"]["revision"]: ModelSpec.model_validate(
            history[str(seq)]["model"]["value"]
        )
        for seq in (2, 3, 4, 7)
    }
    models["no-model"] = None
    models.update(
        {format(n, "x").rjust(40, "a"): model for n, model in [(5, free), (6, free), (7, pinned)]}
    )
    # The study's one question scopes every compared revision alike.
    question = QuestionSpec.model_validate(snapshot["question"]["value"])
    comparisons = {}
    for before_version, left in models.items():
        for after_version, right in models.items():
            parameters = compare_parameters(left, right)
            before = input_fingerprints(left) if left is not None else {}
            after = input_fingerprints(right) if right is not None else {}
            scoped = (
                StructuralSelection.for_question(left, question) if left is not None else None,
                StructuralSelection.for_question(right, question) if right is not None else None,
            )
            constructs, edges = compare_model_graph(*scoped)
            graphs = tuple(model_graph_entities(selection) if selection is not None else ((), ()) for selection in scoped)
            comparisons[f"{before_version}:{after_version}"] = {
                "parameters": [item.model_dump(mode="json") for item in parameters],
                "constructs": [item.model_dump(mode="json") for item in constructs],
                "edges": [item.model_dump(mode="json") for item in edges],
                "before_dispositions": [
                    item.model_dump(mode="json") for item in scoped[0].structural_dispositions
                ]
                if scoped[0] is not None and scoped[0].model.measurement_clock is not None and scoped[0].model.indicators
                else [],
                "after_dispositions": [
                    item.model_dump(mode="json") for item in scoped[1].structural_dispositions
                ]
                if scoped[1] is not None and scoped[1].model.measurement_clock is not None and scoped[1].model.indicators
                else [],
                "before_dynamic_construct_ids": [
                    item.id for item in graphs[0][0] if item.is_dynamic
                ],
                "after_dynamic_construct_ids": [
                    item.id for item in graphs[1][0] if item.is_dynamic
                ],
                "beforeModel": left.model_dump(mode="json") if left is not None else None,
                "afterModel": right.model_dump(mode="json") if right is not None else None,
                "changed_inputs": [
                    key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)
                ],
            }
    return {
        "pinned_model": pinned.model_dump(mode="json"),
        "pinned_inputs": input_fingerprints(pinned),
        "comparisons": comparisons,
    }


def build_outputs():
    # A clean restore proves the fixture does not depend on a developer's local
    # repository, generated JSON projections or retired numbered directories.
    with (
        TemporaryDirectory(prefix="nof1-model-fixture-") as directory,
        patch.object(data_module, "_DATA_URI", directory),
    ):
        workspace = Path(directory) / "DEMO"
        history = workspace / "study/history.git"
        history.parent.mkdir(parents=True)
        subprocess.run(
            ["git", "clone", "--mirror", str(DEMO_ROOT / "study/history.bundle"), str(history)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "--git-dir", str(history), "config", "nof1.format", "18"],
            check=True,
            capture_output=True,
        )
        shutil.copytree(DEMO_ROOT / "store", workspace / "store")
        repository = StudyRepository("DEMO")
        reader = ModelReader("DEMO", at=repository.head())
        commits = {0: reader.records[0].parent_ids[0]}
        commits.update({record.record.seq: record.commit_id for record in reader.records})
        outputs = {
            DEMO_ROOT / "fixture" / name: json.loads(content)
            for name, content in read_fixture_files(repository, reader.state).items()
        }
        outputs.update(
            {
                DEMO_ROOT / "fixture/model_snapshot.json": reader.snapshot().model_dump(
                    mode="json"
                ),
                DEMO_ROOT / "fixture/model_history.json": {
                    str(seq): ModelReader("DEMO", at=commit).snapshot().model_dump(mode="json")
                    for seq, commit in commits.items()
                },
            }
        )
        outputs[WORKBENCH_OUTPUT] = workbench_comparisons(
            outputs[DEMO_ROOT / "fixture/model_snapshot.json"],
            outputs[DEMO_ROOT / "fixture/model_history.json"],
        )
        from scripts.fixtures.visuals import workbench_visuals

        outputs[WORKBENCH_OUTPUT.with_name("workbench-visuals.json")] = workbench_visuals(
            reader, json.loads(WORKBENCH_OUTPUT.with_name("workbench-simulation.json").read_text())
        )
        from nof1_causal_lab.artifacts.simulation import SimulationReport

        outputs[SIMULATION_REPORTS] = [
            SimulationReport.model_validate(value).model_dump(mode="json")
            for value in json.loads(SIMULATION_REPORTS.read_text())
        ]
        return outputs


def rendered_fixtures(outputs):
    """Render fixtures and name their contracts; JSON imports otherwise widen tags/IDs.

    These declarations contain no payloads or shadow schemas. Fixture regeneration
    establishes the types at their Python producers; fixture:check owns their drift.
    """
    contracts = {
        DEMO_ROOT / "fixture/model_snapshot.json": "Domain.ModelSnapshot",
        DEMO_ROOT
        / "fixture/model_history.json": "Readonly<Partial<Record<number, Domain.ModelSnapshot>>>",
        DEMO_ROOT / "fixture/inference.json": "Domain.InferenceReport | null",
        DEMO_ROOT / "fixture/predictive_checks.json": "Domain.PosteriorPredictiveChecks | null",
        SIMULATION_REPORTS: "readonly Domain.SimulationReport[]",
        WORKBENCH_OUTPUT: """Readonly<{
  pinned_model: Domain.ModelSpec;
  pinned_inputs: Record<string, string>;
  comparisons: Readonly<Partial<Record<string, Pick<Domain.ModelDiffReport, "constructs" | "edges" | "before_dispositions" | "after_dispositions" | "before_dynamic_construct_ids" | "after_dynamic_construct_ids" | "parameters" | "changed_inputs"> & {beforeModel: Domain.ModelSpec | null; afterModel: Domain.ModelSpec | null}>>>;
}>""",
        WORKBENCH_OUTPUT.with_name("workbench-visuals.json"): """Readonly<{
  note: string;
  report: Domain.SimulationReport;
  simulation: Domain.SimulationPaths;
  observations: Readonly<Partial<Record<Domain.IndicatorId, Domain.ObservationHistory>>>;
  parameters: Domain.ParameterDraws;
}>""",
    }
    contracts.update(
        {DEMO_ROOT / f"fixture/traces/{name}.json": "Domain.LLMTrace" for name in TRACES}
    )
    contents = {
        path: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        for path, value in outputs.items()
    }
    for path, contract in contracts.items():
        assert path in outputs
        declaration = path.with_name(path.stem + ".d.json.ts")
        contents[declaration] = (
            "/** AUTO-GENERATED by fixture:build from the production fixture owner. */\n"
            'import type * as Domain from "@nof1-causal-lab/api-types";\n'
            f"declare const value: {contract};\nexport default value;\n"
        )
    for path, content in contents.items():
        if path == SIMULATION_REPORTS or path.suffix == ".ts":
            content = subprocess.run(
                ["bun", "x", "--no-install", "biome", "format", "--stdin-file-path", str(path)],
                input=content,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        yield path, content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Generate DEMO and workbench fixtures from the Git bundle"
    )
    build.add_argument("--check", action="store_true")
    projection = commands.add_parser(
        "project", help="Validate and project a workspace for promotion"
    )
    projection.add_argument("source", type=Path)
    projection.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "project":
        print(json.dumps(project(args.source.resolve(), args.destination)))
        return

    outputs = build_outputs()
    mismatches = []
    for path, rendered in rendered_fixtures(outputs):
        if args.check:
            if not path.exists() or path.read_text() != rendered:
                mismatches.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
    if mismatches:
        raise SystemExit("Fixtures are stale: " + ", ".join(mismatches))
    print(f"Restored the Git bundle and composed {len(outputs)} DEMO and workbench fixtures")


if __name__ == "__main__":
    main()
