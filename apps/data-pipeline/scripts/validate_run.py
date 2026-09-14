"""Validate canonical artifact contracts and the exact revisions supporting each result.

Usage: uv run python -m scripts.validate_run --workspace-id DEMO [--up-to model] [--strict]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.raw_data import column_descriptions
from nof1_causal_lab.machine.artifact_files import ARTIFACT_FILE_SPECS, parquet_filename
from nof1_causal_lab.machine.graph import topological_artifact_order
from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa

    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.json_types import UncheckedJsonObject
    from nof1_causal_lab.machine.artifacts import EpisodeState


@dataclass(frozen=True)
class LineageIssue:
    rule: str
    severity: Literal["error", "warning"]
    artifacts: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class RunContext:
    workspace_id: str
    state: EpisodeState
    artifacts: dict[ArtifactId, UncheckedJsonObject]
    artifact_paths: dict[ArtifactId, str]
    model_indicators: set[str] | None
    raw_input_columns: set[str] | None
    raw_table: pa.Table | None = None

    def input(self, output: ArtifactId, dependency: ArtifactId) -> UncheckedJsonObject:
        """Read an original pin, even if the selected model has since changed."""
        version = self.state.current[output].derived_from[dependency]
        selected = self.state.get(dependency)
        if selected is not None and selected.version == version and dependency in self.artifacts:
            return self.artifacts[dependency]
        filename = next(iter(ARTIFACT_FILE_SPECS[dependency].json.values()))
        return ArtifactStore(self.workspace_id).read_json_file(dependency, version, filename)


def load_parquet(path: str):
    import polars as pl

    return pl.read_parquet(path, storage_options=storage.polars_storage_options())


def load_run_context(workspace_id: str, *, up_to: str | None) -> RunContext:
    order = list(topological_artifact_order())
    if up_to is not None:
        if up_to not in order:
            raise ValueError(f"Unknown artifact {up_to!r}. Expected one of: {', '.join(order)}")
        order = order[: order.index(up_to) + 1]
    state = derive_current_state(workspace_id)
    store = ArtifactStore(workspace_id)
    artifacts = {}
    paths = {}
    for artifact_id in order:
        info = state.get(artifact_id)
        files = ARTIFACT_FILE_SPECS[artifact_id].json
        if info is None or not files:
            continue
        filename = next(iter(files.values()))
        artifacts[artifact_id] = store.read_json_file(artifact_id, info.version, filename)
        paths[artifact_id] = store.file_path(artifact_id, info.version, filename)
    model_indicators = None
    if "panel" in order and (panel := state.get("panel")) is not None:
        frame = load_parquet(
            store.file_path("panel", panel.version, parquet_filename("panel", "panel"))
        )
        model_indicators = set(frame["indicator_id"].unique().to_list())
    raw_columns = None
    raw_table = None
    if "raw_data" in order and (raw := state.get("raw_data")) is not None:
        raw_table = store.read_parquet_table(
            "raw_data", raw.version, parquet_filename("raw_data", "raw")
        )
        raw_columns = set(raw_table.column_names)
    return RunContext(
        workspace_id, state, artifacts, paths, model_indicators, raw_columns, raw_table
    )


def rule_contract_conformance(ctx: RunContext) -> list[LineageIssue]:
    issues = []
    if ctx.raw_table is not None:
        try:
            column_descriptions(ctx.raw_table)
        except ValueError as exc:
            issues.append(LineageIssue("contract-conformance", "error", ("raw_data",), str(exc)))
    for artifact_id, payload in ctx.artifacts.items():
        try:
            ARTIFACT_CONTRACTS[artifact_id].model_validate(payload)
        except ValidationError as exc:
            issues.append(LineageIssue("contract-conformance", "error", (artifact_id,), str(exc)))
    return issues


def rule_source_columns_in_raw_data(ctx: RunContext) -> list[LineageIssue]:
    if ctx.raw_input_columns is None or "model" not in ctx.artifacts:
        return []
    model = ModelSpec.model_validate(ctx.artifacts["model"])
    unknown = {
        i.id: sorted(set(i.source_columns) - ctx.raw_input_columns)
        for i in model.indicators
        if set(i.source_columns) - ctx.raw_input_columns
    }
    return (
        [
            LineageIssue(
                "source-columns-in-raw-data",
                "error",
                ("model", "raw_data"),
                f"Indicator source columns absent from raw parquet: {unknown}",
            )
        ]
        if unknown
        else []
    )


def rule_indicators_in_panel(ctx: RunContext) -> list[LineageIssue]:
    if not ctx.state.has("panel") or ctx.model_indicators is None:
        return []
    model = ModelSpec.model_validate(ctx.input("panel", "model"))
    missing = {i.id for i in model.indicators} - ctx.model_indicators
    return (
        [
            LineageIssue(
                "indicators-in-panel",
                "warning",
                ("model", "panel"),
                f"No extracted observations for declared indicators: {sorted(missing)}",
            )
        ]
        if missing
        else []
    )


def rule_indicators_audited_by_validation_report(ctx: RunContext) -> list[LineageIssue]:
    if "validation_report" not in ctx.artifacts:
        return []
    model = ModelSpec.model_validate(ctx.input("validation_report", "model"))
    missing = {i.id for i in model.indicators} - ctx.artifacts["validation_report"][
        "indicators"
    ].keys()
    return (
        [
            LineageIssue(
                "indicators-audited-by-validation-report",
                "error",
                ("model", "validation_report"),
                f"Unaudited indicators: {sorted(missing)}",
            )
        ]
        if missing
        else []
    )


def rule_pinned_model_contracts(ctx: RunContext) -> list[LineageIssue]:
    if "identification_report" in ctx.artifacts:
        IdentificationReport.model_validate(ctx.artifacts["identification_report"]).validate_model(
            ModelSpec.model_validate(ctx.input("identification_report", "model"))
        )
    if "model" in ctx.artifacts:
        ModelSpec.model_validate(ctx.artifacts["model"])
    return []


def rule_posterior_bindings(ctx: RunContext) -> list[LineageIssue]:
    from nof1_causal_lab.machine.inference import inference_report_record
    from nof1_causal_lab.machine.store import EpisodeJournal
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    record = inference_report_record(EpisodeJournal(ctx.workspace_id).read_all(), ctx.state)
    if record is None:
        return []
    posterior = InferenceReport.model_validate(record.diagnostics["report"])
    model = ModelSpec.model_validate(ctx.artifacts["model"])
    allowed = {
        (b.parameter_id, element) for b in parameter_bindings(model)[0] for element in b.elements
    }
    subjects = [item.subject for item in posterior.posterior_marginals or []]
    subjects.extend(
        subject
        for pair in posterior.posterior_pairs or []
        for subject in (pair.subject_x, pair.subject_y)
    )
    unknown = {(item.parameter_id, item.element_id) for item in subjects} - allowed
    return (
        [
            LineageIssue(
                "posterior-bindings",
                "error",
                ("model",),
                f"Posterior subjects absent from the scientific model: {sorted(unknown)}",
            )
        ]
        if unknown
        else []
    )


RULES: list[Callable[[RunContext], list[LineageIssue]]] = [
    rule_source_columns_in_raw_data,
    rule_indicators_in_panel,
    rule_indicators_audited_by_validation_report,
    rule_pinned_model_contracts,
    rule_posterior_bindings,
]


def validate_context(ctx: RunContext) -> list[LineageIssue]:
    issues = rule_contract_conformance(ctx)
    if issues:
        return issues
    for rule in RULES:
        try:
            issues.extend(rule(ctx))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            issues.append(LineageIssue("artifact_lineage", "error", (), str(exc)))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate cross-artifact artifact lineage of a pipeline run"
    )
    parser.add_argument(
        "--workspace-id",
        required=True,
        help="Workspace ID under data/ (e.g. DEMO)",
    )
    parser.add_argument(
        "--up-to",
        default=None,
        help="Validate only up to this artifact (e.g. model). Default: all present artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit on any issue).",
    )
    args = parser.parse_args(argv)

    print(f"Validating run: workspace={args.workspace_id}")

    try:
        ctx = load_run_context(args.workspace_id, up_to=args.up_to)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not ctx.artifacts and ctx.raw_table is None:
        print("error: no current artifact result artifacts found in episode state", file=sys.stderr)
        return 1

    print(f"Artifacts found: {', '.join(ctx.artifacts)}")
    if ctx.model_indicators is not None:
        print(f"Panel indicators: {len(ctx.model_indicators)} unique")
    else:
        print("Panel indicators: (panel/panel.parquet not found)")

    issues = validate_context(ctx)

    if not issues:
        print("\nLineage: OK")
        return 0

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    print(f"\nFound {len(errors)} error(s) and {len(warnings)} warning(s):")
    for issue in issues:
        prefix = "ERROR" if issue.severity == "error" else "WARN "
        scope = "+".join(issue.artifacts)
        print(f"  [{prefix}] {issue.rule} ({scope}): {issue.message}")

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
