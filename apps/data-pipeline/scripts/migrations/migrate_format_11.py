"""Convert format-10 action logs into correlated format-11 outcomes, without execution.

Usage: uv run python -m scripts.migrations.migrate_format_11 SOURCE DESTINATION
The source remains read-only. Missing archived requests and measurements stay absent.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.contracts import ScientificActionRequest
from nof1_causal_lab.study.view_models import DataDiffRequest
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject, JsonValue

_REQUEST = TypeAdapter(ScientificActionRequest | DataDiffRequest)
_REQUEST_FIELDS = {
    "edit_model": {"model", "expected_revision"},
    "prepare_data": {"input"},
    "fit": {"model_revision", "panel_revision", "settings"},
    "simulate": {"model_revision", "start", "end", "interventions"},
    "data_diff": {"left", "right"},
}


def convert_attempt(
    value: JsonObject, workspace_id: str, comparison: JsonObject | None = None
) -> tuple[JsonObject, JsonObject]:
    """Return format-11 wire facts; later converters own later envelopes."""
    action = value["action"]
    inputs = value["inputs"]
    raw_diagnostics = value["diagnostics"]
    assert isinstance(action, str)
    assert isinstance(inputs, Mapping)
    assert isinstance(raw_diagnostics, Mapping)
    diagnostics = dict(raw_diagnostics)
    request = None
    retained: dict[str, JsonValue] = {}
    if _REQUEST_FIELDS[action] <= inputs.keys():
        try:
            request = _REQUEST.validate_python({"action": action, **inputs}).model_dump(mode="json")
        except ValidationError as exc:
            # A saved authoring proposal can predate the currently owned grammar.
            # Preserve its exact facts; do not silently normalize a request.
            retained["request_unavailable_reason"] = "recorded_request_outside_current_schema"
            retained["request_parse_error"] = str(exc)
    if request is None and inputs:
        retained["request_fragment"] = inputs
    common = {
        "produced": value["produced"],
        "retracted": value["retracted"],
        "checks": value.get("checks"),
    }
    status = value["status"]
    if status == "applied":
        result: dict[str, JsonValue] = {"action": action, **common}
        pins = diagnostics.pop("input_pins", {})
        assert isinstance(pins, Mapping)
        if action == "edit_model":
            revision = inputs.get("expected_revision")
            result["base"] = (
                {"workspace_id": workspace_id, "revision": revision, "path": "model.json"}
                if revision
                else None
            )
        elif action == "prepare_data":
            raw = pins.get("raw_data", inputs.get("raw_data_revision"))
            model = pins.get("model", inputs.get("model_revision"))
            preparation = inputs.get("input", {})
            assert isinstance(preparation, Mapping)
            result.update(
                raw_data={"workspace_id": workspace_id, "revision": raw, "path": "raw.parquet"}
                if raw
                else None,
                model={"workspace_id": workspace_id, "revision": model, "path": "model.json"}
                if model
                else None,
                simulation_source=preparation if "replicate" in preparation else None,
                n_observations=diagnostics.pop("n_observations", None),
                workers=diagnostics.pop("workers", []),
                ingestion_reused=diagnostics.pop("ingestion_reused", None),
                extraction_reused=diagnostics.pop("extraction_reused", None),
            )
        elif action == "fit":
            report = diagnostics.pop("report")
            result.update(
                model={
                    "workspace_id": workspace_id,
                    "revision": pins["model"] if "model" in pins else inputs["model_revision"],
                    "path": "model.json",
                },
                panel={
                    "workspace_id": workspace_id,
                    "revision": pins["panel"] if "panel" in pins else inputs["panel_revision"],
                    "path": "panel.parquet",
                },
                report=report,
                retention=diagnostics.pop("retention", "joint"),
            )
            evidence = diagnostics.pop("engine_evidence", None)
            if evidence is not None:
                assert isinstance(report, Mapping)
                engine = report["engine"]
                assert isinstance(engine, Mapping)
                if evidence != engine.get("evidence"):
                    raise ValueError("Conflicting duplicate fit engine evidence")
        elif action == "simulate":
            report = diagnostics.pop("report")
            panel = pins.get("panel")
            result.update(
                report=report,
                panel={"workspace_id": workspace_id, "revision": panel, "path": "panel.parquet"}
                if panel
                else None,
            )
        elif action == "data_diff":
            if comparison is None:
                raise ValueError("Applied comparison is missing its retained report")
            result["report"] = comparison
        else:
            raise ValueError(f"Unknown recorded action {action!r}")
        # Pins whose named owner is not represented above remain archival facts.
        represented = {
            "fit": {"model", "panel"},
            "prepare_data": {"raw_data", "model"},
            "simulate": {"model", "panel"},
        }.get(action, set())
        unknown = {name: revision for name, revision in pins.items() if name not in represented}
        if unknown:
            retained["input_pins"] = unknown
        outcome = {"status": "applied", "result": result}
    elif status == "rejected":
        outcome = {"status": "rejected", "reason": "recorded_rejection", "detail": value["reason"]}
    elif status == "raised":
        if value["error_type"] is None or value["error_message"] is None:
            raise ValueError("Raised archive must retain its actual error type and message")
        outcome = {
            "status": "raised",
            "error_type": value["error_type"],
            "error_message": value["error_message"],
            "details": (),
        }
    else:
        raise ValueError(f"Unknown recorded outcome {status!r}")
    if diagnostics:
        retained["measurements"] = diagnostics
    record = {
        **{name: value[name] for name in ("seq", "branch", "ts", "trace_ids")},
        "messages": value.get("messages", ()),
        "attempt_id": value.get("attempt_id"),
        "attempt": {"action": action, "request": request, "outcome": outcome},
    }
    return record, retained


def convert_study(source: Path, destination: Path) -> dict[str, str]:
    repository = pygit2.Repository(str(source / "study/history.git"))
    archives: dict[str, JsonObject] = {}

    def update_file(oid: str, name: str, payload: JsonObject) -> JsonObject:
        if name != "attempt.json":
            return payload
        tree = repository[pygit2.Oid(hex=oid)].peel(pygit2.Tree)
        comparison = (
            json.loads(tree["data-diff.json"].peel(pygit2.Blob).data)
            if "data-diff.json" in tree
            else None
        )
        record, retained = convert_attempt(payload, source.name, comparison)
        if retained:
            archives[oid] = retained
        return record

    return rewrite_study(
        source,
        destination,
        lambda value: value,
        mapping_name="format-11-revisions.json",
        source_format=10,
        target_format=11,
        preserve_model_meaning=True,
        update_file=update_file,
        rename_entry=lambda name: None if name == "data-diff.json" else name,
        additional_files=lambda oid: (
            {"retained-metadata.json": archives[oid]} if oid in archives else {}
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = convert_study(args.source, args.destination)
    print(f"Converted format 10 → 11: {len(mapping)} mapped Git objects")


if __name__ == "__main__":
    main()
