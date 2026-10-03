"""Fold format-15 result availability into the format-16 sum types.

Usage: uv run python -m scripts.migrations.migrate_format_16 SOURCE DESTINATION
Only a new destination is written. Stored draws, scientific findings and causal
certification are preserved; the converter performs no scientific execution.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonValue


def _availability(payload: JsonValue, reason: JsonValue) -> dict[str, JsonValue]:
    if reason is not None:
        if payload is not None:
            raise ValueError("An available payload cannot also have an unavailable reason")
        return {"kind": "unavailable", "reason": reason}
    return {"kind": "available", "value": payload}


def convert_payload(value: JsonValue) -> JsonValue:
    """Translate stored availability without changing the retained evidence."""
    if isinstance(value, (list, tuple)):
        return [convert_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    record = {key: convert_payload(item) for key, item in value.items()}
    if {"design", "latent_paths", "causal_result", "causal_unavailable_reason"} <= record.keys():
        result, reason = record.pop("causal_result"), record.pop("causal_unavailable_reason")
        design = record["design"]
        assert isinstance(design, Mapping)
        if result is not None:
            record["causal"] = _availability(result, reason)
        elif not design["interventions"]:
            if reason is not None:
                raise ValueError(
                    "A simulation without interventions cannot have a causal rejection"
                )
            record["causal"] = {
                "kind": "not_applicable",
                "reason": "No intervention was requested.",
            }
        else:
            record["causal"] = {
                "kind": "unavailable",
                "reason": reason if reason is not None else "No causal effect was recorded.",
            }
    if {
        "indicator_id",
        "left",
        "right",
        "reference_side",
        "predictive_unavailable_reason",
    } <= record.keys():
        side = record.pop("reference_side")
        checks, reason = (
            record.pop("predictive_checks"),
            record.pop("predictive_unavailable_reason"),
        )
        if checks is not None or reason is not None:
            evaluation = _availability(checks, reason)
        else:
            evaluation = {"kind": "not_applicable", "reason": "Comparison inputs are incompatible."}
        if side is not None:
            record["predictive"] = {
                "kind": "comparison",
                "reference_side": side,
                "evaluation": evaluation,
            }
        elif checks is not None:
            raise ValueError("Predictive checks require a selected reference history")
        elif reason is not None:
            record["predictive"] = evaluation
        else:
            record["predictive"] = {
                "kind": "not_applicable",
                "reason": "A predictive comparison requires replicated histories.",
            }
    if set(record) == {"columns", "unavailable_reason"}:
        columns, reason = record["columns"], record["unavailable_reason"]
        if reason is not None:
            if columns:
                raise ValueError("Unavailable parameter draws cannot contain retained columns")
            return {"kind": "unavailable", "reason": reason}
        return {"kind": "available", "value": columns}
    return record


def migrate(source: Path, destination: Path) -> dict[str, str]:
    """Rewrite only a new study copy, preserving model and numerical meaning."""
    return rewrite_study(
        source,
        destination,
        convert_payload,
        mapping_name="format-16-revisions.json",
        source_format=15,
        target_format=16,
        preserve_model_meaning=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate(args.source.resolve(), args.destination.resolve())
    print(f"Converted {len(mapping)} Git objects into {args.destination}")


if __name__ == "__main__":
    main()
