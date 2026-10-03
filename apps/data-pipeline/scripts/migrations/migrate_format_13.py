"""Compose observation and extraction definitions in stopped format-12 studies.

Usage: uv run python -m scripts.migrations.migrate_format_13 SOURCE DESTINATION
Only the new destination is written. No numerical values are regenerated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from scripts.migrations.migrate_format_12 import _check_preimages
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonValue

_OBSERVATION_FIELDS = frozenset(
    {
        "id",
        "name",
        "measurement_dtype",
        "aggregation",
        "observation_window",
        "ordinal_levels",
        "categorical_levels",
    }
)
_EXTRACTION_FIELDS = frozenset(
    {"how_to_measure", "source_columns", "computed_rule", "fill_null", "fill_null_limit"}
)


def convert_payload(value: JsonValue) -> JsonValue:
    """Move owned fields into their components throughout a retained JSON record."""
    if isinstance(value, (list, tuple)):
        return [convert_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    converted = {key: convert_payload(item) for key, item in value.items()}
    if "measurement_dtype" in converted and "id" in converted:
        if "how_to_measure" in converted or "construct_polarity" in converted:
            observation = {
                key: converted.pop(key) for key in _OBSERVATION_FIELDS if key in converted
            }
            if "how_to_measure" in converted:
                mode = converted.pop("extraction_mode", "semantic")
                extraction = {
                    key: converted.pop(key) for key in _EXTRACTION_FIELDS if key in converted
                }
                extraction["kind"] = mode
                if mode == "semantic":
                    for key in ("computed_rule", "fill_null", "fill_null_limit"):
                        if extraction.pop(key, None) is not None:
                            raise ValueError(f"Semantic extraction carries an unexplained {key}")
                converted["extraction"] = extraction
            converted["observation"] = observation
        # Preparation policies belong only to the computed extraction recipe.
        converted.pop("fill_null", None)
        converted.pop("fill_null_limit", None)
    if {"indicators", "dataset_issues", "preflight"} <= converted.keys():
        data = {key: converted.pop(key) for key in ("indicators", "dataset_issues")}
        converted["data"] = DataProfileArtifact.model_validate(data).model_dump(mode="json")
    return converted


def convert_study(source: Path, destination: Path) -> dict[str, str]:
    from scripts.migrations.migrate_format_14 import convert_model

    return rewrite_study(
        source,
        destination,
        convert_payload,
        mapping_name="format-13-revisions.json",
        source_format=12,
        target_format=13,
        preserve_model_meaning=True,
        model_definition=convert_model,
        update_file=_convert_file,
        check_preimages=_check_preimages(source),
    )


def _convert_file(_revision: str, name: str, value: JsonValue) -> JsonValue:
    """The artifact filename identifies validation reports with omitted preflight defaults."""
    if name != "validation_report.json":
        return value
    if not isinstance(value, Mapping):
        raise ValueError("A validation report must be an object")
    report = dict(value)
    profile = {key: report.pop(key) for key in ("indicators", "dataset_issues")}
    return ValidationReportArtifact.model_validate({"data": profile, **report}).model_dump(
        mode="json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = convert_study(args.source, args.destination)
    print(f"Converted format 12 → 13: {len(mapping)} mapped Git objects")


if __name__ == "__main__":
    main()
