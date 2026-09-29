"""Unify archived observation declarations with flat fill_null in an offline study copy.

Usage: uv run python -m scripts.migrations.migrate_fill_null SOURCE DESTINATION
Numerical tables and results are preserved, not recomputed. Future preparation
uses Polars semantics: zero filling also replaces explicitly null values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nof1_causal_lab.artifacts.data_preparation import DataPreparationSpec
from scripts.migrations.study_rewrite import rewrite_study


def update_observation_definitions(value):
    """Translate retired recording/nested filling and project one observation schema."""
    if isinstance(value, list):
        return [update_observation_definitions(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: update_observation_definitions(item) for key, item in value.items()}
    if {"recording", "measurement_dtype", "aggregation"} <= result.keys():
        if "fill_null" in result:
            raise ValueError("Archived variable declares both recording and fill_null")
        result["fill_null"] = {
            "samples": None,
            "changes": "forward",
            "events": 0,
        }[result.pop("recording")]
    if {"measurement_dtype", "aggregation"} <= result.keys() and isinstance(
        result.get("fill_null"), dict
    ):
        fill = result.pop("fill_null")
        result["fill_null"] = (
            fill["strategy"] if fill.get("strategy") is not None else fill["value"]
        )
        if fill.get("limit") is not None:
            result["fill_null_limit"] = fill["limit"]
    if {"source", "variables", "preparation"} <= result.keys() and result[
        "preparation"
    ] is not None:
        recipe = DataPreparationSpec.model_validate(result["preparation"])
        result["variables"] = [item.model_dump(mode="json") for item in recipe.observation_schema()]
    return result


def migrate_workspace(source: Path, destination: Path) -> dict[str, str]:
    """Copy a stopped study and rewrite its Git graph and stored revision references."""
    return rewrite_study(
        source, destination, update_observation_definitions, mapping_name="fill-null-mapping.json"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate_workspace(args.source, args.destination)
    print(f"Migrated {len(mapping)} Git objects into {args.destination}; source unchanged")


if __name__ == "__main__":
    main()
