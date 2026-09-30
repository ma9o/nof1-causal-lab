"""Drop quantile bands and keep a few sample series in an offline study copy's predictive overlays.

Usage: uv run python -m scripts.migrations.migrate_predictive_overlays SOURCE DESTINATION
Observed values, medians and every other check result are preserved, not recomputed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nof1_causal_lab.models.posterior_predictive import PREDICTIVE_SAMPLE_SERIES
from scripts.migrations.study_rewrite import rewrite_study

_BANDS = ("q025", "q25", "q75", "q975")


def update_predictive_overlays(value):
    """Remove retired band arrays and keep evenly spaced stored draws, as new checks do."""
    if isinstance(value, list):
        return [update_predictive_overlays(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: update_predictive_overlays(item) for key, item in value.items()}
    if not {"indicator_id", "observed", "median", "spaghetti_draws"} <= result.keys():
        return result
    for band in _BANDS:
        result.pop(band, None)
    draws = result["spaghetti_draws"]
    if len(draws) > PREDICTIVE_SAMPLE_SERIES:
        keep = np.linspace(0, len(draws) - 1, PREDICTIVE_SAMPLE_SERIES).astype(int)
        result["spaghetti_draws"] = [draws[index] for index in keep]
    return result


def migrate_workspace(source: Path, destination: Path) -> dict[str, str]:
    """Copy a stopped study and slim its stored predictive overlays."""
    return rewrite_study(
        source,
        destination,
        update_predictive_overlays,
        mapping_name="predictive-overlay-mapping.json",
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
