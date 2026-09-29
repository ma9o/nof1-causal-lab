"""Store retained posterior draws as one batched point mass in an offline study copy.

Usage: uv run python -m scripts.migrations.migrate_empirical_laws SOURCE DESTINATION
Draw arrays are reused unchanged; each law's equal weights move into the array store.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.utils.arrays import write_array
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from collections.abc import Callable


def update_empirical_laws(value, write: Callable[[np.ndarray], str]):
    """Replace mixtures that list one point mass per stored draw with a batched Delta."""
    if isinstance(value, list):
        return [update_empirical_laws(item, write) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: update_empirical_laws(item, write) for key, item in value.items()}
    if result.get("distribution") != "MixtureGeneral":
        return result
    params = result["params"]
    components = params["component_distributions"]
    if not all(
        component["distribution"] == "Delta"
        and set(component["params"]) == {"v", "event_dim"}
        and component["params"]["event_dim"] == 1
        and isinstance(component["params"]["v"], dict)
        and "array_ref" in component["params"]["v"]
        for component in components
    ):
        return result
    draws = {**components[0]["params"]["v"], "index": []}
    if draws["shape"][0] != len(components) or any(
        component["params"]["v"] != {**draws, "index": [index]}
        for index, component in enumerate(components)
    ):
        raise ValueError("Archived point mixture does not enumerate the rows of one array")
    probs = params["mixing_distribution"]["params"]["probs"]
    weights = np.asarray(probs["array"], dtype=probs["dtype"])
    return {
        "distribution": "MixtureSameFamily",
        "params": {
            "mixing_distribution": {
                "distribution": "CategoricalProbs",
                "params": {
                    "probs": {
                        "array_ref": write(weights),
                        "shape": list(weights.shape),
                        "dtype": str(weights.dtype),
                        "index": [],
                    }
                },
            },
            "component_distribution": {
                "distribution": "Delta",
                "params": {"v": draws, "event_dim": 1},
            },
        },
    }


def migrate_workspace(source: Path, destination: Path) -> dict[str, str]:
    """Copy a stopped study and re-encode its retained posterior laws."""
    arrays = str(destination / "store/arrays")
    return rewrite_study(
        source,
        destination,
        lambda value: update_empirical_laws(value, lambda weights: write_array(arrays, weights)),
        mapping_name="empirical-law-mapping.json",
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
