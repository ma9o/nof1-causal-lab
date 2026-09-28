"""Generate the illustrative workbench comparisons with the production comparison readers.

This pins one retained DEMO coefficient for the story; it performs no inference.
Run with uv from apps/data-pipeline. --check verifies the checked-in responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nof1_causal_lab.actions.revisions import (
    compare_model_definitions,
    compare_model_graph,
    compare_parameters,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_inputs import input_fingerprints

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "apps/web/src/components/__fixtures__/workbench-comparisons.json"


def build_outputs():
    snapshot = json.loads((ROOT / "data/DEMO/fixture/model_snapshot.json").read_text())
    free = ModelSpec.model_validate(snapshot["model"]["value"])
    parameter = next(
        item
        for item in free.parameters
        if item.name == "beta_perceived_stress_burden_internalizing_symptom_burden"
    )
    fixed = parameter.model_copy(
        update={
            "value": 0,
            "distribution": None,
            "distribution_transform": "identity",
            "reference_interval_days": None,
        }
    )
    pinned = free.revised(
        parameters=tuple(fixed if item.id == parameter.id else item for item in free.parameters),
        distributions={
            key: law for key, law in free.distributions.items() if key != parameter.distribution
        },
    )
    history = json.loads((ROOT / "data/DEMO/fixture/model_history.json").read_text())
    models = {
        history[str(seq)]["context"]["state"]["current"]["model"][
            "revision"
        ]: ModelSpec.model_validate(history[str(seq)]["model"]["value"])
        for seq in (2, 3, 4, 7)
    }
    models.update(
        {format(n, "x").rjust(40, "a"): model for n, model in [(5, free), (6, free), (7, pinned)]}
    )
    comparisons = {}
    for before_version, left in models.items():
        for after_version, right in models.items():
            parameters = compare_parameters(left, right)
            before = input_fingerprints(left)
            comparisons[f"{before_version}:{after_version}"] = {
                "definition_changes": [
                    item.model_dump(mode="json") for item in compare_model_definitions(left, right)
                ],
                "parameters": [item.model_dump(mode="json") for item in parameters],
                "graph": compare_model_graph(left, right, parameters).model_dump(mode="json"),
                "changed_inputs": [
                    key for key, value in input_fingerprints(right).items() if before[key] != value
                ],
            }
    return {
        "pinned_model": pinned.model_dump(mode="json"),
        "pinned_inputs": input_fingerprints(pinned),
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = json.dumps(build_outputs(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if OUTPUT.read_text() != output:
            raise SystemExit("Workbench comparison fixtures are stale; regenerate them.")
    else:
        OUTPUT.write_text(output)
    print("Workbench comparison fixtures are current.")
