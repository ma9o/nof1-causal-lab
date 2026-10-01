"""Rewrite format-8 models with lawless given inputs in a new format-9 study copy.

Usage: uv run python -m scripts.migrations.migrate_format_9 SOURCE DESTINATION
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from nof1_causal_lab.numpyro_json import (
    empirical_atoms,
    empirical_distribution,
    encode_distribution,
)
from nof1_causal_lab.study.git_objects import read_file
from nof1_causal_lab.study.lineage import scientific_inference_report
from nof1_causal_lab.utils.arrays import read_array, write_array
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject


def update_model(payload: JsonObject, *, array_loader=None, array_writer=None) -> JsonObject:
    """Remove only given-input laws, preserving the retained joint coordinates."""
    definitions = {
        endpoint["id"]: endpoint
        for edge in payload["edges"]
        for endpoint in (edge["cause"], edge["effect"])
        if "name" in endpoint
    }
    inputs = set()
    for identity, construct in definitions.items():
        if construct["role"] != "exogenous":
            continue
        indicators = construct["indicators"]
        if indicators and all(
            indicator["likelihood"] is not None
            and indicator["likelihood"]["law"]
            == {
                "distribution": "Delta",
                "arguments": {"v": {"kind": "state", "construct_id": identity}},
            }
            and not indicator["likelihood"]["standardized"]
            and indicator["aggregation"] != "std"
            for indicator in indicators
        ):
            inputs.add(identity)
    original = _modeled_states(payload)
    context = {"distribution_array_loader": array_loader}
    previous = ModelSpec.model_validate(original, context=context)
    if not inputs:
        return original
    updated = deepcopy(original)
    for edge in updated["edges"]:
        for construct in (edge["cause"], edge["effect"]):
            if "name" not in construct:
                continue
            if construct["id"] in inputs:
                construct.update(
                    role="exogenous",
                    dynamics=[],
                    coefficients=[],
                    distribution=None,
                    innovation_family="gaussian",
                )
            construct["coefficients"] = [
                operand
                for operand in construct["coefficients"]
                if not inputs.intersection(operand["construct_ids"])
            ]

    def references(value):
        if isinstance(value, dict):
            if value.get("kind") == "coefficient" and isinstance(value.get("value"), str):
                yield value["value"]
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)

    used = set(references(updated["edges"]))
    updated["parameters"] = [
        parameter for parameter in updated["parameters"] if parameter["id"] in used
    ]
    bare = deepcopy(updated)
    bare["distributions"] = {}
    for parameter in bare["parameters"]:
        parameter["distribution"] = None
    for edge in bare["edges"]:
        for construct in (edge["cause"], edge["effect"]):
            if "name" in construct:
                construct["distribution"] = None
    current = ModelSpec.model_validate(bare)
    laws = {}
    replacements = {}
    for identity, law in previous.distributions.items():
        old_parameters = [p.id for p in previous.parameters if p.distribution == identity]
        old_constructs = [c.id for c in previous.constructs if c.distribution == identity]
        parameters = [p.id for p in current.parameters if p.id in old_parameters]
        constructs = [
            c.id for c in current.constructs if c.id in old_constructs and c.id not in inputs
        ]
        if not parameters and not constructs:
            continue
        if parameters == old_parameters and constructs == old_constructs:
            laws[identity] = payload["distributions"][identity]
            replacements[identity] = identity
            continue
        old_bindings, _ = parameter_bindings(previous)
        new_bindings, _ = parameter_bindings(current)
        old_layout = JointLawLayout.from_bindings(
            old_bindings,
            parameters=old_parameters,
            constructs=old_constructs,
            time_points=previous.time_points,
        )
        layout = JointLawLayout.from_bindings(
            new_bindings,
            parameters=parameters,
            constructs=constructs,
            time_points=current.time_points,
        )
        columns = [old_layout.parameter_columns[element] for element in layout.parameter_columns]
        columns.extend(
            column
            for construct in layout.constructs
            for column in range(
                old_layout.trajectory_slices[construct].start,
                old_layout.trajectory_slices[construct].stop,
            )
        )
        atoms = empirical_atoms(law)[:, columns]
        replacement = layout.distribution_id
        laws[replacement] = encode_distribution(
            empirical_distribution(atoms, array_loader=array_loader, array_writer=array_writer)
        )
        replacements[identity] = replacement
    updated["distributions"] = laws
    for parameter in updated["parameters"]:
        if parameter["distribution"] is not None:
            parameter["distribution"] = replacements[parameter["distribution"]]
    for edge in updated["edges"]:
        for construct in (edge["cause"], edge["effect"]):
            if "name" in construct and construct["distribution"] is not None:
                construct["distribution"] = replacements[construct["distribution"]]
    ModelSpec.model_validate(updated, context=context)
    return updated


def _modeled_states(payload: JsonObject) -> JsonObject:
    """Read old engine coordinates under their generated-state semantics."""
    original = deepcopy(payload)
    for edge in original["edges"]:
        for construct in (edge["cause"], edge["effect"]):
            if "name" in construct:
                construct["role"] = "endogenous"
    return original


def migrate(source: Path, destination: Path) -> dict[str, str]:
    import pygit2

    arrays = str(destination / "store/arrays")
    original = pygit2.Repository(str(source / "study/history.git"))

    def update_file(_oid, name, value):
        if name != "attempt.json" or value["action"] != "fit" or value["status"] != "applied":
            return value
        revision = value["diagnostics"]["input_pins"]["model"]
        model = ModelSpec.model_validate(
            _modeled_states(json.loads(read_file(original, revision, "model.json"))),
            context={"distribution_array_loader": lambda ref: read_array(arrays, ref)},
        )
        report = scientific_inference_report(
            model, InferenceReport.model_validate(value["diagnostics"]["report"])
        )
        return {
            **value,
            "diagnostics": {
                **value["diagnostics"],
                "report": {
                    **value["diagnostics"]["report"],
                    "inference_diagnostics": report.inference_diagnostics,
                },
            },
        }

    def update(value):
        if (
            isinstance(value, dict)
            and "edges" in value
            and "parameters" in value
            and "distributions" in value
        ):
            return update_model(
                value,
                array_loader=lambda ref: read_array(arrays, ref),
                array_writer=lambda values: write_array(arrays, values),
            )
        return value

    return rewrite_study(
        source,
        destination,
        update,
        update_file=update_file,
        mapping_name="format-9-revisions.json",
        source_format=8,
        target_format=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate(args.source, args.destination)
    print(f"Rewrote {len(mapping)} objects into {args.destination}")


if __name__ == "__main__":
    main()
