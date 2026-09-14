"""Move retained inline laws into ModelSpec.distributions without changing their values."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nof1_causal_lab.artifacts.identity import ParameterId
from nof1_causal_lab.models.model_distributions import parameter_distribution_id

type InlineLawModelPayload = dict[str, Any]


def convert_distribution_references(payload: InlineLawModelPayload) -> InlineLawModelPayload:
    """Convert one retained model payload; the source object remains unchanged."""
    converted = deepcopy(payload)
    laws = converted.setdefault("distributions", {})
    for parameter in converted.get("parameters", []):
        law = parameter.get("distribution")
        if not isinstance(law, dict):
            continue
        identity = parameter_distribution_id(ParameterId(parameter["id"]))
        if identity in laws:
            raise ValueError(f"Inline law conflicts with existing distribution {identity}")
        laws[identity] = law
        parameter["distribution"] = identity

    # Trajectory laws have explicit joint event coordinates, even with one construct.
    constructs = {}
    for edge in converted["edges"]:
        for side in ("cause", "effect"):
            construct = edge[side]
            if "distribution" in construct and isinstance(construct["distribution"], dict):
                constructs[construct["id"]] = construct
    if constructs:
        from nof1_causal_lab.artifacts.identity import ConstructId
        from nof1_causal_lab.artifacts.model_spec import ModelSpec
        from nof1_causal_lab.models.model_distributions import joint_distribution_id

        trajectory_laws = {
            identity: construct.pop("distribution") for identity, construct in constructs.items()
        }
        model = ModelSpec.model_validate(converted)
        for identity, law in trajectory_laws.items():
            reference = joint_distribution_id(
                model, (), (ConstructId(identity),), model.time_points
            )
            if reference in laws:
                raise ValueError(f"Inline law conflicts with existing distribution {reference}")
            laws[reference] = law
            constructs[identity]["distribution"] = reference
    return converted


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    model = ModelSpec.model_validate(
        convert_distribution_references(json.loads(args.source.read_text()))
    )
    with args.destination.open("x") as output:
        output.write(model.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
