"""Offline conversion from construct catalogues to connected graphs with shared endpoints."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.json_types import JsonObject


def connect_endpoints(payload: JsonObject) -> JsonObject:
    """Move existing definitions to their edges; never invent or discard a construct."""
    value = deepcopy(payload)
    objects = TypeAdapter(list[JsonObject])
    identity_type = TypeAdapter(ConstructId)
    constructs = objects.validate_python(value.pop("constructs"))
    definitions = {
        identity_type.validate_python(construct["id"]): construct for construct in constructs
    }
    if len(definitions) != len(constructs):
        raise ValueError("Construct definitions must have unique identities")
    referenced = set()
    edges = objects.validate_python(value["edges"])
    for edge in edges:
        for endpoint in ("cause", "effect"):
            identity = identity_type.validate_python(edge.pop(f"{endpoint}_id"))
            if identity not in definitions:
                raise ValueError(f"Undefined construct endpoint {identity!r}")
            edge[endpoint] = (
                {"kind": "construct", "id": identity}
                if identity in referenced
                else definitions[identity]
            )
            referenced.add(identity)
    if isolated := definitions.keys() - referenced:
        raise ValueError(
            f"Isolated constructs need an explicit scientific revision: {sorted(isolated)}"
        )
    value["edges"] = list(edges)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("The destination must not exist; preserve the original model")
    model = ModelSpec.model_validate(connect_endpoints(json.loads(args.source.read_text())))
    args.destination.write_text(model.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
