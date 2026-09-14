"""Offline identity conversion for the retired anonymous-mechanism format."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.artifacts.identity import MechanismRef, scientific_id
from nof1_causal_lab.models.model_mechanisms import default_mechanism_id

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


type HistoricalModelPayload = dict[str, Any]


def remap_references(value: Any, identities: dict[str, str]) -> Any:
    """Replace only exact scientific identities, including map keys; leave numerical data intact."""
    if isinstance(value, dict):
        return {
            identities.get(key, key): remap_references(item, identities)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [remap_references(item, identities) for item in value]
    if isinstance(value, str):
        return identities.get(value, value)
    return value


def identify_mechanisms(
    payload: HistoricalModelPayload,
) -> tuple[HistoricalModelPayload, dict[str, str]]:
    """Assign named default identities to a historical value with one anonymous term per kind.

    Repeated anonymous terms need an explicit migration decision: list positions
    cannot establish which scientific term survived between historical revisions.
    """
    converted = deepcopy(payload)
    parameters = {item["id"]: item for item in converted.get("parameters", [])}
    claimed: set[str] = set()
    identities: dict[str, str] = {}
    for field, terms in (("constructs", "dynamics"), ("edges", "mechanisms")):
        for owner in converted[field]:
            for mechanism in owner.get(terms, []):
                if "id" in mechanism:
                    raise ValueError("This migration consumes anonymous mechanisms")
                identity = default_mechanism_id(owner["id"], mechanism["kind"])
                if identity in claimed:
                    raise ValueError(
                        "Repeated anonymous terms require explicit historical identity mapping"
                    )
                claimed.add(identity)
                mechanism["id"] = identity
                for coefficient in mechanism.values():
                    if not isinstance(coefficient, dict) or coefficient.get("kind") != "estimated":
                        continue
                    old = coefficient["parameter_id"]
                    if old in identities:
                        raise ValueError(
                            "A retired parameter references more than one mechanism slot"
                        )
                    parameter = parameters[old]
                    parameter["owners"].append(MechanismRef(id=identity).model_dump(mode="json"))
                    new = scientific_id(
                        "parameter",
                        [parameter["quantity"], sorted(item["id"] for item in parameter["owners"])],
                    )
                    identities[old] = new
                    identities[scientific_id("element", [old, "scalar"])] = scientific_id(
                        "element", [new, "scalar"]
                    )
    return remap_references(converted, identities), identities


def retired_identity_map(model: ModelSpec) -> dict[str, str]:
    """Recover retired coefficient IDs from a converted model for its retained result artifacts."""
    identities = {}
    for parameter in model.parameters:
        if not any(
            owner.kind == "mechanism" for owner in model.parameter_context(parameter.id).owners
        ):
            continue
        owners = tuple(
            owner
            for owner in model.parameter_context(parameter.id).owners
            if owner.kind != "mechanism"
        )
        old = scientific_id(
            "parameter",
            [
                model.parameter_context(parameter.id).quantity.value,
                sorted(owner.id for owner in owners),
            ],
        )
        if old == parameter.id:
            continue
        if old in identities:
            raise ValueError("Retired coefficient identity is ambiguous")
        identities[old] = parameter.id
        identities[scientific_id("element", [old, "scalar"])] = scientific_id(
            "element", [parameter.id, "scalar"]
        )
    return identities
