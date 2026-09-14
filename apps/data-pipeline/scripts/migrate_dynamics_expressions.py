"""Offline conversion of retired mechanism records to compositional expressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nof1_causal_lab.artifacts.expressions import (
    coefficient,
    hill,
    linear_effect,
    restoring_potential,
    state,
)
from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from scripts.migrate_coefficient_values import convert_value

type RetiredMechanismPayload = dict[str, Any]


def convert_mechanism(
    payload: RetiredMechanismPayload, *, target: str, source: str | None
) -> RetiredMechanismPayload:
    """Preserve mechanism and parameter identities and every fixed model-scale value."""
    values = {
        key: convert_value(value)
        for key, value in payload.items()
        if isinstance(value, dict) and value.get("kind") in {"fixed", "parameter"}
    }
    match payload["kind"]:
        case "node_potential":
            expression = restoring_potential(ConstructId(target), **values)
        case "constant_drift":
            expression = coefficient(values["intercept"], "intercept")
        case "linear":
            if source is None:
                raise ValueError("A retired linear edge requires its containing edge")
            expression = linear_effect(ConstructId(source), values["weight"])
        case "interaction":
            if source is None:
                raise ValueError("A retired interaction requires its containing edge")
            expression = linear_effect(ConstructId(source), values["weight"]) * state(
                payload["moderator_id"]
            )
        case "hill":
            if source is None:
                raise ValueError("A retired Hill edge requires its containing edge")
            expression = hill(state(ConstructId(source)), **values)
        case _:
            raise ValueError(f"Unknown retired mechanism kind {payload['kind']!r}")
    return DynamicsMechanismSpec(
        id=payload["id"],
        kind="potential" if payload["kind"] == "node_potential" else "drift",
        expression=expression,
    ).model_dump(mode="json")


def convert_owned_expressions(payload: Any) -> Any:
    """Convert owned terms within models, snapshots, and retained authoring traces."""
    if isinstance(payload, list):
        return [convert_owned_expressions(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    value = {key: convert_owned_expressions(item) for key, item in payload.items()}
    for field in ("dynamics", "mechanisms"):
        if "id" not in value or field not in value:
            continue
        value[field] = [
            convert_mechanism(
                term,
                target=value["effect_id"] if field == "mechanisms" else value["id"],
                source=value["cause_id"] if field == "mechanisms" else None,
            )
            if "kind" in term
            else term
            for term in value[field]
        ]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for path in args.paths:
        original = json.loads(path.read_text())
        converted = convert_owned_expressions(original)
        if converted != original:
            if args.write:
                path.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n")
            print(f"{'Converted' if args.write else 'Would convert'} {path}")


if __name__ == "__main__":
    main()
