"""Offline conversion of former family/link likelihood records to conditional expressions."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    LiteralExpression,
    coefficient,
    map_expression,
    state,
)
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLaw
from nof1_causal_lab.models.likelihoods import observation_law

type RetiredPayload = dict[str, Any]

_RETIRED_SLOTS = {
    "loading": "loading",
    "observation_intercept": "intercept",
    "observation_scale": "scale",
    "degrees_of_freedom": "degrees_of_freedom",
    "shape": "shape",
    "dispersion": "dispersion",
    "concentration": "concentration",
    "cutpoint_base": "cutpoint_base",
    "cutpoint_gaps": "cutpoint_gaps",
    "category_intercepts": "category_intercepts",
    "category_slopes": "category_slopes",
}


def _coefficient(value):
    if value is None:
        return None
    return (FixedCoefficient if value["kind"] == "fixed" else ParameterCoefficient).model_validate(
        value
    )


def convert_likelihood(
    value: RetiredPayload, construct_id: str, *, ordinal_levels=()
) -> RetiredPayload:
    """Convert one explicit retired definition; running application schemas reject that form."""
    old = deepcopy(value)
    family, link = old.pop("distribution"), old.pop("link")
    shell = LikelihoodSpec(
        law=observation_law(construct_id, family, link), reasoning=old["reasoning"]
    )
    original_predictor = shell.terms.predictor
    replacements = {
        role: _coefficient(old.pop(name, None)) for role, name in _RETIRED_SLOTS.items()
    }
    cross_loadings = old.pop("cross_loadings", [])

    def operand(node):
        if isinstance(node, CoefficientExpression):
            if (
                node.role == "cutpoint_gaps"
                and ordinal_levels
                and len(ordinal_levels) <= 2
                and replacements[node.role] is None
            ):
                return LiteralExpression(value=0)
            return node.model_copy(update={"coefficient": replacements[node.role]})
        return node

    predictor = map_expression(original_predictor, operand)
    extended = predictor
    for loading in cross_loadings:
        extended = extended + coefficient(_coefficient(loading["coefficient"]), "loading") * state(
            loading["other_id"]
        )

    def add_cross_loadings(node):
        return extended if node == predictor else node

    law = ObservationLaw(
        distribution=shell.law.distribution,
        arguments={
            name: map_expression(map_expression(argument, operand), add_cross_loadings)
            for name, argument in shell.law.arguments.items()
        },
    )
    return LikelihoodSpec(law=law, **old).model_dump(mode="json")


def convert_payload[T](payload: T) -> T:
    """Traverse retained model definitions within fixtures, journal records, or artifacts."""
    value = deepcopy(payload)

    def visit(item, owner=None):
        if isinstance(item, dict):
            identity = item.get("id")
            if isinstance(identity, str) and identity.startswith("construct:"):
                owner = identity
            for key, child in tuple(item.items()):
                if key == "likelihood" and isinstance(child, dict) and "distribution" in child:
                    if owner is None:
                        raise ValueError("A retired likelihood requires its owning construct")
                    item[key] = convert_likelihood(
                        child, owner, ordinal_levels=item.get("ordinal_levels", ())
                    )
                else:
                    visit(child, owner)
        elif isinstance(item, list):
            for child in item:
                visit(child, owner)

    visit(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument(
        "--write", action="store_true", help="Replace supplied JSON files after conversion"
    )
    args = parser.parse_args()
    changed = 0
    for path in args.paths:
        original = json.loads(path.read_text())
        converted = convert_payload(original)
        if converted == original:
            continue
        changed += 1
        if args.write:
            path.write_text(
                json.dumps(converted, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
    print(f"{'Converted' if args.write else 'Would convert'} {changed} file(s)")


if __name__ == "__main__":
    main()
