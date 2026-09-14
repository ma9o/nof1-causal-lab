"""Explicit offline migration from the flat quantity catalogue to component-owned slots."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.parameter_planning import complete_component_slots

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject

type RetiredModelPayload = dict[str, Any]


_LIKELIHOOD_FIELDS = {
    "loading": "loading",
    "manifest_means": "intercept",
    "manifest_var_diag": "scale",
    "obs_df": "degrees_of_freedom",
    "obs_shape": "shape",
    "obs_r": "dispersion",
    "obs_concentration": "concentration",
    "obs_ordered_base": "cutpoint_base",
    "obs_ordered_gaps": "cutpoint_gaps",
    "obs_cat_intercepts": "category_intercepts",
    "obs_cat_slopes": "category_slopes",
}


def convert_component_slots(payload: RetiredModelPayload) -> JsonObject:
    """Preserve parameter IDs, laws, constants and the two distinct dependency coordinates."""
    value = deepcopy(payload)
    policies = value.pop(
        "policies", {"initialization": "stationary", "observation_intercept": "free"}
    )
    constructs = {item["id"]: item for item in value["constructs"]}
    indicators = {
        indicator["id"]: indicator
        for item in constructs.values()
        for indicator in item.get("indicators", ())
    }
    indicator_owners = {
        indicator["id"]: item["id"]
        for item in constructs.values()
        for indicator in item.get("indicators", ())
    }
    families = {key: item.pop("innovation_family", "gaussian") for key, item in constructs.items()}
    order = {identity: index for index, identity in enumerate(constructs)}

    def fixed(number):
        return {"kind": "fixed", "value": number}

    for parameter in value.get("parameters", ()):
        kind = parameter.pop("quantity")
        owners = parameter.pop("owners")
        states = sorted(
            (owner["id"] for owner in owners if owner["kind"] == "construct"), key=order.__getitem__
        )
        manifests = [owner["id"] for owner in owners if owner["kind"] == "indicator"]
        reference = {"kind": "parameter", "parameter_id": parameter["id"]}
        if kind in _LIKELIHOOD_FIELDS:
            for identity in manifests:
                if kind == "loading" and states[0] != indicator_owners[identity]:
                    indicators[identity]["likelihood"].setdefault("cross_loadings", []).append(
                        {"other_id": states[0], "coefficient": reference}
                    )
                else:
                    indicators[identity]["likelihood"][_LIKELIHOOD_FIELDS[kind]] = reference
        elif kind in {"diffusion_diag", "diffusion_lower", "proc_df"}:
            targets = states if kind == "proc_df" else [states[-1]]
            for identity in targets:
                target = constructs[identity]
                noise = target.setdefault(
                    "innovation", {"distribution": families[identity], "scale": fixed(0)}
                )
                if kind == "diffusion_diag":
                    noise["scale"] = reference
                elif kind == "proc_df":
                    noise["degrees_of_freedom"] = reference
                else:
                    noise.setdefault("loadings", []).append(
                        {"other_id": states[0], "coefficient": reference}
                    )
        elif kind in {"t0_means", "t0_var_diag", "t0_var_lower", "static_state_sd"}:
            for identity in states if kind == "static_state_sd" else [states[-1]]:
                initial = constructs[identity].setdefault(
                    "initial_state", {"mean": fixed(0), "scale": fixed(1)}
                )
                if kind == "t0_means":
                    initial["mean"] = reference
                elif kind == "t0_var_lower":
                    initial.setdefault("correlations", []).append(
                        {"other_id": states[0], "coefficient": reference}
                    )
                else:
                    initial["scale"] = reference

    def coefficient_kinds(node):
        if isinstance(node, dict):
            if node.get("kind") == "estimated":
                node["kind"] = "parameter"
            for child in node.values():
                coefficient_kinds(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                coefficient_kinds(child)

    coefficient_kinds(value)
    from scripts.migrate_connected_graph import connect_endpoints
    from scripts.migrate_dynamics_expressions import convert_owned_expressions
    from scripts.migrate_likelihood_expressions import convert_payload

    model = ModelSpec.model_validate(
        connect_endpoints(convert_payload(convert_owned_expressions(value)))
    )
    if model.parameters or any(construct.dynamics for construct in model.constructs):
        model = complete_component_slots(
            model,
            free_initial=policies["initialization"] == "free",
            free_observation_intercepts=policies["observation_intercept"] == "free",
        )
    return model.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace the specified files with their converted values",
    )
    args = parser.parse_args()
    for path in args.paths:
        converted = convert_component_slots(json.loads(path.read_text()))
        if args.write:
            path.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n")
        print(f"{'Converted' if args.write else 'Validated'} {path}")


if __name__ == "__main__":
    main()
