"""Offline conversion of construct uncertainty containers to shared coefficient expressions."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from scripts.migrate_coefficient_values import convert_value


def convert_payload[T](payload: T) -> T:
    """Preserve scalar values, parameter identities, and joint construct references."""
    value = deepcopy(payload)

    def visit(item):
        if isinstance(item, dict):
            if "id" in item and "temporal_status" in item:
                operands = list(item.get("coefficients", ()))

                def append(role, coefficient, construct_ids=()):
                    if coefficient is not None:
                        operands.append(
                            {
                                "kind": "coefficient",
                                "role": role,
                                "value": convert_value(coefficient),
                                "construct_ids": list(construct_ids),
                            }
                        )

                noise = item.pop("innovation", None)
                if noise is not None:
                    item["innovation_family"] = noise.get("distribution", "gaussian")
                    append("diffusion_scale", noise["scale"])
                    append("process_degrees_of_freedom", noise.get("degrees_of_freedom"))
                    for loading in noise.get("loadings", ()):
                        append("diffusion_loading", loading["coefficient"], (loading["other_id"],))
                initial = item.pop("initial_state", None)
                if initial is not None:
                    append("initial_mean", initial["mean"])
                    append("initial_scale", initial["scale"])
                    for correlation in initial.get("correlations", ()):
                        append(
                            "initial_correlation",
                            correlation["coefficient"],
                            (correlation["other_id"],),
                        )
                if noise is not None or initial is not None:
                    item["coefficients"] = operands
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--write", action="store_true", help="Replace the supplied JSON files")
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
