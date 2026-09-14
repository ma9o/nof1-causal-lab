"""Deterministic scientific-expression descriptions for runtime cache keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import UncheckedJsonObject

    from .spec import DynamicsSpec


def dynamics_spec_to_dict(spec: DynamicsSpec) -> UncheckedJsonObject:
    """Describe bound expressions without persisting an executable model."""
    return {
        "n_latent": spec.n_latent,
        "components": [
            {
                "kind": component.kind,
                "expression": component.expression.model_dump(mode="json"),
                "target": component.target,
                "source": component.source,
                "state_ids": list(component.state_ids),
            }
            for component in spec.components
        ],
    }
