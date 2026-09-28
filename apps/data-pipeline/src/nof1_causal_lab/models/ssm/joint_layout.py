"""Scientific column order shared by joint-law identity, storage and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.identity import scientific_id

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import jax

    from nof1_causal_lab.artifacts.identity import (
        ConstructId,
        DistributionId,
        ParameterElementId,
        ParameterId,
    )
    from nof1_causal_lab.models.ssm.compile.bindings import CompiledParameterBinding


@dataclass(frozen=True)
class JointLawLayout:
    """An explicit law membership ordered independently of native tensor positions.

    Parameters sort by identity, then by scientific element identity. Trajectories
    follow in construct-identity order, retaining the supplied time-point order.
    Element identities include their categorical or covariance basis.
    """

    parameters: tuple[tuple[ParameterId, tuple[ParameterElementId, ...]], ...]
    constructs: tuple[ConstructId, ...]
    time_points: tuple[float, ...]

    @classmethod
    def from_bindings(
        cls,
        bindings: Collection[CompiledParameterBinding],
        *,
        parameters: Collection[ParameterId],
        constructs: Collection[ConstructId],
        time_points: Sequence[float],
    ) -> JointLawLayout:
        by_id = {binding.parameter_id: binding for binding in bindings}
        return cls(
            parameters=tuple(
                (identity, tuple(sorted(by_id[identity].coordinates)))
                for identity in sorted(parameters)
            ),
            constructs=tuple(sorted(constructs)),
            time_points=tuple(float(value) for value in time_points),
        )

    @property
    def parameter_columns(self) -> dict[ParameterElementId, int]:
        return {
            identity: index
            for index, identity in enumerate(
                element for _, elements in self.parameters for element in elements
            )
        }

    @property
    def trajectory_slices(self) -> dict[ConstructId, slice]:
        offset = len(self.parameter_columns)
        steps = len(self.time_points)
        return {
            identity: slice(offset + index * steps, offset + (index + 1) * steps)
            for index, identity in enumerate(self.constructs)
        }

    @property
    def width(self) -> int:
        return len(self.parameter_columns) + len(self.constructs) * len(self.time_points)

    @property
    def distribution_id(self) -> DistributionId:
        return scientific_id(
            "distribution",
            [
                [[identity, list(elements)] for identity, elements in self.parameters],
                [[identity, list(self.time_points)] for identity in self.constructs],
            ],
        )

    def pack(
        self,
        parameters: Mapping[ParameterElementId, np.ndarray | jax.Array],
        trajectories: Mapping[ConstructId, np.ndarray | jax.Array],
    ) -> np.ndarray:
        """Pack aligned scientific draws into a host array for persistence."""
        columns = [np.asarray(parameters[identity])[:, None] for identity in self.parameter_columns]
        columns.extend(np.asarray(trajectories[identity]) for identity in self.trajectory_slices)
        return np.concatenate(columns, axis=1)

    def unpack(
        self, draws: jax.Array
    ) -> tuple[dict[ParameterElementId, jax.Array], dict[ConstructId, jax.Array]]:
        """Read aligned scientific quantities from a draw-by-event runtime array."""
        if draws.ndim != 2 or draws.shape[1] != self.width:
            raise ValueError(
                "Joint event coordinates do not match the model's scientific quantities"
            )
        return (
            {identity: draws[:, column] for identity, column in self.parameter_columns.items()},
            {identity: draws[:, span] for identity, span in self.trajectory_slices.items()},
        )
