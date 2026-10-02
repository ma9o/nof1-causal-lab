"""Leaf factory for concrete marginal likelihood backend construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec, NormalLawSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.expression import BoundExpression

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.likelihood import Law
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.execution.contracts import ObservationLaws
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime


def build_laplace_backend(
    spec: CompiledModel,
    n_ieks_iters: int,
    observation_support: ObservationSupportRuntime | None = None,
):
    """Construct a Laplace likelihood backend for a compiled spec."""
    from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood

    return LaplaceLikelihood(
        n_latent=numeric.n_states(spec),
        n_manifest=numeric.n_observations(spec),
        n_ieks_iters=n_ieks_iters,
        observation_support=observation_support,
    )


def initialization_observation_laws(laws: ObservationLaws) -> ObservationLaws:
    """Gaussian initialization view only; particle targets retain the original Delta."""

    def gaussian(law: Law[BoundExpression]) -> Law[BoundExpression]:
        if not isinstance(law, DeltaLawSpec):
            return law
        scale = BoundExpression(
            law.v.expression,
            lambda _eta, scale, _values: scale,
            (),
            (),
            0,
            0,
            lambda eta: eta,
            law.v.link,
        )
        return NormalLawSpec(loc=law.v, scale=scale)

    return tuple(gaussian(law) for law in laws)
