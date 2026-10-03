"""State-Space Models (SSM) in NumPyro.

This module implements Bayesian state-space models with:
- Continuous-time dynamics via stochastic differential equations
- Dynestyx model interpretation for irregular time intervals
- Local marginalized Particle Gibbs inference
- Automatic reparameterization via AutoReparam
"""

from nof1_causal_lab.models.ssm.autoreparam import AutoReparam, Strategy
from nof1_causal_lab.models.ssm.transition_kinds import (
    LATENT_TRANSITION_EULER_MARUYAMA,
)

__all__ = [
    # Model
    "LATENT_TRANSITION_EULER_MARUYAMA",
    # Reparameterization
    "AutoReparam",
    "Strategy",
]
