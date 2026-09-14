"""Runtime interventional and counterfactual simulations over nonlinear dynamics.

Abducted starts use exact particle-smoother latent paths from the conditioned
model. Timed clamps and forward simulation use the shared dynamics engine.
"""

from __future__ import annotations

from .estimands import (
    summarize_draws,
)
from .orchestration import (
    ClampSpec,
    build_segment_bounds,
    vmap_simulate_clamps_from_state,
)

__all__ = [
    "ClampSpec",
    "build_segment_bounds",
    "summarize_draws",
    "vmap_simulate_clamps_from_state",
]
