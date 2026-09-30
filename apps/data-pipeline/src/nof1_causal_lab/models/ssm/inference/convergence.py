"""Convergence checks over one fit's retained parameter chains."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.artifacts.identity import ParameterRef
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    ParameterConvergenceFailure,
    ParameterConvergenceReport,
)

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject

# Vehtari et al. (2021), https://doi.org/10.1214/20-BA1221: use the draws only when
# rank-normalized split R-hat is below 1.01 and bulk and tail ESS reach about 100
# per chain.
R_HAT_LIMIT = 1.01
ESS_PER_CHAIN = 100


class _ParameterMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    parameter: str
    r_hat: float | None
    ess_bulk: float | None
    ess_tail: float | None
    subject: ParameterRef | None = None


class _ChainMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    num_chains: int
    per_parameter: list[_ParameterMetrics]


def _failed_criteria(chains: _ChainMetrics):
    """Share criteria between raw engine warnings and the scientific projection."""
    minimum_ess = ESS_PER_CHAIN * chains.num_chains
    checks = (
        (f"R-hat < {R_HAT_LIMIT}", lambda p: p.r_hat is not None and p.r_hat < R_HAT_LIMIT),
        (
            f"bulk ESS ≥ {minimum_ess}",
            lambda p: p.ess_bulk is not None and p.ess_bulk >= minimum_ess,
        ),
        (
            f"tail ESS ≥ {minimum_ess}",
            lambda p: p.ess_tail is not None and p.ess_tail >= minimum_ess,
        ),
    )
    for requirement, passes in checks:
        if failing := [p for p in chains.per_parameter if not passes(p)]:
            yield requirement, failing


def parameter_convergence(inference_diagnostics: JsonObject) -> ParameterConvergenceReport:
    """Project failures with required scientific identities; latent paths are not checked."""
    chains = _ChainMetrics.model_validate(inference_diagnostics["mcmc"])
    failures = [
        ParameterConvergenceFailure(
            parameter=p.parameter,
            subject=ParameterRef.model_validate(p.subject),
            criterion=requirement,
        )
        for requirement, failing in _failed_criteria(chains)
        for p in failing
    ]
    return ParameterConvergenceReport(
        checked=len(chains.per_parameter),
        passed=bool(chains.per_parameter) and not failures,
        failures=failures,
    )


def convergence_failures(inference_diagnostics: JsonObject) -> list[str]:
    """Explain the same verdict for action warnings and causal certification."""
    chains = _ChainMetrics.model_validate(inference_diagnostics["mcmc"])
    return [
        f"{criterion} fails for {len(failing)} of {len(chains.per_parameter)} "
        f"parameters, including {failing[0].parameter}"
        for criterion, failing in _failed_criteria(chains)
    ]
