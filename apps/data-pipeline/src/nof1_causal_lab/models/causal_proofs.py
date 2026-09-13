"""Nominal evidence required before emitting numeric causal results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.causal_design import CausalDesign
    from nof1_causal_lab.artifacts.identity import CausalDesignRef
    from nof1_causal_lab.models.ssm.inference.types import FittedArtifact


@dataclass(frozen=True, slots=True)
class IdentifiedEstimand:
    """Positive identification evidence for one treatment/outcome estimand."""

    causal_design: CausalDesignRef
    treatment: str
    outcome: str
    method: str
    estimand: str


@dataclass(frozen=True, slots=True)
class ReportablePosterior:
    """A persisted posterior certified as production particle-MCMC output."""

    artifact: FittedArtifact

    def __post_init__(self) -> None:
        _validate_reportable_artifact(self.artifact)


@dataclass(frozen=True, slots=True)
class CertifiedCausalAnalysis:
    """Identification and particle-posterior evidence joined by provenance."""

    causal_design: CausalDesign
    causal_design_ref: CausalDesignRef
    estimands: tuple[IdentifiedEstimand, ...]
    posterior: ReportablePosterior

    def __post_init__(self) -> None:
        if not self.estimands:
            raise ValueError("at least one identified estimand is required")
        posterior_design = self.posterior.artifact.provenance.causal_design
        if posterior_design != self.causal_design_ref:
            raise ValueError(
                "causal design payload and posterior provenance reference different designs"
            )
        outcomes = {estimand.outcome for estimand in self.estimands}
        if len(outcomes) != 1:
            raise ValueError("all identified estimands must target the same outcome")
        treatments = [estimand.treatment for estimand in self.estimands]
        if len(treatments) != len(set(treatments)):
            raise ValueError("identified estimands must not contain duplicate treatments")
        by_name = {
            construct.name: construct.id for construct in self.causal_design.latent.constructs
        }
        status = self.causal_design.identifiability
        for estimand in self.estimands:
            if estimand.causal_design != self.causal_design_ref:
                raise ValueError(
                    "identification evidence and posterior provenance reference different "
                    "causal designs"
                )
            details = (
                status.identifiable_treatments.get(by_name[estimand.treatment])
                if status is not None
                else None
            )
            if details is None:
                raise ValueError(f"effect of {estimand.treatment!r} is not identified")
            if details.method != estimand.method or details.estimand != estimand.estimand:
                raise ValueError("identification evidence does not match the causal design")

    @property
    def treatments(self) -> list[str]:
        return [estimand.treatment for estimand in self.estimands]

    @property
    def outcome(self) -> str:
        return self.estimands[0].outcome


def certify_identified_estimand(
    causal_design: CausalDesign,
    *,
    causal_design_ref: CausalDesignRef,
    treatment: str,
    outcome: str,
) -> IdentifiedEstimand:
    """Validate and materialize identification evidence for one estimand."""
    default_outcome = causal_design.latent.default_outcome
    declared_outcome = next(
        (
            construct.name
            for construct in causal_design.latent.constructs
            if default_outcome is not None and construct.id == default_outcome.id
        ),
        None,
    )
    if outcome != declared_outcome:
        raise ValueError(
            f"{outcome!r} does not match the outcome covered by the causal design identification"
        )
    construct_ids = {construct.name: construct.id for construct in causal_design.latent.constructs}
    if treatment not in construct_ids:
        raise ValueError(f"{treatment!r} is not a construct in the causal design")
    status = causal_design.identifiability
    details = (
        status.identifiable_treatments.get(construct_ids[treatment]) if status is not None else None
    )
    if details is None:
        raise ValueError(f"effect of {treatment!r} on {outcome!r} is not identified")
    return IdentifiedEstimand(
        causal_design=causal_design_ref,
        treatment=treatment,
        outcome=outcome,
        method=details.method,
        estimand=details.estimand,
    )


def _validate_reportable_artifact(artifact: FittedArtifact) -> None:
    from nof1_causal_lab.models.ssm.inference.types import ParticleMCMCPosterior

    if not isinstance(artifact.result, ParticleMCMCPosterior):
        raise TypeError("reporting requires a ParticleMCMCPosterior")
    evidence = artifact.result.evidence
    if evidence.engine != "marginal_particle_gibbs":
        raise ValueError("posterior was not produced by marginalized Particle Gibbs")
    if evidence.latent_transition != "euler_maruyama":
        raise ValueError("posterior did not target the nonlinear Euler-Maruyama transition")
    samples = artifact.result.get_samples()
    if not samples:
        raise ValueError("posterior contains no retained samples")
    draw_counts = {int(values.shape[0]) for values in samples.values()}
    if len(draw_counts) != 1 or next(iter(draw_counts)) < 1:
        raise ValueError("posterior sample sites must share a positive draw dimension")


def certify_reportable_posterior(artifact: FittedArtifact) -> ReportablePosterior:
    """Validate particle-engine evidence and non-empty retained posterior draws."""
    return ReportablePosterior(artifact=artifact)
