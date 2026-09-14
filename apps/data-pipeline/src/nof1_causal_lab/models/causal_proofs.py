"""Nominal evidence required before emitting numeric causal results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.identity import ModelRevision
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.machine.store import TransitionRecord


@dataclass(frozen=True, slots=True)
class IdentifiedEstimand:
    """Positive identification evidence for one treatment/outcome estimand."""

    model: ModelRevision
    treatment: str
    outcome: str
    method: str
    estimand: str


@dataclass(frozen=True, slots=True)
class CertifiedCausalAnalysis:
    """Identification and particle-posterior evidence joined by provenance."""

    model: ModelSpec
    model_revision: ModelRevision
    identification: IdentificationReport
    estimands: tuple[IdentifiedEstimand, ...]
    inference: TransitionRecord

    def __post_init__(self) -> None:
        if not self.estimands:
            raise ValueError("at least one identified estimand is required")
        certify_conditioned_model(self.model, self.model_revision, self.inference)
        outcomes = {estimand.outcome for estimand in self.estimands}
        if len(outcomes) != 1:
            raise ValueError("all identified estimands must target the same outcome")
        treatments = [estimand.treatment for estimand in self.estimands]
        if len(treatments) != len(set(treatments)):
            raise ValueError("identified estimands must not contain duplicate treatments")
        for estimand in self.estimands:
            if estimand.model != self.model_revision:
                raise ValueError(
                    "identification evidence and posterior provenance reference different "
                    "model revisions"
                )
            expected = certify_identified_estimand(
                self.model,
                self.identification,
                model_revision=self.model_revision,
                treatment=estimand.treatment,
                outcome=estimand.outcome,
            )
            if expected != estimand:
                raise ValueError("identification evidence does not match the model findings")

    @property
    def treatments(self) -> list[str]:
        return [estimand.treatment for estimand in self.estimands]

    @property
    def outcome(self) -> str:
        return self.estimands[0].outcome


def certify_identified_estimand(
    model: ModelSpec,
    identification: IdentificationReport,
    *,
    model_revision: ModelRevision,
    treatment: str,
    outcome: str,
) -> IdentifiedEstimand:
    """Validate and materialize identification evidence for one estimand."""
    identification.validate_model(model)
    default_outcome = model.default_outcome
    declared_outcome = next(
        (
            construct.name
            for construct in model.constructs
            if default_outcome is not None and construct.id == default_outcome.id
        ),
        None,
    )
    if (
        default_outcome is None
        or outcome != declared_outcome
        or identification.outcome != default_outcome.id
    ):
        raise ValueError(
            f"{outcome!r} does not match the outcome covered by the model identification"
        )
    construct_ids = {construct.name: construct.id for construct in model.constructs}
    if treatment not in construct_ids:
        raise ValueError(f"{treatment!r} is not a construct in the model")
    details = identification.status.identifiable_treatments.get(construct_ids[treatment])
    if details is None:
        raise ValueError(f"effect of {treatment!r} on {outcome!r} is not identified")
    return IdentifiedEstimand(
        model=model_revision,
        treatment=treatment,
        outcome=outcome,
        method=details.method,
        estimand=details.estimand,
    )


def certify_conditioned_model(
    model: ModelSpec, revision: ModelRevision, record: TransitionRecord
) -> None:
    """Join the current scientific value to committed exact-engine evidence in its log."""
    from nof1_causal_lab.machine.inference import inference_record
    from nof1_causal_lab.models.model_inputs import input_fingerprints

    if inference_record([record], revision.version) is None:
        raise ValueError(
            "Causal reporting requires the committed inference transition for this model revision"
        )
    produced = next(info for info in record.produced if info.artifact_id == "model")
    if produced.model_inputs["belief"] != input_fingerprints(model)["belief"]:
        raise ValueError("The model value differs from the revision certified by the inference log")
    evidence = record.diagnostics["engine_evidence"]
    if evidence["engine"] != "marginal_particle_gibbs":
        raise ValueError("Inference did not use the production particle-MCMC target")
    if evidence["latent_transition"] != "euler_maruyama":
        raise ValueError("Inference did not target the nonlinear Euler-Maruyama transition")
    if not model.distributions or not model.time_points:
        raise ValueError("The model has no retained joint uncertainty")
