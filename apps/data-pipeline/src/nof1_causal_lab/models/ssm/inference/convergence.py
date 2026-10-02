"""Convergence criteria measured directly on producer-owned chain diagnostics."""

from __future__ import annotations

from nof1_causal_lab.artifacts.checks import (
    ConvergenceCriterion,
    ConvergenceSubject,
    Evaluated,
    NotEvaluated,
    NumericCriterionEvidence,
)
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    ChainDiagnostics,
    ParameterConvergenceReport,
)

# Vehtari et al. (2021), https://doi.org/10.1214/20-BA1221.
R_HAT_LIMIT = 1.01
ESS_PER_CHAIN = 100


def parameter_convergence(chains: ChainDiagnostics | None) -> ParameterConvergenceReport:
    if chains is None:
        return ParameterConvergenceReport(
            checked=0,
            messages=("No retained chain measurements are available.",),
            status="not_evaluated",
            assessments=(
                NotEvaluated(
                    subject="recorded_parameter_chains",
                    reason="NO_RETAINED_CHAINS",
                    detail="No retained chain measurements are available.",
                ),
            ),
        )
    assessments = []
    minimum = ESS_PER_CHAIN * chains.num_chains
    for parameter in chains.per_parameter:
        for criterion, value, lower, upper in (
            (ConvergenceCriterion.R_HAT, parameter.r_hat, None, R_HAT_LIMIT),
            (ConvergenceCriterion.ESS_BULK, parameter.ess_bulk, float(minimum), None),
            (ConvergenceCriterion.ESS_TAIL, parameter.ess_tail, float(minimum), None),
        ):
            subject = ConvergenceSubject(
                parameter=parameter.subject, criterion=criterion, label=parameter.parameter
            )
            if value is None:
                assessments.append(
                    NotEvaluated(
                        subject=subject,
                        reason="INSUFFICIENT_CHAIN_SAMPLES",
                        detail=f"{parameter.parameter}: {criterion.value} is unavailable.",
                    )
                )
            else:
                passes = value < upper if upper is not None else value >= minimum
                evidence = NumericCriterionEvidence(
                    criterion=criterion.value,
                    value=value,
                    lower=lower,
                    upper=upper,
                    upper_inclusive=upper is None,
                    note=parameter.parameter,
                )
                assessments.append(
                    Evaluated(
                        subject=subject, outcome="passed" if passes else "failed", evidence=evidence
                    )
                )
    status = (
        "failed"
        if any(isinstance(item, Evaluated) and item.outcome == "failed" for item in assessments)
        else "not_evaluated"
        if not assessments or any(isinstance(item, NotEvaluated) for item in assessments)
        else "passed"
    )
    return ParameterConvergenceReport(
        checked=len(chains.per_parameter),
        status=status,
        assessments=tuple(assessments),
        messages=tuple(
            item.detail
            if isinstance(item, NotEvaluated)
            else f"{item.evidence.criterion} fails for {item.evidence.note}: {item.evidence.value:g}"
            for item in assessments
            if isinstance(item, NotEvaluated) or item.outcome != "passed"
        ),
    )


def convergence_failures(report: ParameterConvergenceReport) -> tuple[str, ...]:
    """Action messages use the same measured verdict, without reconstruction."""
    return report.messages
