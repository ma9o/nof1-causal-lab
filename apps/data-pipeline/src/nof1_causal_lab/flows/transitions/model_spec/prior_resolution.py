"""Application projection between executable priors and evidence-rich proposals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.prior import ExecutablePrior, PriorPlan
from nof1_causal_lab.artifacts.prior_proposal import PriorProposal
from nof1_causal_lab.json_types import UncheckedJsonObject
from nof1_causal_lab.models.ssm.compile.artifact import resolve_executable_priors

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact

type PriorPayload = UncheckedJsonObject


def resolve_prior_proposals(
    compiled_ssm: CompiledSSMArtifact,
    *,
    authored_priors: Mapping[str, PriorPayload],
) -> list[PriorPayload]:
    """Combine compiler-owned prior membership with authored evidence metadata."""
    definitions = {parameter.id: parameter for parameter in compiled_ssm.parameters}
    ids_by_name = {parameter.name: parameter.id for parameter in compiled_ssm.parameters}
    authored_by_parameter = {
        ids_by_name[parameter]: PriorProposal.model_validate(
            {**payload, "parameter_id": ids_by_name[parameter]}
        )
        for parameter, payload in authored_priors.items()
    }
    authored_plan = PriorPlan(
        priors={
            parameter: ExecutablePrior(
                parameter_id=parameter,
                distribution=proposal.distribution,
                params=proposal.params,
                reference_interval_days=proposal.reference_interval_days,
            )
            for parameter, proposal in authored_by_parameter.items()
        }
    )
    resolved = resolve_executable_priors(compiled_ssm, authored_plan=authored_plan)

    rows: list[PriorPayload] = []
    for prior in resolved:
        authored = authored_by_parameter.get(prior.parameter_id)
        if authored is not None:
            rows.append(authored.model_dump(mode="json"))
            continue
        rows.append(
            PriorProposal(
                parameter_id=definitions[prior.parameter_id].id,
                distribution=prior.distribution,
                params=prior.params,
                sources=[],
                reasoning=f"Compiler-resolved prior for {definitions[prior.parameter_id].name}.",
                reference_interval_days=prior.reference_interval_days,
            ).model_dump(mode="json")
        )
    return rows


__all__ = ["resolve_prior_proposals"]
