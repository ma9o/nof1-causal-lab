"""Validate a whole-model candidate for the latent structure operation."""

from nof1_causal_lab.flows.model_authoring import validate_model_submission
from nof1_causal_lab.json_types import UncheckedJsonObject


def latent_structure_grounding(data: UncheckedJsonObject) -> tuple[UncheckedJsonObject | None, str]:
    return validate_model_submission(data, measurements=False)
