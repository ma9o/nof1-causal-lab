"""Validate a whole-model candidate for the measurement structure operation."""

from nof1_causal_lab.flows.model_authoring import validate_model_submission
from nof1_causal_lab.json_types import JsonObject


def measurement_structure_grounding(
    data: object,
) -> tuple[JsonObject | None, str]:
    return validate_model_submission(data, measurements=True)
