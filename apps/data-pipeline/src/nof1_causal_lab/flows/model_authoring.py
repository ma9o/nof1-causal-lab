"""Shared whole-model submissions for scientific authoring operations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.json_types import JsonObject  # noqa: TC001


class ModelSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_json: str = Field(
        description="The full candidate ModelSpec as JSON, preserving existing entity IDs and all retained details."
    )


def validate_model_submission(
    data: object, *, measurements: bool = False
) -> tuple[JsonObject | None, str]:
    try:
        model = ModelSpec.model_validate(data)
        if not model.constructs:
            raise IncompleteModelError("The proposal requires at least one construct")
        if measurements:
            from nof1_causal_lab.models.model_checks import (
                collect_measurement_compile_errors,
            )

            model.require_measurements()
            errors = collect_measurement_compile_errors(model)
            if errors:
                return None, "VALIDATION ERRORS:\n" + "\n".join(errors)
            model.require_execution_structure()
    except (ValidationError, ValueError) as exc:
        return None, f"VALIDATION ERRORS:\n{exc}"
    return model.model_dump(mode="json"), "VALID"
