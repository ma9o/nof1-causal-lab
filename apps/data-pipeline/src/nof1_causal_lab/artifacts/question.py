"""The research question that roots an episode."""

from pydantic import Field, field_validator

from .base import ArtifactPayload


class QuestionArtifact(ArtifactPayload):
    """A research question states the observational causal question under investigation."""

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question text must be non-empty")
        return value
