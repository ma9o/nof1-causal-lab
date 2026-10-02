"""Shared configuration for owned scientific and API values."""

from typing import Self

from pydantic import BaseModel, ConfigDict


class Value(BaseModel):
    """An owned value has frozen fields, rejects extra fields and shares output presence.

    Owned collections expose read-only interfaces; native arrays and foreign
    objects keep their documented ownership constraints.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_serialization_defaults_required=True
    )

    def revised(self, **changes: object) -> Self:
        """Validate a new value at its concrete owner, retaining no derived caches."""
        return self.model_validate(
            {**{name: getattr(self, name) for name in type(self).model_fields}, **changes}
        )
