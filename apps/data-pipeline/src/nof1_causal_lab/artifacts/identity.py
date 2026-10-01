"""Persistent authored entity IDs, independent of names and model revisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, NewType, cast, get_args, overload

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler

if TYPE_CHECKING:
    from pydantic_core import core_schema


@dataclass(frozen=True)
class _IdentitySchema:
    """Keep nominal string IDs named in JSON Schema and generated clients."""

    name: str

    def __get_pydantic_core_schema__(
        self, source: type[str], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        schema = cast("core_schema.StringSchema", handler(source))
        schema["ref"] = self.name
        return schema


GitOid = NewType(
    "GitOid",
    Annotated[str, Field(pattern=r"^[0-9a-f]{40}$"), _IdentitySchema("GitOid")],
)


ConstructId = NewType(
    "ConstructId",
    Annotated[
        str,
        Field(
            pattern=r"^construct:[A-Za-z0-9_.-]+$",
            json_schema_extra={"tsType": "`construct:${string}`"},
        ),
        _IdentitySchema("ConstructId"),
    ],
)
EdgeId = NewType(
    "EdgeId",
    Annotated[
        str,
        Field(pattern=r"^edge:[A-Za-z0-9_.-]+$", json_schema_extra={"tsType": "`edge:${string}`"}),
        _IdentitySchema("EdgeId"),
    ],
)
IndicatorId = NewType(
    "IndicatorId",
    Annotated[
        str,
        Field(
            pattern=r"^indicator:[A-Za-z0-9_.-]+$",
            json_schema_extra={"tsType": "`indicator:${string}`"},
        ),
        _IdentitySchema("IndicatorId"),
    ],
)

MechanismId = NewType(
    "MechanismId",
    Annotated[
        str,
        Field(
            pattern=r"^mechanism:[A-Za-z0-9_.-]+$",
            json_schema_extra={"tsType": "`mechanism:${string}`"},
        ),
        _IdentitySchema("MechanismId"),
    ],
)

DistributionId = NewType(
    "DistributionId",
    Annotated[
        str,
        Field(
            pattern=r"^distribution:[A-Za-z0-9_.-]+$",
            description="A native law whose membership is defined by the model's scientific quantities.",
            json_schema_extra={"tsType": "`distribution:${string}`"},
        ),
        _IdentitySchema("DistributionId"),
    ],
)


ParameterId = NewType(
    "ParameterId",
    Annotated[
        str,
        Field(
            pattern=r"^parameter:[0-9a-f]{64}$",
            json_schema_extra={"tsType": "`parameter:${string}`"},
        ),
        _IdentitySchema("ParameterId"),
    ],
)

ParameterElementId = NewType(
    "ParameterElementId",
    Annotated[
        str,
        Field(
            pattern=r"^element:[0-9a-f]{64}$", json_schema_extra={"tsType": "`element:${string}`"}
        ),
        _IdentitySchema("ParameterElementId"),
    ],
)


type ArtifactId = Literal[
    "raw_data",
    "model",
    "identification_report",
    "panel",
    "data_profile",
    "validation_report",
]

# Actions name work; several actions can enrich the same model artifact.
type ScientificActionId = Literal["edit_model", "prepare_data", "fit", "simulate"]
type ActionId = ScientificActionId | Literal["data_diff"]

ARTIFACT_IDS: tuple[ArtifactId, ...] = get_args(ArtifactId.__value__)
SCIENTIFIC_ACTION_IDS: tuple[ScientificActionId, ...] = get_args(ScientificActionId.__value__)


class IdentityRef(BaseModel):
    """References distinguish identity from the authored values at a particular revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GitRef(IdentityRef):
    """An exact file in a study's Git object database: repository, object, and path."""

    workspace_id: str = Field(min_length=1)
    revision: GitOid
    path: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")


class ConstructRef(IdentityRef):
    """A construct reference identifies a construct independently of its current name or
    revision.
    """

    kind: Literal["construct"] = "construct"
    id: ConstructId


class EdgeRef(IdentityRef):
    """An edge reference identifies a causal relationship independently of edits to its
    definition.
    """

    kind: Literal["edge"] = "edge"
    id: EdgeId


class IndicatorRef(IdentityRef):
    """An indicator reference identifies a measurement definition independently of its name or
    revision.
    """

    kind: Literal["indicator"] = "indicator"
    id: IndicatorId


class MechanismRef(IdentityRef):
    """A particular additive term, independently of its position or coefficient values."""

    kind: Literal["mechanism"] = "mechanism"
    id: MechanismId


class ParameterRef(IdentityRef):
    """A scalar finding identifies its scientific parameter and declared logical component."""

    parameter_id: ParameterId
    element_id: ParameterElementId


type EntityRef = Annotated[
    ConstructRef | EdgeRef | IndicatorRef | MechanismRef, Field(discriminator="kind")
]


@overload
def scientific_id(prefix: Literal["construct"], payload: object) -> ConstructId: ...


@overload
def scientific_id(prefix: Literal["edge"], payload: object) -> EdgeId: ...


@overload
def scientific_id(prefix: Literal["indicator"], payload: object) -> IndicatorId: ...


@overload
def scientific_id(prefix: Literal["mechanism"], payload: object) -> MechanismId: ...


@overload
def scientific_id(prefix: Literal["distribution"], payload: object) -> DistributionId: ...


@overload
def scientific_id(prefix: Literal["parameter"], payload: object) -> ParameterId: ...


@overload
def scientific_id(prefix: Literal["element"], payload: object) -> ParameterElementId: ...


@overload
def scientific_id(prefix: str, payload: object) -> str: ...


def scientific_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}:{hashlib.sha256(encoded.encode()).hexdigest()}"
