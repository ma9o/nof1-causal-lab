"""Persistent authored entity IDs, independent of names and model revisions."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

type ConstructId = Annotated[
    str,
    Field(
        pattern=r"^construct:[A-Za-z0-9_.-]+$",
        json_schema_extra={"tsType": "`construct:${string}`"},
    ),
]
type EdgeId = Annotated[
    str,
    Field(pattern=r"^edge:[A-Za-z0-9_.-]+$", json_schema_extra={"tsType": "`edge:${string}`"}),
]
type IndicatorId = Annotated[
    str,
    Field(
        pattern=r"^indicator:[A-Za-z0-9_.-]+$",
        json_schema_extra={"tsType": "`indicator:${string}`"},
    ),
]

type MechanismId = Annotated[
    str,
    Field(
        pattern=r"^mechanism:[A-Za-z0-9_.-]+$",
        json_schema_extra={"tsType": "`mechanism:${string}`"},
    ),
]

type DistributionId = Annotated[
    str,
    Field(
        pattern=r"^distribution:[A-Za-z0-9_.-]+$",
        description="A shared native law whose membership is defined by the model's scientific quantities.",
        json_schema_extra={"tsType": "`distribution:${string}`"},
    ),
]


type ParameterId = Annotated[
    str,
    Field(
        pattern=r"^parameter:[0-9a-f]{64}$", json_schema_extra={"tsType": "`parameter:${string}`"}
    ),
]

type ParameterElementId = Annotated[
    str,
    Field(pattern=r"^element:[0-9a-f]{64}$", json_schema_extra={"tsType": "`element:${string}`"}),
]


type ArtifactId = Literal[
    "question",
    "raw_data",
    "model",
    "identification_report",
    "panel",
    "validation_report",
    "admission_report",
    "baseline_report",
]

# Operations name work; several operations can enrich the same model artifact.
type OperationId = Literal[
    "raw_data",
    "latent_structure",
    "measurement_structure",
    "measurements",
    "statistical_model_spec",
    "posterior",
    "baseline_report",
]

ARTIFACT_IDS: tuple[ArtifactId, ...] = get_args(ArtifactId.__value__)


class IdentityRef(BaseModel):
    """References distinguish identity from the authored values at a particular revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRevision(IdentityRef):
    """The workspace and version of the scientific design supporting an inference."""

    workspace_id: str = Field(min_length=1)
    version: int = Field(ge=1)


class ModelRef(IdentityRef):
    """A model reference identifies the workspace that owns the scientific model."""

    kind: Literal["model"] = "model"
    id: str = Field(min_length=1)


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


class ArtifactRef(IdentityRef):
    """An artifact reference identifies the exact stored version that supports a model fact."""

    artifact_id: ArtifactId
    version: int = Field(ge=1)


class ParameterRef(IdentityRef):
    """A scalar finding identifies its scientific parameter and declared logical component."""

    parameter_id: ParameterId
    element_id: ParameterElementId


type EntityRef = Annotated[
    ConstructRef | EdgeRef | IndicatorRef | MechanismRef, Field(discriminator="kind")
]


def scientific_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}:{hashlib.sha256(encoded.encode()).hexdigest()}"


class TransitionRef(IdentityRef):
    """An immutable entry in the workspace transition journal."""

    seq: int = Field(ge=1)
