"""Base contract for persisted scientific artifacts."""

from pydantic import BaseModel, ConfigDict


class ArtifactPayload(BaseModel):
    """Shared base for persisted artifact payloads.

    Contracts are pure artifacts: execution failure is a typed exception on
    the transition (state unchanged, attempt journaled), and negative
    findings are report-present / enabling-artifact-absent — never an
    ``outcome`` enum on the payload.
    """

    model_config = ConfigDict(extra="forbid")
