"""Test validation against the production artifact catalog."""

from typing import Any

from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS


def validate_artifact_payload(artifact_id: str, data: dict[str, Any]) -> dict[str, Any]:
    if artifact_id not in ARTIFACT_CONTRACTS:
        known = ", ".join(sorted(ARTIFACT_CONTRACTS))
        raise ValueError(f"Unknown artifact_id '{artifact_id}'. Expected one of: {known}")
    return ARTIFACT_CONTRACTS[artifact_id].model_validate(data).model_dump(mode="json")
