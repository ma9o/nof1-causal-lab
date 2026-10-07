"""Authoring defaults and resolved windows share one observation owner."""

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.availability import Evaluation
from nof1_causal_lab.artifacts.duration import Duration
from nof1_causal_lab.artifacts.observations import AuthoredObservationSpec, ResolvedObservationSpec

pytestmark = pytest.mark.contract


def test_resolution_preserves_the_definition_and_requires_its_window():
    authored = AuthoredObservationSpec.model_validate(
        {"id": "indicator:x", "name": "X", "measurement_dtype": "continuous", "aggregation": "last"}
    )
    assert authored.observation_window is None
    resolved = authored.resolved(Duration("6h"))
    assert resolved.observation_window == Duration("6h")
    assert ResolvedObservationSpec.model_validate_json(resolved.model_dump_json()) == resolved
    assert resolved.model_dump(exclude={"observation_window"}) == authored.model_dump(
        exclude={"observation_window"}
    )
    assert "observation_window" not in AuthoredObservationSpec.model_json_schema()["required"]
    assert "observation_window" in ResolvedObservationSpec.model_json_schema()["required"]
    for value in (authored.model_dump(), authored.model_dump(exclude={"observation_window"})):
        with pytest.raises(ValidationError):
            ResolvedObservationSpec.model_validate(value)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "available", "value": []},
        {"kind": "unavailable", "reason": "No retained draws."},
        {"kind": "not_applicable", "reason": "No intervention was requested."},
    ],
)
def test_availability_preserves_empty_payloads_and_explicit_absence(payload):
    adapter = TypeAdapter(Evaluation[tuple[str, ...]])
    value = adapter.validate_python(payload)
    assert adapter.validate_json(adapter.dump_json(value)) == value


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "available", "value": [], "reason": "Unavailable"},
        {"kind": "unavailable", "reason": "Unavailable", "value": []},
        {"kind": "not_applicable", "reason": "Not applicable", "value": []},
        {"kind": "available"},
    ],
)
def test_availability_rejects_contradictory_or_missing_payloads(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(Evaluation[tuple[str, ...]]).validate_python(payload)
