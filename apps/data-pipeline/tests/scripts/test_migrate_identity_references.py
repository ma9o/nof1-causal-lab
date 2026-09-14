"""Reference conversion preserves scientific identities and historical source pins."""

from scripts.migrate_identity_references import convert_payload

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.scenarios import ScenarioRequest
from nof1_causal_lab.artifacts.validation_report import ValidationIssue
from tests.helpers import make_model


def test_conversion_preserves_versions_sources_and_shared_endpoints():
    model = make_model(["x", "y", "z"], [("x", "y"), ("y", "z")])
    outcome = model.constructs[-1].id
    payload = model.model_dump(mode="json")
    payload["default_outcome"] = {"kind": "construct", "id": outcome}
    history = {
        "context": {"workspace": {"kind": "model", "id": "study"}, "seq": 12},
        "model": {
            "value": payload,
            "source": {"ref": {"artifact_id": "model", "version": 3}, "pointer": ""},
        },
        "provenance": {"model": {"workspace_id": "study", "version": 2}},
        "fit_source": {"ref": {"seq": 9}, "pointer": "/diagnostics/report"},
        "derived_from": {"model": 2},
        "disposition": {
            "target": {"kind": "construct", "id": outcome},
            "disposition": "retained_state",
            "reason": "Selected state",
        },
        "request": {
            "clamps": [
                {"target": {"kind": "construct", "id": outcome}, "mode": "shift", "amount": 1}
            ],
            "outcome": {"kind": "construct", "id": outcome},
        },
    }

    converted = convert_payload(history)
    assert history["context"]["workspace"] == {"kind": "model", "id": "study"}
    assert converted["context"] == {"workspace_id": "study", "seq": 12}
    assert converted["model"]["source"] == history["model"]["source"]
    for key in ("provenance", "fit_source", "derived_from", "disposition"):
        assert converted[key] == history[key]
    restored = ModelSpec.model_validate(converted["model"]["value"])
    assert restored.default_outcome == outcome
    assert restored.edges[0].effect is restored.edges[1].cause
    request = ScenarioRequest.model_validate(converted["request"])
    assert request.outcome == request.clamps[0].target == outcome
    assert convert_payload(converted) == converted


def test_conversion_preserves_indicator_and_dataset_issue_meanings():
    findings = [
        {"subject": subject, "issue_type": "check", "severity": "warning", "message": "Review"}
        for subject in (
            None,
            {"kind": "indicator", "id": "indicator:x"},
            {"kind": "construct", "id": "construct:x"},
        )
    ]
    converted = [ValidationIssue.model_validate(item) for item in convert_payload(findings)]
    assert [item.indicator_id for item in converted] == [None, "indicator:x", None]
    assert "construct:x" in converted[2].message
