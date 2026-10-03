"""The question's quick checks report what the model and the record still lack."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import state
from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec, LikelihoodSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.question_checks import question_findings
from tests.helpers import make_model

pytestmark = pytest.mark.contract

_ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def _model():
    """A given dose read exactly each day, and the mood it may change."""
    model = make_model(["Dose", "Mood"], [("Dose", "Mood")])
    dose = model.constructs[0]
    reading = dose.indicators[0].revised(
        likelihood=LikelihoodSpec(
            law=DeltaLawSpec(v=state(dose.id)), reasoning="The dose is read exactly."
        )
    )
    return model.revised(
        edges=replace_constructs(
            model.edges, (dose.revised(role="exogenous", indicators=(reading,)),)
        )
    )


def _question(model, *, start="2024-01-05", value=0.0, outcome=None):
    dose, mood = model.constructs
    return QuestionSpec(
        text="Does lowering my dose change my mood?",
        outcome=outcome or mood.id,
        queries={
            "taper": {
                "start": start,
                "horizon": "2w",
                "interventions": [{"target": dose.id, "after": "1w", "value": value}],
            }
        },
    )


def _panel(model):
    dose = model.constructs[0]
    days = range(10)
    return pl.DataFrame(
        {
            "indicator_id": [dose.indicators[0].observation.id] * len(days),
            "value": [10.0 if day < 5 else 20.0 for day in days],
            "support_start": [(_ORIGIN + timedelta(days=day)).replace(tzinfo=None) for day in days],
            "support_end": [
                (_ORIGIN + timedelta(days=day + 1)).replace(tzinfo=None) for day in days
            ],
        }
    )


def _findings(question, model, panel):
    return {
        (finding.subject.check, finding.kind, getattr(finding, "outcome", finding.kind))
        for finding in question_findings(
            question,
            StructuralSelection.for_question(model, question),
            panel=panel,
            time_origin=_ORIGIN if panel is not None else None,
        )
    }


def test_undefined_constructs_and_a_missing_record_are_not_evaluated():
    model = make_model(["Stress"])
    question = _question(_model())
    findings = question_findings(
        question,
        StructuralSelection.for_question(model, question),
        panel=None,
        time_origin=None,
    )
    assert {(item.subject.check, item.kind) for item in findings} == {
        ("outcome", "not_evaluated"),
        ("target", "not_evaluated"),
        ("identification", "not_evaluated"),
        ("window", "not_evaluated"),
        ("range", "not_evaluated"),
    }
    reasons = {item.subject.check: item.reason for item in findings if item.kind == "not_evaluated"}
    assert reasons["outcome"] == reasons["range"] == "CONSTRUCT_UNDEFINED"
    assert reasons["window"] == "NO_PANEL"


def test_queries_are_checked_against_the_model_and_the_record():
    model = _model()
    panel = _panel(model)
    assert _findings(_question(model, value=15.0), model, panel) == {
        ("outcome", "evaluated", "passed"),
        ("target", "evaluated", "passed"),
        ("identification", "evaluated", "passed"),
        ("window", "evaluated", "passed"),
        ("range", "evaluated", "passed"),
    }
    outside = _findings(_question(model, start="2023-12-25", value=0.0), model, panel)
    assert ("window", "evaluated", "failed") in outside
    assert ("range", "evaluated", "failed") in outside
    dose = model.constructs[0]
    with pytest.raises(ValueError, match="intervenes on the outcome"):
        _question(model, outcome=dose.id)
    exogenous_outcome = QuestionSpec(text="Does the dose change?", outcome=dose.id)
    assert _findings(exogenous_outcome, model, panel) == {("outcome", "evaluated", "failed")}


def test_an_edit_defines_the_question_and_models_its_outcome():
    from nof1_causal_lab.actions.edit_model import question_edit_reason

    model = _model()
    dose, _ = model.constructs
    assert question_edit_reason(model, _question(model)) is None
    missing = question_edit_reason(make_model(["Mood"]), _question(model))
    assert missing is not None
    assert dose.id in missing
    given = QuestionSpec(text="Does my dose change?", outcome=dose.id)
    assert (
        question_edit_reason(model, given)
        == "The question's outcome must reference an endogenous construct"
    )
