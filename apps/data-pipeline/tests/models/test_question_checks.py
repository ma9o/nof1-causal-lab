"""The question's quick checks report what the model and the record still lack."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nof1_causal_lab.actions.io import EditModelInput
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import state
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec, LikelihoodSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.models.model_checks import question_edit_reason
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
    return model.with_entities(
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


def test_model_only_checks_undefined_constructs_without_data_findings():
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
    }
    reasons = {item.subject.check: item.reason for item in findings if item.kind == "not_evaluated"}
    assert reasons["outcome"] == "CONSTRUCT_UNDEFINED"


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


@pytest.mark.parametrize(
    ("extra_edges", "outside"),
    [
        ([("Dose", "Side")], ("Side",)),
        ([("Mood", "Side")], ("Side",)),
        ([("Dose", "Side"), ("Side", "Loop"), ("Loop", "Side")], ("Loop", "Side")),
    ],
)
def test_an_edit_rejects_connected_branches_without_a_path_to_the_outcome(extra_edges, outside):
    model = make_model(["Dose", "Mood", *outside], [("Dose", "Mood"), *extra_edges])
    draft = model.revised(measurement_clock=None).with_entities(
        edges=replace_constructs(
            model.edges, tuple(construct.revised(indicators=()) for construct in model.constructs)
        ),
    )
    reason = question_edit_reason(draft, _question(_model()))
    assert reason is not None
    assert "directed path to the question's outcome 'Mood'" in reason
    assert reason.split("No directed path: ")[1] == ", ".join(
        sorted(
            f"{construct.name!r} ({construct.id})"
            for construct in draft.constructs
            if construct.name in outside
        )
    )


def test_an_edit_allows_upstream_causes_confounders_and_feedback_that_reaches_the_outcome():
    model = make_model(
        ["U", "Dose", "Mediator", "Mood", "Moderator"],
        [
            ("U", "Dose"),
            ("U", "Mood"),
            ("Dose", "Mediator"),
            ("Mediator", "Mood"),
            ("Mood", "Mediator"),
            ("Moderator", "Mood"),
        ],
    ).revised(measurement_clock=None)
    confounder = next(construct for construct in model.constructs if construct.name == "U")
    model = model.with_entities(
        edges=replace_constructs(model.edges, (confounder.revised(indicators=()),))
    )
    assert question_edit_reason(model, _question(_model())) is None


def test_an_off_path_model_edit_is_rejected_before_writing_a_revision(tmp_path, monkeypatch):
    from nof1_causal_lab.actions.contracts import EditModelRequest
    from nof1_causal_lab.actions.edit_model import edit_model
    from nof1_causal_lab.study.records import Rejected
    from nof1_causal_lab.study.state import StudyState
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data
    from tests.helpers import write_question

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("question-path")
    state = StudyState().with_artifacts([write_question(store, _question(_model()))])
    before = frozenset(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    model = make_model(["Dose", "Mood", "Side"], [("Dose", "Mood"), ("Dose", "Side")])
    result = edit_model(
        store.workspace_id,
        EditModelRequest[GitOid](
            input=EditModelInput[GitOid](parent_ref=state.current["question"].revision, model=model)
        ),
    )
    assert isinstance(result, Rejected)
    assert result.reason == "scientific_inputs"
    assert "'Side'" in result.detail
    assert frozenset(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
