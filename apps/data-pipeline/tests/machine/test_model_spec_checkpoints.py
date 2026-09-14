from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpyro.distributions as dist

from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.machine.moves import RunOperation
from nof1_causal_lab.machine.store import EpisodeJournal, ResumeRef, TransitionRecord
from nof1_causal_lab.machine.temporal.model_spec_checkpoints import (
    AcceptedConstructCheckpoint,
    ModelSpecAdmissionEvaluation,
    ModelSpecCheckpoint,
    latest_failed_model_spec_checkpoint_ref,
    model_spec_admission_evaluation_key,
    model_spec_admission_evaluation_path,
    read_model_spec_admission_evaluation,
    read_model_spec_checkpoint,
    rebase_accepted_constructs,
    restore_construct_state,
    write_accepted_model_spec_checkpoint,
    write_initial_model_spec_checkpoint,
    write_model_spec_admission_evaluation,
)
from tests.helpers import make_model
from tests.slot_fixtures import fixture_parameter_id

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


def _entity(name):
    from nof1_causal_lab.artifacts.coefficient import ParameterCoefficient
    from nof1_causal_lab.artifacts.state_distribution import InnovationSpec

    construct = make_model([name]).constructs[0]
    reference = ParameterCoefficient(
        parameter_id=fixture_parameter_id(SiteKind.DIFFUSION_DIAG, [ConstructRef(id=construct.id)])
    )
    return construct.model_copy(update={"innovation": InnovationSpec(scale=reference)})


def _parameters(name, scale):
    owner = ConstructRef(id=_entity(name).id)
    return (
        ParameterSpec(
            id=fixture_parameter_id(SiteKind.DIFFUSION_DIAG, [owner]),
            name=f"sigma_{name}",
            description="Test diffusion",
            distribution=dist.HalfNormal(scale),
        ),
    )


def _workspace(monkeypatch, tmp_path) -> str:
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "checkpoint-test"


def test_accepted_checkpoint_is_immutable_and_idempotent(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    pins: dict[ArtifactId, int] = {
        "question": 1,
        "identification_report": 2,
        "panel": 3,
        "validation_report": 3,
    }
    initial_ref = write_initial_model_spec_checkpoint(
        workspace_id=workspace_id,
        run_id="seq-000004",
        seq=4,
        pins=pins,
        accepted_constructs=[],
        search_queries={},
        search_cache={},
        repair_feedback={},
        parent_ref=None,
        rebase=None,
    )
    initial = read_model_spec_checkpoint(workspace_id, initial_ref)
    accepted = AcceptedConstructCheckpoint(
        edges=(),
        submission_id="tool-call-1",
        entity=_entity("sleep"),
        parameters=_parameters("sleep", 0.5),
        results=[
            {
                "check": "C2 latent scale",
                "target": "sleep",
                "value": "median scale 1.0",
                "band": "[0.33, 3.0]",
                "passed": True,
                "note": "ok",
                "diagnosis": [],
                "mode": "soft",
            }
        ],
        outcome="ADMITTED",
        feedback="accepted",
    )

    first_ref = write_accepted_model_spec_checkpoint(
        parent_ref=initial_ref,
        parent=initial,
        accepted=accepted,
        search_queries={},
        search_cache={},
    )
    retry_ref = write_accepted_model_spec_checkpoint(
        parent_ref=initial_ref,
        parent=initial,
        accepted=accepted,
        search_queries={},
        search_cache={},
    )

    assert retry_ref == first_ref
    checkpoint = read_model_spec_checkpoint(workspace_id, first_ref)
    assert checkpoint.parent_ref == initial_ref
    assert checkpoint.checkpoint_index == 1
    assert checkpoint.input_pins == pins
    assert [item.construct_name for item in checkpoint.accepted_constructs] == ["sleep"]
    assert checkpoint.accepted_constructs[0].results[0]["target"] == "sleep"


def test_admission_evaluation_key_is_scoped_to_causal_ancestors(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    accepted_a = AcceptedConstructCheckpoint(
        edges=(),
        submission_id="submission-a",
        entity=_entity("A"),
        parameters=_parameters("A", 0.2),
        outcome="ADMITTED",
        feedback="accepted",
    )
    accepted_b = AcceptedConstructCheckpoint(
        edges=(),
        submission_id="submission-b",
        entity=_entity("B"),
        parameters=_parameters("B", 0.3),
        outcome="ADMITTED",
        feedback="accepted",
    )
    checkpoint = ModelSpecCheckpoint(
        workspace_id=workspace_id,
        run_id="seq-000001",
        seq=1,
        checkpoint_index=2,
        input_pins={"panel": 3},
        accepted_constructs=[accepted_a, accepted_b],
        created_at="2026-07-13T00:00:00+00:00",
    )
    proposal: dict[str, Any] = {
        "ancestor_constructs": {"A"},
        "construct": _entity("X"),
        "edges": (),
        "construct_name": "X",
        "parameters": _parameters("X", 0.5),
        "accept": [],
        "n_draws": 200,
        "seed": 0,
    }

    key = model_spec_admission_evaluation_key(
        input_identity={"artifact_pins": checkpoint.input_pins},
        accepted_constructs=checkpoint.accepted_constructs,
        **proposal,
    )
    changed_sibling = checkpoint.model_copy(
        update={
            "accepted_constructs": [
                accepted_a,
                accepted_b.model_copy(update={"parameters": _parameters("B", 99.0)}),
            ]
        }
    )
    changed_ancestor = checkpoint.model_copy(
        update={
            "accepted_constructs": [
                accepted_a.model_copy(update={"parameters": _parameters("A", 99.0)}),
                accepted_b,
            ]
        }
    )

    assert (
        model_spec_admission_evaluation_key(
            input_identity={"artifact_pins": changed_sibling.input_pins},
            accepted_constructs=changed_sibling.accepted_constructs,
            **proposal,
        )
        == key
    )
    assert (
        model_spec_admission_evaluation_key(
            input_identity={"artifact_pins": changed_ancestor.input_pins},
            accepted_constructs=changed_ancestor.accepted_constructs,
            **proposal,
        )
        != key
    )

    path = model_spec_admission_evaluation_path(workspace_id, key)
    evaluation = ModelSpecAdmissionEvaluation(
        evaluation_key=key,
        construct_name="X",
        admitted=False,
        outcome="NEEDS REVISION",
        feedback="revise",
    )
    write_model_spec_admission_evaluation(path, evaluation)
    write_model_spec_admission_evaluation(path, evaluation)
    assert read_model_spec_admission_evaluation(path) == evaluation


def test_latest_failed_stage_four_checkpoint_is_the_resume_source(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    journal = EpisodeJournal(workspace_id)
    journal.append(
        TransitionRecord(
            seq=1,
            ts="2026-07-11T00:00:00+00:00",
            move=RunOperation(operation_id="statistical_model_spec"),
            status="raised",
            trace_ids=[],
            resume=ResumeRef(
                kind="model_spec",
                run_id="seq-000001",
                checkpoint_id="old.json",
            ),
        )
    )
    journal.append(
        TransitionRecord(
            seq=2,
            ts="2026-07-11T00:01:00+00:00",
            move=RunOperation(operation_id="posterior"),
            status="raised",
            trace_ids=[],
            resume=None,
        )
    )
    journal.append(
        TransitionRecord(
            seq=3,
            ts="2026-07-11T00:02:00+00:00",
            move=RunOperation(operation_id="statistical_model_spec"),
            status="raised",
            trace_ids=[],
            resume=ResumeRef(
                kind="model_spec",
                run_id="seq-000003",
                checkpoint_id="new.json",
            ),
        )
    )

    assert (
        latest_failed_model_spec_checkpoint_ref(workspace_id)
        == "model-spec-checkpoint:seq-000003/new.json"
    )

    journal.append(
        TransitionRecord(
            seq=4,
            ts="2026-07-11T00:03:00+00:00",
            move=RunOperation(operation_id="statistical_model_spec"),
            status="applied",
            trace_ids=[],
            resume=None,
        )
    )
    assert latest_failed_model_spec_checkpoint_ref(workspace_id) is None


def test_target_restore_uses_only_its_causal_ancestor_closure(monkeypatch):
    from nof1_causal_lab.flows.transitions.model_spec.agentic import construct_flow
    from nof1_causal_lab.models.ssm import construct_admission
    from nof1_causal_lab.models.ssm.construct_admission import AdmissionState

    class FakeState:
        def __init__(self, *, order, **_kwargs):
            self.order = order
            self.cursor = 0
            self.admission = AdmissionState(model=make_model(order))
            self.admitted_contributions = {}
            self.search_queries = {}
            self.search_cache = {}
            self.last_tool_feedback = None

        @property
        def current_construct(self):
            return self.order[self.cursor]

        def parameter_inventory_for(self, _construct):
            return SimpleNamespace(catalog=object())

    monkeypatch.setattr(construct_flow, "ConstructBuildState", FakeState)
    monkeypatch.setattr(
        construct_flow,
        "contribution_from_payload",
        lambda _design, payload: payload["construct"],
    )
    monkeypatch.setattr(
        construct_admission,
        "trial_admission_state",
        lambda admission, _contribution: admission,
    )
    checkpoint = ModelSpecCheckpoint(
        workspace_id="workspace",
        run_id="seq-000001",
        seq=1,
        checkpoint_index=2,
        input_pins={},
        accepted_constructs=[
            AcceptedConstructCheckpoint(
                edges=(),
                submission_id=f"submission-{name}",
                entity=_entity(name),
                parameters=(),
                outcome="ADMITTED",
                feedback="accepted",
            )
            for name in ("A", "B")
        ],
        created_at="2026-07-11T00:00:00+00:00",
    )
    model = ModelSpec.model_validate(
        make_model(
            ["A", "B", "X"],
            [("A", "X"), ("A", "B")],
        )
    )

    state = restore_construct_state(
        checkpoint,
        model=model,
        data_for_model=object(),
        workspace_id=None,
        target_construct="X",
    )

    assert state.order == ["A", "X"]
    assert set(state.admitted_contributions) == {"A"}


def test_rebase_retains_independent_branch_and_reopens_failed_descendants(monkeypatch):
    from nof1_causal_lab.machine.temporal import model_spec_checkpoints
    from nof1_causal_lab.machine.temporal.model_spec_checkpoints import ModelSpecCheckpoint
    from nof1_causal_lab.models.ssm import construct_admission

    class FakeState:
        def __init__(self, target):
            self.target = target
            self.search_queries = {}
            self.search_cache = {}
            self.last_report = None
            self.attempt = 0
            self.submission_made = False

        @property
        def current_construct(self):
            return self.target

        def submit_construct(self, *, construct, **_kwargs):
            if construct["name"] == "stress":
                return "stress no longer passes the scale check"
            self.target = None
            self.last_report = SimpleNamespace(
                name=construct["name"],
                admitted=True,
                annotations=(),
                outcome="ADMITTED",
            )
            return "accepted"

    monkeypatch.setattr(
        construct_admission,
        "build_construct_order",
        lambda _structure: ["stress", "sleep", "mood"],
    )
    monkeypatch.setattr(
        construct_admission,
        "build_construct_units",
        lambda _structure: [
            SimpleNamespace(unit_id="stress", constructs=("stress",), predecessors=()),
            SimpleNamespace(unit_id="sleep", constructs=("sleep",), predecessors=()),
            SimpleNamespace(unit_id="mood", constructs=("mood",), predecessors=("stress",)),
        ],
    )
    monkeypatch.setattr(
        model_spec_checkpoints,
        "restore_construct_state",
        lambda _checkpoint, **kwargs: FakeState(kwargs.get("target_construct")),
    )
    accepted = [
        AcceptedConstructCheckpoint(
            edges=(),
            submission_id=f"submission-{name}",
            entity=_entity(name),
            parameters=(),
            outcome="ADMITTED",
            feedback="accepted",
        )
        for name in ("stress", "sleep", "mood")
    ]
    source = ModelSpecCheckpoint(
        workspace_id="workspace",
        run_id="seq-000001",
        seq=1,
        checkpoint_index=3,
        input_pins={
            "question": 1,
            "identification_report": 1,
            "panel": 1,
            "validation_report": 1,
        },
        accepted_constructs=accepted,
        created_at="2026-07-11T00:00:00+00:00",
    )

    _state, retained, reopened, reason = rebase_accepted_constructs(
        source,
        model=ModelSpec.model_validate(make_model(["stress"], [])),
        data_for_model=object(),
    )

    assert [item.construct_name for item in retained] == ["sleep"]
    assert reopened == "stress"
    assert reason == "stress no longer passes the scale check"


def test_admission_evaluation_key_tracks_fixed_mechanism_choices():
    from nof1_causal_lab.artifacts.coefficient import FixedCoefficient

    mechanism = DynamicsMechanism(
        id="mechanism:checkpoint-hill",
        expression=expr_hill(
            expr_state("construct:A"),
            emax=FixedCoefficient(value=1),
            ec50=FixedCoefficient(value=1),
            n=FixedCoefficient(value=2),
        ),
    )

    def key(mechanism):
        return model_spec_admission_evaluation_key(
            input_identity={},
            accepted_constructs=[],
            ancestor_constructs=set(),
            construct_name="B",
            construct=_entity("B"),
            parameters=(),
            edges=(
                make_model(["A", "B"], [("A", "B")])
                .edges[0]
                .model_copy(update={"mechanisms": (mechanism,)}),
            ),
            accept=[],
            n_draws=16,
            seed=7,
        )

    from nof1_causal_lab.artifacts.expressions import CoefficientExpression, map_expression

    changed = map_expression(
        mechanism.expression,
        lambda node: (
            CoefficientExpression(role="exponent", coefficient=FixedCoefficient(value=3))
            if isinstance(node, CoefficientExpression) and node.role == "exponent"
            else node
        ),
    )
    assert key(mechanism) != key(mechanism.model_copy(update={"expression": changed}))
