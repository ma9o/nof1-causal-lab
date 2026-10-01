"""Tests for the project-specific type-boundary checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract


def _load_checker() -> Any:
    module_name = "check_type_boundaries_under_test"
    path = Path(__file__).resolve().parents[3] / "scripts" / "checks" / "check_type_boundaries.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_any_member_makes_union_a_violation_even_inside_generic() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

def parse(value: tuple[str, Any | None]) -> Any | int:
    ...
""",
        path="src/example.py",
    )

    assert [violation.code for violation in violations] == ["CUSTOM001", "CUSTOM001"]
    assert {violation.annotation for violation in violations} == {
        "Any | None",
        "Any | int",
    }


def test_typing_unions_optional_and_import_aliases_are_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
import typing as t
from typing import Any as Dynamic
from typing import Optional as Maybe
from typing import Union as Either

type Loose = Dynamic
type Looser = Loose

def parse(
    first: Either[str, Dynamic],
    second: t.Optional[t.Any],
    third: Maybe[Looser],
    fourth: Looser | int,
    valid: list[Dynamic] | None,
) -> None:
    ...
""",
        path="src/example.py",
    )

    assert [violation.annotation for violation in violations] == [
        "Either[str, Dynamic]",
        "t.Optional[t.Any]",
        "Maybe[Looser]",
        "Looser | int",
    ]


def test_legacy_explicit_type_alias_value_is_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any, TypeAlias, Union

Loose: TypeAlias = Union[str, Any]
""",
        path="src/example.py",
    )

    assert [violation.code for violation in violations] == ["CUSTOM001"]
    assert violations[0].target == "alias:Loose"
    assert violations[0].annotation == "Union[str, Any]"


def test_rules_can_be_selected_independently() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

def compile_plan(
    plan: ModelSpec | dict[str, Any],
    fallback: Any | None,
) -> None:
    ...
""",
        path="tests/example.py",
        rules=frozenset({"CUSTOM001"}),
    )

    assert [violation.code for violation in violations] == ["CUSTOM001"]
    assert violations[0].target == "parameter:fallback"


def test_domain_model_must_not_be_union_member_with_dict() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any, Union
from pydantic import BaseModel

class Indicator(BaseModel):
    name: str

def compile_plan(
    plan: ModelSpec | dict[str, Any] | None,
    indicator: Indicator | dict[str, Any],
    legacy: Union[Indicator, dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> None:
    ...
        """,
        path="src/example.py",
        rules=frozenset({"CUSTOM002"}),
    )

    assert [violation.code for violation in violations] == [
        "CUSTOM002",
        "CUSTOM002",
        "CUSTOM002",
    ]
    assert {violation.target for violation in violations} == {
        "parameter:indicator",
        "parameter:legacy",
        "parameter:plan",
    }


def test_recursive_json_type_alias_is_not_treated_as_domain_union() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

def register(value: dict | Callable) -> None:
    ...
""",
        path="src/example.py",
    )

    assert violations == []


def test_anonymous_any_dictionary_requires_a_named_boundary() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

def parse(
    payload: dict[str, Any],
    nested: list[dict[str, Any]] | None,
    indirect: dict[str, tuple[Any, Any]],
) -> dict[str, Any]:
    local: dict[str, Any] = {}
    return local
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM004"}),
    )

    assert [violation.target for violation in violations] == [
        "parameter:payload",
        "parameter:nested",
        "parameter:indirect",
        "return",
        "variable:local",
    ]
    assert all(violation.code == "CUSTOM004" for violation in violations)


def test_unsafe_dictionary_requires_an_explicitly_unsafe_or_domain_name() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any

type JsonObject = dict[str, Any]
type UncheckedJsonObject = dict[str, Any]
type RuntimeMap = dict[str, Any]

def parse(payload: UncheckedJsonObject) -> RuntimeMap:
    return payload
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM004"}),
    )

    assert [violation.target for violation in violations] == ["alias:JsonObject"]


@pytest.mark.parametrize("path", ["src/new_boundary.py", "scripts/new_reader.py"])
def test_unchecked_json_is_local_to_validation(path: str) -> None:
    violations = _load_checker().scan_text(
        """
from nof1_causal_lab.json_types import UncheckedJsonObject as RawJson

def parse(payload: RawJson) -> ModelSpec:
    return ModelSpec.model_validate(payload)

def read(text: str) -> ModelSpec:
    payload: RawJson = json.loads(text)
    return TypeAdapter(ModelSpec).validate_python(payload)
""",
        path=path,
        rules=frozenset({"CUSTOM005"}),
    )
    assert violations == []


def test_unchecked_json_cannot_escape_or_skip_validation() -> None:
    violations = _load_checker().scan_text(
        """
from nof1_causal_lab.json_types import UncheckedJsonObject as RawJson

type DomainPayload = RawJson

class State:
    payload: RawJson

def passthrough(payload: RawJson) -> RawJson:
    return payload

def fake_parser(payload: RawJson) -> ModelSpec:
    ModelSpec.model_validate(payload)
    save(payload)
    return result

def leaking_parser(payload: RawJson) -> ModelSpec:
    return ModelSpec.model_validate(save(payload))

def non_input_argument(payload: RawJson) -> ModelSpec:
    return ModelSpec.model_validate({}, context=payload)
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM005"}),
    )
    assert [item.target for item in violations] == [
        "alias:DomainPayload",
        "variable:payload",
        "parameter:payload",
        "return",
        "parameter:payload",
        "parameter:payload",
        "parameter:payload",
    ]


def test_reject_only_optional_parameter_is_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        '''
from typing import Optional as Maybe

def compile_plan(plan: Maybe[ModelSpec]) -> None:
    """Compile an already validated plan."""
    if plan is None:
        raise ValueError("plan is required")

def compile_spec(spec: StatisticalModelSpec | None = None) -> None:
    if None is spec:
        raise ValueError("spec is required")

def compile_after_audit(plan: ModelSpec | None) -> None:
    audit_request()
    audit_complete = True
    if plan is None:
        raise ValueError("plan is required")
''',
        path="src/example.py",
    )

    assert [violation.code for violation in violations] == [
        "CUSTOM003",
        "CUSTOM003",
        "CUSTOM003",
    ]
    assert {violation.target for violation in violations} == {
        "parameter:plan",
        "parameter:spec",
    }


def test_conditional_or_non_rejecting_optional_parameter_is_allowed() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
def conditionally_require_plan(
    plan: ModelSpec | None,
    *,
    enabled: bool,
) -> None:
    if enabled:
        if plan is None:
            raise ValueError("enabled compilation requires a plan")

def default_plan(plan: ModelSpec | None = None) -> None:
    if plan is None:
        return

def return_before_rejection(plan: ModelSpec | None, enabled: bool) -> None:
    if not enabled:
        return
    if plan is None:
        raise ValueError("plan is required")

def replace_before_rejection(plan: ModelSpec | None) -> None:
    if plan is None:
        plan = default_plan()
    if plan is None:
        raise ValueError("plan is required")
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM003"}),
    )

    assert violations == []


@pytest.mark.parametrize(
    "operation",
    [
        "value.model_construct()",
        "value.model_copy(update={})",
        "value.model_copy(**changes)",
        "object.__setattr__(value, 'question', 'changed')",
        "Evidence(spec=value)",
        "IncompleteModel('missing')",
        "UnsupportedFit(('unsupported',))",
        "cast('Evidence', value)",
        "cast('IncompleteModel', value)",
        "cast('UnsupportedFit', value)",
        "cast('list[int]', value)",
        "cast('dict[str, str]', value)",
        "cast('set[int]', value)",
        "cast('MutableSequence[int]', value)",
    ],
)
def test_core_bypasses_match_calls_without_receiver_types(operation: str) -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        f"""from typing import cast
from collections.abc import MutableSequence
from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs as Evidence

def consumer(value):
    return {operation}
""",
        path="tests/consumer.py",
        rules=frozenset({"CORE001"}),
    )
    assert len(violations) == 1
    assert violations[0].code == "CORE001"
    assert "owner" in violations[0].message


def test_core_bypasses_allow_self_revision_and_unmodified_copies() -> None:
    checker = _load_checker()
    source = """class Request:
    def revise(self, value, changes):
        self.model_copy(update={})
        self.model_copy(**changes)
        object.__setattr__(self, 'question', 'changed')
        value.model_copy()
        cast('Mapping[str, int]', value)
"""
    assert (
        checker.scan_text(
            source,
            path="src/nof1_causal_lab/artifacts/model_spec.py",
            rules=frozenset({"CORE001"}),
        )
        == []
    )


def test_core_owner_file_only_exempts_evidence_construction() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """CompiledFitInputs()
IncompleteModel('missing')
UnsupportedFit(('unsupported',))
value.model_copy(update={})
value.model_construct()
cast('CompiledFitInputs', value)
""",
        path="src/nof1_causal_lab/models/ssm/compile/inputs.py",
        rules=frozenset({"CORE001"}),
    )
    assert [v.target for v in violations] == ["model_copy", "model_construct", "cast"]


def test_core_collections_cover_fields_and_public_properties_only() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """from typing import Dict as MutableDict
class ModelSpec:
    entries: tuple[MutableDict[str, int], ...]
    readonly: Mapping[str, int]
    _builder: dict[str, int]
    @cached_property
    def derived(self) -> dict[str, int]: ...
    @property
    def safe(self) -> tuple[int, ...]: ...
    def _build(self) -> list[int]: ...
    def edit(self) -> None:
        local: list[int] = []
class Request:
    entries: dict[str, int]
""",
        path="src/nof1_causal_lab/artifacts/model_spec.py",
        rules=frozenset({"CORE002"}),
    )
    assert [(v.code, v.target) for v in violations] == [
        ("CORE002", "field:entries"),
        ("CORE002", "return"),
    ]


@pytest.mark.parametrize(
    "operation",
    [
        "raise ValueError('invalid')",
        "assert model.indicators",
        "model.require_priors()",
        "validate_execution(model)",
        "validate_parameter_anchors(model)",
        "Result.model_validate(payload)",
        "Result.model_validate_json(payload)",
        "Result.model_validate_strings(payload)",
    ],
)
def test_pure_projection_scope_rejects_revalidation(operation: str) -> None:
    checker = _load_checker()
    source = f"def newly_added_projection(model: ModelSpec, payload):\n    {operation}\n"
    assert [
        v.code
        for v in checker.scan_text(
            source, path="src/nof1_causal_lab/study/equations.py", rules=frozenset({"VIEW001"})
        )
    ] == ["VIEW001"]
    assert checker.scan_text(source, path="src/boundary.py", rules=frozenset({"VIEW001"})) == []


def test_projection_allows_exhaustiveness() -> None:
    checker = _load_checker()
    assert (
        checker.scan_text(
            "def project(value):\n    assert_never(value)\n",
            path="src/nof1_causal_lab/study/equations.py",
            rules=frozenset({"VIEW001"}),
        )
        == []
    )
