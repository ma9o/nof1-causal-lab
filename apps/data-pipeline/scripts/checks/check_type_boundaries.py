"""Reject type annotations that erase domain types behind ``dict`` or ``Any``.

The checker owns project-specific rules that Ruff and ty cannot express:

``CUSTOM001``
    ``Any`` must not be a member of a union. Such a union is equivalent to
    ``Any`` and therefore provides no narrowing.

``CUSTOM002``
    A named domain type must not be unioned with ``dict``. Raw mappings are
    parsed at I/O boundaries; internal code receives the validated type.

``CUSTOM003``
    A parameter must not include ``None`` in its type only to reject ``None``
    on every normally completing path. Make the parameter required and let the
    type checker push absence handling to the boundary where it originates.

``CUSTOM004``
    An annotation must not spell an anonymous dictionary containing ``Any``.
    Use the explicitly unsafe ``UncheckedJsonObject`` boundary type or define a
    domain-specific named mapping alias so unchecked values cannot spread
    invisibly. ``JsonObject`` is reserved for recursively JSON-safe values.

``CUSTOM005``
    ``UncheckedJsonObject`` is confined to parser parameters and local inputs.
    Every read of such an input must feed model/TypeAdapter validation in that
    function. It cannot be stored on objects, aliased, or returned unchecked.

``CORE001``
    Construction bypasses and mutable casts are checked by call name. Only
    self-revision and compiler-owned evidence construction are permitted.

``CORE002``
    The explicitly listed scientific owners expose read-only collections,
    including derived properties.

``VIEW001``
    Declared pure projections cannot raise, assert, or revalidate the core.
    Exhaustive ``assert_never`` arms are permitted.

"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import override

_ANY_UNION = "CUSTOM001"
_DOMAIN_DICT_UNION = "CUSTOM002"
_REJECT_ONLY_OPTIONAL_PARAMETER = "CUSTOM003"
_ANONYMOUS_ANY_DICT = "CUSTOM004"
_UNCHECKED_JSON_ESCAPE = "CUSTOM005"
_CORE_BYPASS = "CORE001"
_MUTABLE_CORE = "CORE002"
_PARTIAL_VIEW = "VIEW001"
_ALL_RULES = frozenset(
    {
        _ANY_UNION,
        _DOMAIN_DICT_UNION,
        _REJECT_ONLY_OPTIONAL_PARAMETER,
        _ANONYMOUS_ANY_DICT,
        _UNCHECKED_JSON_ESCAPE,
        _CORE_BYPASS,
        _MUTABLE_CORE,
        _PARTIAL_VIEW,
    }
)
_DOMAIN_TYPE_SUFFIXES = (
    "Artifact",
    "Contract",
    "Design",
    "Model",
    "Plan",
    "Proposal",
    "Report",
    "Result",
    "Spec",
    "Structure",
)
_MODEL_BASE_NAMES = frozenset({"BaseModel", "NamedTuple", "Protocol", "TypedDict"})


@dataclass(frozen=True)
class _TypingBindings:
    """Names through which one module refers to special typing forms."""

    any_names: frozenset[str]
    optional_names: frozenset[str]
    type_alias_names: frozenset[str]
    union_names: frozenset[str]
    module_names: frozenset[str]


@dataclass(frozen=True)
class Violation:
    """One source-level type-boundary violation."""

    path: str
    line: int
    column: int
    code: str
    scope: str
    target: str
    annotation: str
    message: str

    def diagnostic(self) -> str:
        """Render in the concise format understood by editors and CI."""
        return (
            f"{self.path}:{self.line}:{self.column}: "
            f"{self.code} {self.message} [{self.scope} {self.target}]"
        )


def _discover_typing_bindings(tree: ast.Module) -> _TypingBindings:
    """Resolve common direct and aliased imports from typing modules."""
    names = {
        "Any": {"Any"},
        "Optional": {"Optional"},
        "TypeAlias": {"TypeAlias"},
        "Union": {"Union"},
    }
    module_names = {"typing", "typing_extensions"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in {"typing", "typing_extensions"}:
                    module_names.add(imported.asname or imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "typing",
            "typing_extensions",
        }:
            for imported in node.names:
                if imported.name in names:
                    names[imported.name].add(imported.asname or imported.name)
    return _TypingBindings(
        any_names=frozenset(names["Any"]),
        optional_names=frozenset(names["Optional"]),
        type_alias_names=frozenset(names["TypeAlias"]),
        union_names=frozenset(names["Union"]),
        module_names=frozenset(module_names),
    )


def _is_typing_form(
    node: ast.expr,
    *,
    direct_names: frozenset[str],
    attribute_name: str,
    bindings: _TypingBindings,
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in direct_names
    return bool(
        isinstance(node, ast.Attribute)
        and node.attr == attribute_name
        and isinstance(node.value, ast.Name)
        and node.value.id in bindings.module_names
    )


def _direct_union_members(
    node: ast.expr,
    bindings: _TypingBindings,
) -> list[ast.expr] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return [node.left, node.right]
    if not isinstance(node, ast.Subscript):
        return None
    if _is_typing_form(
        node.value,
        direct_names=bindings.union_names,
        attribute_name="Union",
        bindings=bindings,
    ):
        if isinstance(node.slice, ast.Tuple):
            return list(node.slice.elts)
        return [node.slice]
    if _is_typing_form(
        node.value,
        direct_names=bindings.optional_names,
        attribute_name="Optional",
        bindings=bindings,
    ):
        return [node.slice, ast.Constant(value=None)]
    return None


def _flatten_union(
    node: ast.expr,
    bindings: _TypingBindings,
) -> list[ast.expr]:
    direct_members = _direct_union_members(node, bindings)
    if direct_members is None:
        return [node]
    return [
        flattened for member in direct_members for flattened in _flatten_union(member, bindings)
    ]


def _is_none_member(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_optional_annotation(
    node: ast.expr,
    bindings: _TypingBindings,
) -> bool:
    direct_members = _direct_union_members(node, bindings)
    return bool(
        direct_members and any(_is_none_member(member) for member in _flatten_union(node, bindings))
    )


def _directly_rejected_none_parameter(statement: ast.stmt) -> str | None:
    """Return the parameter rejected by an exact ``if x is None: raise`` guard."""
    if (
        not isinstance(statement, ast.If)
        or statement.orelse
        or len(statement.body) != 1
        or not isinstance(statement.body[0], ast.Raise)
    ):
        return None

    test = statement.test
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Is)
        or len(test.comparators) != 1
    ):
        return None

    left, right = test.left, test.comparators[0]
    if isinstance(left, ast.Name) and _is_none_member(right):
        return left.id
    if _is_none_member(left) and isinstance(right, ast.Name):
        return right.id
    return None


def _body_without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _executed_nodes(node: ast.AST):
    """Yield nodes executed by this scope without descending into nested scopes."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    for child in ast.iter_child_nodes(node):
        yield from _executed_nodes(child)


def _prefix_can_accept_none(
    statements: list[ast.stmt],
    parameter: str,
) -> bool:
    """Whether a prefix may finish normally or replace the incoming parameter."""
    for statement in statements:
        for node in _executed_nodes(statement):
            if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                return True
            if (
                isinstance(node, ast.Name)
                and node.id == parameter
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                return True
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == parameter
            ):
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split(".", maxsplit=1)[0]) == parameter
                for alias in node.names
            ):
                return True
            if isinstance(node, ast.ExceptHandler) and node.name == parameter:
                return True
    return False


def _dominating_none_rejections(body: list[ast.stmt]) -> frozenset[str]:
    """Find top-level None-rejection guards reached before normal completion."""
    statements = _body_without_docstring(body)
    rejected: set[str] = set()
    for index, statement in enumerate(statements):
        parameter = _directly_rejected_none_parameter(statement)
        if parameter is not None and not _prefix_can_accept_none(statements[:index], parameter):
            rejected.add(parameter)
    return frozenset(rejected)


def _union_nodes(node: ast.AST, bindings: _TypingBindings):
    """Yield maximal PEP 604, Union, and Optional expressions."""
    if isinstance(node, ast.expr) and _direct_union_members(node, bindings) is not None:
        yield node
        for member in _flatten_union(node, bindings):
            for child in ast.iter_child_nodes(member):
                yield from _union_nodes(child, bindings)
        return
    for child in ast.iter_child_nodes(node):
        yield from _union_nodes(child, bindings)


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _generic_base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Subscript):
        return _terminal_name(node.value)
    return _terminal_name(node)


def _is_any_member(
    node: ast.expr,
    bindings: _TypingBindings,
    any_aliases: frozenset[str],
) -> bool:
    if isinstance(node, ast.Name) and node.id in any_aliases:
        return True
    return _is_typing_form(
        node,
        direct_names=bindings.any_names,
        attribute_name="Any",
        bindings=bindings,
    )


def _is_dict_member(node: ast.expr) -> bool:
    return _generic_base_name(node) in {"dict", "Dict"}


def _anonymous_any_dict_nodes(
    node: ast.AST,
    bindings: _TypingBindings,
    any_aliases: frozenset[str],
):
    """Yield explicit ``dict[..., Any]`` expressions, including nested ones."""
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Subscript) or not _is_dict_member(candidate):
            continue
        arguments = (
            list(candidate.slice.elts)
            if isinstance(candidate.slice, ast.Tuple)
            else [candidate.slice]
        )
        if any(
            _is_any_member(descendant, bindings, any_aliases)
            for argument in arguments
            for descendant in ast.walk(argument)
            if isinstance(descendant, ast.expr)
        ):
            yield candidate


def _discover_unchecked_json_names(tree: ast.Module) -> frozenset[str]:
    """Names through which one module imports ``UncheckedJsonObject``."""
    names = {"UncheckedJsonObject"}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported in node.names:
            if imported.name == "UncheckedJsonObject":
                names.add(imported.asname or imported.name)
    return frozenset(names)


def _unchecked_json_nodes(node: ast.AST, names: frozenset[str]):
    """Yield direct references to the shared unchecked JSON escape hatch."""
    for candidate in ast.walk(node):
        if (isinstance(candidate, ast.Name) and candidate.id in names) or (
            isinstance(candidate, ast.Attribute) and candidate.attr == "UncheckedJsonObject"
        ):
            yield candidate


def _is_named_domain_member(
    node: ast.expr,
    domain_type_names: frozenset[str],
) -> bool:
    name = _generic_base_name(node)
    return bool(name and (name in domain_type_names or name.endswith(_DOMAIN_TYPE_SUFFIXES)))


def _is_any_equivalent(
    node: ast.expr,
    bindings: _TypingBindings,
    any_aliases: frozenset[str],
) -> bool:
    """Whether a type alias expression is equivalent to Any."""
    if _is_any_member(node, bindings, any_aliases):
        return True
    direct_members = _direct_union_members(node, bindings)
    return bool(
        direct_members
        and any(_is_any_equivalent(member, bindings, any_aliases) for member in direct_members)
    )


def _discover_any_aliases(
    tree: ast.Module,
    bindings: _TypingBindings,
) -> frozenset[str]:
    """Find module-level aliases that resolve to Any, directly or through a union."""
    definitions: list[tuple[str, ast.expr]] = []
    for node in tree.body:
        if isinstance(node, ast.TypeAlias):
            definitions.append((node.name.id, node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and _is_typing_form(
                node.annotation,
                direct_names=bindings.type_alias_names,
                attribute_name="TypeAlias",
                bindings=bindings,
            )
        ):
            definitions.append((node.target.id, node.value))
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            # Implicit aliases such as ``Loose = Any`` remain valid typing syntax.
            definitions.append((node.targets[0].id, node.value))

    aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        known_aliases = frozenset(aliases)
        for name, value in definitions:
            if name not in aliases and _is_any_equivalent(value, bindings, known_aliases):
                aliases.add(name)
                changed = True
    return frozenset(aliases)


def _discover_domain_type_names(trees: list[ast.Module]) -> frozenset[str]:
    """Find local model-like classes without importing application modules."""
    bases_by_class: dict[str, set[str]] = {}
    domain_names: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases_by_class.setdefault(node.name, set()).update(
                name for base in node.bases if (name := _generic_base_name(base)) is not None
            )
            if any(
                _generic_base_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
                == "dataclass"
                for decorator in node.decorator_list
            ):
                domain_names.add(node.name)

    changed = True
    while changed:
        changed = False
        known_bases = _MODEL_BASE_NAMES | domain_names
        for class_name, base_names in bases_by_class.items():
            if class_name not in domain_names and base_names & known_bases:
                domain_names.add(class_name)
                changed = True
    return frozenset(domain_names)


def _only_read_by_validator(function: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Recognize consumption by validation, independent of file or function names."""
    nodes = [part for statement in function.body for part in _executed_nodes(statement)]
    reads = {
        id(part)
        for part in nodes
        if isinstance(part, ast.Name) and part.id == name and isinstance(part.ctx, ast.Load)
    }
    validated = {
        id(part)
        for call in nodes
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr
        in {"model_validate", "model_validate_json", "validate_python", "validate_json"}
        for part in (
            *call.args[:1],
            *(
                keyword.value
                for keyword in call.keywords
                if keyword.arg in {"obj", "object", "json_data", "data"}
            ),
        )
        if isinstance(part, ast.Name) and part.id == name
    }
    return bool(reads) and reads <= validated


class _AnnotationVisitor(ast.NodeVisitor):
    def __init__(
        self,
        path: str,
        domain_type_names: frozenset[str],
        typing_bindings: _TypingBindings,
        any_aliases: frozenset[str],
        unchecked_json_names: frozenset[str],
        rules: frozenset[str],
    ) -> None:
        self.path = path
        self.domain_type_names = domain_type_names
        self.typing_bindings = typing_bindings
        self.any_aliases = any_aliases
        self.unchecked_json_names = unchecked_json_names
        self.rules = rules
        self.scope: list[str] = []
        self.violations: list[Violation] = []
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    @override
    def visit(self, node: ast.AST):
        # These Python 3.12 nodes are dispatched explicitly so static dead-code
        # analysis can see the references that NodeVisitor otherwise resolves by name.
        if isinstance(node, ast.AnnAssign):
            self.visit_AnnAssign(node)
            return None
        if isinstance(node, ast.TypeAlias):
            self.visit_TypeAlias(node)
            return None
        return super().visit(node)

    def _scope_name(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _check_annotation(
        self,
        annotation: ast.expr,
        *,
        target: str,
        check_domain_dict: bool = True,
    ) -> None:
        if _UNCHECKED_JSON_ESCAPE in self.rules:
            for raw in _unchecked_json_nodes(annotation, self.unchecked_json_names):
                name = target.split(":", 1)[-1]
                if (
                    target.startswith(("parameter:", "variable:"))
                    and self.functions
                    and _only_read_by_validator(self.functions[-1], name)
                ):
                    continue
                self.violations.append(
                    Violation(
                        path=self.path,
                        line=raw.lineno,
                        column=raw.col_offset + 1,
                        code=_UNCHECKED_JSON_ESCAPE,
                        scope=self._scope_name(),
                        target=target,
                        annotation=ast.unparse(raw),
                        message="Unchecked JSON belongs only to parser inputs consumed by validation; use a validated type here",
                    )
                )
        check_anonymous_dict = not target.startswith("alias:") or target == "alias:JsonObject"
        if _ANONYMOUS_ANY_DICT in self.rules and check_anonymous_dict:
            for anonymous_dict in _anonymous_any_dict_nodes(
                annotation,
                self.typing_bindings,
                self.any_aliases,
            ):
                annotation_text = ast.unparse(anonymous_dict)
                self.violations.append(
                    Violation(
                        path=self.path,
                        line=anonymous_dict.lineno,
                        column=anonymous_dict.col_offset + 1,
                        code=_ANONYMOUS_ANY_DICT,
                        scope=self._scope_name(),
                        target=target,
                        annotation=annotation_text,
                        message=(
                            f"`{annotation_text}` is an anonymous unchecked dictionary; "
                            "use `UncheckedJsonObject` or a domain-specific named mapping alias"
                        ),
                    )
                )
        for union in _union_nodes(annotation, self.typing_bindings):
            members = _flatten_union(union, self.typing_bindings)
            annotation_text = ast.unparse(union)
            if _ANY_UNION in self.rules and any(
                _is_any_member(member, self.typing_bindings, self.any_aliases) for member in members
            ):
                self.violations.append(
                    Violation(
                        path=self.path,
                        line=union.lineno,
                        column=union.col_offset + 1,
                        code=_ANY_UNION,
                        scope=self._scope_name(),
                        target=target,
                        annotation=annotation_text,
                        message=(
                            f"`{annotation_text}` contains `Any`, so the entire union "
                            "collapses to `Any`"
                        ),
                    )
                )
            if (
                _DOMAIN_DICT_UNION in self.rules
                and check_domain_dict
                and any(_is_dict_member(member) for member in members)
                and any(
                    _is_named_domain_member(member, self.domain_type_names) for member in members
                )
            ):
                self.violations.append(
                    Violation(
                        path=self.path,
                        line=union.lineno,
                        column=union.col_offset + 1,
                        code=_DOMAIN_DICT_UNION,
                        scope=self._scope_name(),
                        target=target,
                        annotation=annotation_text,
                        message=(
                            f"`{annotation_text}` mixes a named domain type with `dict`; "
                            "parse the mapping at the I/O boundary"
                        ),
                    )
                )

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.functions.append(node)
        args = node.args
        parameters = (*args.posonlyargs, *args.args, *args.kwonlyargs)
        for arg in parameters:
            if arg.annotation is not None:
                self._check_annotation(arg.annotation, target=f"parameter:{arg.arg}")
        if args.vararg is not None and args.vararg.annotation is not None:
            self._check_annotation(
                args.vararg.annotation,
                target=f"parameter:*{args.vararg.arg}",
            )
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self._check_annotation(
                args.kwarg.annotation,
                target=f"parameter:**{args.kwarg.arg}",
            )
        if node.returns is not None:
            self._check_annotation(node.returns, target="return")

        if _REJECT_ONLY_OPTIONAL_PARAMETER in self.rules:
            rejected_parameters = _dominating_none_rejections(node.body)
            for rejected_arg in parameters:
                annotation = rejected_arg.annotation
                if (
                    rejected_arg.arg not in rejected_parameters
                    or annotation is None
                    or not _is_optional_annotation(annotation, self.typing_bindings)
                ):
                    continue
                annotation_text = ast.unparse(annotation)
                self.violations.append(
                    Violation(
                        path=self.path,
                        line=annotation.lineno,
                        column=annotation.col_offset + 1,
                        code=_REJECT_ONLY_OPTIONAL_PARAMETER,
                        scope=self._scope_name(),
                        target=f"parameter:{rejected_arg.arg}",
                        annotation=annotation_text,
                        message=(
                            f"`{rejected_arg.arg}: {annotation_text}` admits `None` only to "
                            "reject it before normal completion; remove `None` and handle "
                            "absence at the upstream boundary"
                        ),
                    )
                )

        for statement in node.body:
            self.visit(statement)
        self.functions.pop()
        self.scope.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and node.value is not None
            and _is_typing_form(
                node.annotation,
                direct_names=self.typing_bindings.type_alias_names,
                attribute_name="TypeAlias",
                bindings=self.typing_bindings,
            )
        ):
            self._check_annotation(
                node.value,
                target=f"alias:{node.target.id}",
                check_domain_dict=False,
            )
            return
        self._check_annotation(
            node.annotation,
            target=f"variable:{ast.unparse(node.target)}",
        )
        if node.value is not None:
            self.visit(node.value)

    @override
    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        # Recursive JSON aliases legitimately contain both named aliases and dict.
        # CUSTOM001 remains meaningful in aliases, but CUSTOM002 applies to value
        # annotations at domain boundaries.
        self._check_annotation(
            node.value,
            target=f"alias:{ast.unparse(node.name)}",
            check_domain_dict=False,
        )


# Explicit scientific owners; transport DTOs and other Pydantic models are not core.
_CORE_OWNERS = {
    "ModelSpec": "artifacts/model_spec.py",
    "ConstructSpec": "artifacts/construct.py",
    "CausalEdgeSpec": "artifacts/construct.py",
    "IndicatorSpec": "artifacts/indicator.py",
    "ObservationSpec": "artifacts/observations.py",
    "LikelihoodSpec": "artifacts/likelihood.py",
    "ObservationLawSpec": "artifacts/likelihood.py",
    "ParameterSpec": "artifacts/parameter_spec.py",
    "DynamicsMechanismSpec": "artifacts/mechanism.py",
    "LiteralExpression": "artifacts/expressions.py",
    "StateExpression": "artifacts/expressions.py",
    "CoefficientExpression": "artifacts/expressions.py",
    "BinaryExpression": "artifacts/expressions.py",
    "CallExpression": "artifacts/expressions.py",
    "CompiledFitInputs": "models/ssm/compile/inputs.py",
    "IncompleteModel": "models/ssm/compile/inputs.py",
    "UnsupportedFit": "models/ssm/compile/inputs.py",
}
_PROJECTION_SCOPES = frozenset({"src/nof1_causal_lab/study/equations.py"})
_CORE_REVALIDATION = frozenset(
    {
        "require_priors",
        "require_measurements",
        "require_execution_structure",
        "check_execution",
        "validate_execution",
        "validate_execution_structure",
        "validate_parameter_anchors",
    }
)
_EVIDENCE_TYPES = frozenset({"CompiledFitInputs", "IncompleteModel", "UnsupportedFit"})
_MUTABLE_NAMES = frozenset(
    {
        "list",
        "dict",
        "set",
        "List",
        "Dict",
        "Set",
        "MutableMapping",
        "MutableSequence",
        "MutableSet",
    }
)


def _annotation_expr(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ast.parse(node.value, mode="eval").body
    return node


class _CoreVisitor(ast.NodeVisitor):
    """Syntactic call-name and core collection checks, without receiver inference."""

    def __init__(self, tree: ast.Module, path: str, rules: frozenset[str]) -> None:
        self.path = path
        self.rules = rules
        self.scope: list[str] = []
        self.class_name: str | None = None
        self.in_function = False
        self.names: dict[str, str] = {}
        self.violations: list[Violation] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    self.names[name.asname or name.name] = name.name

    def _name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return self.names.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _owned_here(self, name: str) -> bool:
        return self.path == "src/nof1_causal_lab/" + _CORE_OWNERS[name]

    def _mutable(self, node: ast.expr) -> bool:
        return any(
            self._name(part) in _MUTABLE_NAMES
            for part in ast.walk(_annotation_expr(node))
            if isinstance(part, ast.expr)
        )

    def _add(self, node: ast.expr | ast.stmt, code: str, target: str, message: str) -> None:
        if code in self.rules:
            self.violations.append(
                Violation(
                    path=self.path,
                    line=node.lineno,
                    column=node.col_offset + 1,
                    code=code,
                    scope=".".join(self.scope) or "<module>",
                    target=target,
                    annotation=ast.unparse(node),
                    message=message,
                )
            )

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous = self.class_name
        self.class_name = node.name
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        self.class_name = previous

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous_function = self.in_function
        self.scope.append(node.name)
        if (
            self.class_name in _CORE_OWNERS
            and not node.name.startswith("_")
            and node.returns is not None
            and any(
                self._name(decorator) in {"property", "cached_property"}
                for decorator in node.decorator_list
            )
            and self._mutable(node.returns)
        ):
            self._add(
                node.returns,
                _MUTABLE_CORE,
                "return",
                "Expose a read-only tuple, frozenset or Mapping from the core owner; keep mutable builders private",
            )
        self.in_function = True
        self.generic_visit(node)
        self.scope.pop()
        self.in_function = previous_function

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and self.class_name in _CORE_OWNERS
            and not self.in_function
            and not node.target.id.startswith("_")
            and self._mutable(node.annotation)
        ):
            self._add(
                node.annotation,
                _MUTABLE_CORE,
                "field:" + node.target.id,
                "Core fields expose tuple, frozenset or Mapping; own the data at construction",
            )
        self.generic_visit(node)

    def _view_check(self, node: ast.expr | ast.stmt, target: str) -> None:
        if self.path in _PROJECTION_SCOPES:
            self._add(
                node,
                _PARTIAL_VIEW,
                target,
                "Projections are total; fix the core type/constructor, or resolve external input at the boundary before projecting",
            )

    @override
    def visit_Raise(self, node: ast.Raise) -> None:
        self._view_check(node, "raise")
        self.generic_visit(node)

    @override
    def visit_Assert(self, node: ast.Assert) -> None:
        self._view_check(node, "assert")
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        name = self._name(node.func)
        if name in _CORE_REVALIDATION or name.startswith("model_validate"):
            self._view_check(node, name)
        bypass = (
            name == "model_construct"
            or (
                name == "model_copy"
                and any(keyword.arg in {"update", None} for keyword in node.keywords)
                and not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                )
            )
            or (name in _EVIDENCE_TYPES and not self._owned_here(name))
        )
        if (
            name == "__setattr__"
            and isinstance(node.func, ast.Attribute)
            and self._name(node.func.value) == "object"
            and node.args
        ):
            bypass = not (isinstance(node.args[0], ast.Name) and node.args[0].id == "self")
        if name == "cast" and len(node.args) == 2:
            target = _annotation_expr(node.args[0])
            bypass = self._mutable(target) or any(
                self._name(part) in _EVIDENCE_TYPES
                for part in ast.walk(target)
                if isinstance(part, ast.expr)
            )
        if bypass:
            self._add(
                node,
                _CORE_BYPASS,
                name,
                "Use validated construction/revision or the compiler owner, and preserve read-only interfaces",
            )
        self.generic_visit(node)


def scan_text(
    source: str,
    *,
    path: str,
    domain_type_names: frozenset[str] | None = None,
    rules: frozenset[str] | None = None,
) -> list[Violation]:
    """Return violations from one Python source string."""
    tree = ast.parse(source, filename=path)
    resolved_domain_types = (
        _discover_domain_type_names([tree]) if domain_type_names is None else domain_type_names
    )
    typing_bindings = _discover_typing_bindings(tree)
    visitor = _AnnotationVisitor(
        path,
        resolved_domain_types,
        typing_bindings,
        _discover_any_aliases(tree, typing_bindings),
        _discover_unchecked_json_names(tree),
        _ALL_RULES if rules is None else rules,
    )
    visitor.visit(tree)
    core_visitor = _CoreVisitor(tree, path, _ALL_RULES if rules is None else rules)
    core_visitor.visit(tree)
    return visitor.violations + core_visitor.violations


def scan_paths(
    paths: list[Path],
    *,
    repo_root: Path,
    rules: frozenset[str] | None = None,
) -> list[Violation]:
    """Scan Python files under the requested paths."""
    python_files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            python_files.add(path)
        elif path.is_dir():
            python_files.update(path.rglob("*.py"))

    parsed_sources: list[tuple[str, str, ast.Module]] = []
    for path in sorted(python_files):
        relative_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
        source = path.read_text(encoding="utf-8")
        parsed_sources.append(
            (
                relative_path,
                source,
                ast.parse(
                    source,
                    filename=relative_path,
                ),
            )
        )

    domain_type_names = _discover_domain_type_names(
        [tree for _path, _source, tree in parsed_sources]
    )
    violations: list[Violation] = []
    for relative_path, source, _tree in parsed_sources:
        violations.extend(
            scan_text(
                source,
                path=relative_path,
                domain_type_names=domain_type_names,
                rules=rules,
            )
        )
    return sorted(
        violations,
        key=lambda item: (item.path, item.line, item.column, item.code),
    )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            repo_root / "src",
            repo_root / "scripts",
            repo_root / "evaluation",
            repo_root / "notebooks",
        ],
        help="Python files or directories to scan (default: production Python trees)",
    )
    parser.add_argument(
        "--select",
        action="append",
        choices=sorted(_ALL_RULES),
        help="Run only this rule (repeatable; default: all rules)",
    )
    args = parser.parse_args(argv)

    paths = [path if path.is_absolute() else Path.cwd() / path for path in args.paths]
    selected_rules = frozenset(args.select or _ALL_RULES)
    violations = scan_paths(paths, repo_root=repo_root, rules=selected_rules)
    for violation in violations:
        print(violation.diagnostic(), file=sys.stderr)
    if violations:
        print(f"type-boundary check failed: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
