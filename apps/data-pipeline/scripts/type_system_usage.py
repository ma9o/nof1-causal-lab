"""Collect deliberately bounded source evidence for the exported type analysis.

Only direct accesses on annotated parameters and instance-method ``self`` are
attributed to a type. This is an observation index, not a Python type checker:
aliases, container elements, dynamic access, and other languages remain unknown.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx


@dataclass(frozen=True)
class FieldUse:
    location: str
    fields: frozenset[str]
    internal: bool
    behavior: bool


@dataclass
class SourceEvidence:
    files: int = 0
    declarations: dict[str, str] = field(default_factory=dict)
    methods: dict[str, tuple[str, ...]] = field(default_factory=dict)
    uses: dict[str, list[FieldUse]] = field(default_factory=lambda: defaultdict(list))


def _body_nodes(function: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Keep nested functions, classes, and lambdas in their own lexical scope."""
    result: list[ast.AST] = []
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        result.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return result


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_qualified_name(node.value)}.{node.attr}"
    return ""


def _annotation_types(node: ast.AST | None, names: dict[str, str]) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _annotation_types(ast.parse(node.value, mode="eval").body, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_types(node.left, names) | _annotation_types(node.right, names)
    if isinstance(node, ast.Subscript):
        arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        wrapper = _qualified_name(node.value).rsplit(".", 1)[-1]
        if wrapper == "Annotated":
            return _annotation_types(arguments[0], names)
        if wrapper in {"Optional", "Union"}:
            return set().union(*(_annotation_types(arg, names) for arg in arguments))
        # A list[T] parameter is a list, not an instance of T.
        return set()
    name = _qualified_name(node) if node is not None else ""
    return {names[name]} if name in names else set()


def _module_names(
    tree: ast.Module, module: str, package: str, exports: dict[str, str]
) -> dict[str, str]:
    """Resolve explicit import aliases against the schema's declared Python owner."""
    names = {
        qualified.removeprefix(module + "."): name
        for qualified, name in exports.items()
        if qualified.rsplit(".", 1)[0] == module
    }
    # Imports under TYPE_CHECKING are useful evidence as well.
    for node in _body_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: 1 - node.level]
                prefix = ".".join(parts + ([prefix] if prefix else []))
            for alias in node.names:
                qualified = f"{prefix}.{alias.name}"
                if qualified in exports:
                    names[alias.asname or alias.name] = exports[qualified]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for qualified, name in exports.items():
                    if qualified.startswith(alias.name + "."):
                        local = (
                            (alias.asname + qualified[len(alias.name) :])
                            if alias.asname
                            else qualified
                        )
                        names[local] = name
    return names


def _record_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    owner: str | None,
    names: dict[str, str],
    graph: nx.DiGraph,
    location: str,
    evidence: SourceEvidence,
) -> None:
    arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    receivers = {arg.arg: _annotation_types(arg.annotation, names) for arg in arguments}
    if owner is not None and arguments and arguments[0].arg == "self":
        receivers["self"] = {owner}
    nodes = _body_nodes(function)
    # Rebinding an annotated parameter invalidates this simple attribution rule.
    rebound = {
        node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    for name in rebound:
        receivers.pop(name, None)
    accessed: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            for name in receivers.get(node.value.id, ()):
                if node.attr in graph.nodes[name]["schema"].get("properties", {}):
                    accessed[name].add(node.attr)
    decorators = [ast.unparse(decorator) for decorator in function.decorator_list]
    behavior = any("validator" in item or "serializer" in item for item in decorators) or any(
        word in function.name
        for word in ("resolve", "normalize", "coerce", "parse", "serialize", "convert")
    )
    # Before-validators often read raw dictionaries, rather than self.field.
    if owner is not None:
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Call) and _qualified_name(decorator.func).rsplit(".", 1)[
                -1
            ] in {"field_validator", "field_serializer"}:
                accessed[owner].update(
                    arg.value
                    for arg in decorator.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value in graph.nodes[owner]["schema"].get("properties", {})
                )
    for name, fields in accessed.items():
        if fields:
            evidence.uses[name].append(
                FieldUse(location, frozenset(fields), owner == name, behavior)
            )


def collect_source_evidence(graph: nx.DiGraph, source_root: Path) -> SourceEvidence:
    """Index source declarations and observed field use without importing models."""
    exports = {
        f"{data['schema']['x-python-module']}.{name}": name
        for name, data in graph.nodes(data=True)
        if "x-python-module" in data["schema"] and "[" not in data["schema"].get("title", "")
    }
    evidence = SourceEvidence()
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        evidence.files += 1
        parts = list(path.relative_to(source_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        package = module if path.stem == "__init__" else module.rpartition(".")[0]
        names = _module_names(tree, module, package, exports)
        owners: dict[int, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            owner = exports.get(f"{module}.{node.name}")
            if owner is None:
                continue
            evidence.declarations[owner] = f"{path}:{node.lineno}"
            methods = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            evidence.methods[owner] = tuple(f"{path}:{item.lineno} {item.name}" for item in methods)
            owners.update({id(item): owner for item in methods})
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                owner = owners.get(id(node))
                label = f"{owner}.{node.name}" if owner is not None else node.name
                _record_function(
                    node,
                    owner=owner,
                    names=names,
                    graph=graph,
                    location=f"{path}:{node.lineno} {label}",
                    evidence=evidence,
                )
    return evidence
