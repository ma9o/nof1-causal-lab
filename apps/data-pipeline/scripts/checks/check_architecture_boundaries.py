"""Enforce the role-derived dependency rules of the architecture.

Static package seams are import-linter contracts in ``pyproject.toml``. The rules
here follow the module roles owned by ``architecture_roles``:

``ARCH007``
    Numerical execution signatures accept compiled models, bound observations
    and the owned resolved SamplerSpec, never authoring or transport inputs,
    partial configurations or forwarding TypedDicts.

``ARCH008``
    Pure roles never reach edge or shell modules, I/O, clock or environment
    acquisition, directly or transitively.

Imports guarded by ``TYPE_CHECKING`` are excluded because these rules constrain
runtime ownership and initialization, not type annotation dependencies.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import grimp

from scripts.checks.architecture_roles import PACKAGE, fix_owner, role_for_module, role_inventory

_PACKAGE = PACKAGE


@dataclass(frozen=True, slots=True)
class ImportRef:
    """One runtime import between project modules."""

    path: Path
    line: int
    importer: str
    imported: str


@dataclass(frozen=True, slots=True)
class Violation:
    """One forbidden dependency."""

    ref: ImportRef
    code: str
    message: str

    def diagnostic(self, source_root: Path) -> str:
        path = self.ref.path.relative_to(source_root.parent).as_posix()
        role = role_for_module(self.ref.importer)
        owner_path = f"src/{self.ref.importer.replace('.', '/')}.py"
        return (
            f"{path}:{self.ref.line}: {self.code} {self.message} "
            f"[role={role}; fix at {fix_owner(owner_path)}]"
        )


def find_violations(source_root: Path) -> tuple[Violation, ...]:
    """Return every forbidden runtime dependency below ``source_root``."""
    violations: list[Violation] = []

    # Resolve signatures by type ownership, including aliases and renamed arguments.
    # Raw data/configuration is bound before execution; native numerical leaves remain valid.
    from scripts.checks.architecture_roles import role_for_path
    from scripts.checks.check_type_boundaries import TypeIndex

    inventory = role_inventory(source_root)
    sources = list(
        (
            "src/" + _PACKAGE + "/" + path.relative_to(source_root).as_posix(),
            ast.parse(path.read_text(encoding="utf-8")),
        )
        for path in inventory
    )
    index = TypeIndex(sources)
    raw = {
        f"{_PACKAGE}.artifacts.model_spec.ModelSpec",
        f"{_PACKAGE}.artifacts.likelihood.ObservationLawSpec",
        f"{_PACKAGE}.artifacts.likelihood.LikelihoodSpec",
        f"{_PACKAGE}.artifacts.posterior.FitSettingsSpec",
        "polars.DataFrame",
        "pandas.DataFrame",
        "pyarrow.Table",
    }

    def unresolved(node: ast.expr, module: str, seen: frozenset[str] = frozenset()) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                expression = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return False  # Jaxtyping dimension strings are metadata, not Python types.
            return unresolved(expression, module, seen)
        name = index.resolve(node, module)
        if isinstance(node, ast.Subscript) and name.rsplit(".", 1)[-1] == "Literal":
            return False
        if (
            isinstance(node, ast.Subscript)
            and name.rsplit(".", 1)[-1] == "Annotated"
            and isinstance(node.slice, ast.Tuple)
        ):
            return unresolved(node.slice.elts[0], module, seen)
        if name in raw:
            return True
        if name in seen:
            return False
        if name in index.aliases:
            return unresolved(index.aliases[name], name.rpartition(".")[0], seen | {name})
        if name in index.classes:
            owner, declaration = index.classes[name]
            if role_for_path(owner) in {"edge", "shell"}:
                return True
            if any(
                keyword.arg == "total"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in declaration.keywords
            ):
                return True
            return any(
                unresolved(base, name.rpartition(".")[0], seen | {name})
                for base in declaration.bases
            )
        if (
            isinstance(node, ast.Subscript)
            and name.rsplit(".", 1)[-1]
            in {
                "dict",
                "Dict",
                "Mapping",
                "MutableMapping",
            }
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
        ):
            value = node.slice.elts[1]
            if index.resolve(value, module).rsplit(".", 1)[-1] in {"Any", "object"}:
                return True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.BitOr)
            and any(
                isinstance(part, ast.Constant) and part.value is None for part in ast.walk(node)
            )
            and f"{_PACKAGE}.sampler_config.SamplerSpec" in index.return_variants(node, module)
        ):
            return True
        return any(
            unresolved(child, module, seen)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        )

    for (relative, tree), (path, role) in zip(sources, inventory.items(), strict=True):
        if role != "execution":
            continue
        importer = index.module(relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if argument.annotation is not None and unresolved(argument.annotation, importer):
                    violations.append(
                        Violation(
                            ImportRef(
                                path, argument.lineno, importer, ast.unparse(argument.annotation)
                            ),
                            "ARCH007",
                            "execution requires compiled models, bound observations and resolved sampler values; bind authoring/transport inputs at their compiler or shell owner",
                        )
                    )

    # grimp owns runtime import discovery, local imports and transitive traversal.
    # A caller-supplied source root is also used by isolated checker regressions.
    previous_package = sys.modules.pop(_PACKAGE, None)
    sys.path.insert(0, str(source_root.parent.resolve()))
    importlib.invalidate_caches()
    try:
        graph = grimp.build_graph(
            _PACKAGE,
            include_external_packages=True,
            exclude_type_checking_imports=True,
            cache_dir=None,
        )
    finally:
        sys.path.pop(0)
        if previous_package is not None:
            sys.modules[_PACKAGE] = previous_package

    modules = {
        _PACKAGE
        + (
            "."
            + path.relative_to(source_root)
            .with_suffix("")
            .as_posix()
            .removesuffix("/__init__")
            .replace("/", ".")
            if path != source_root / "__init__.py"
            else ""
        ): path
        for path in inventory
    }
    pure_roles = {"domain", "compiler", "execution", "projection"}
    acquisition = {
        "os",
        "time",
        "socket",
        "subprocess",
        "tempfile",
        "shutil",
        "fsspec",
        "httpx",
        "openai",
        "pygit2",
        "dotenv",
        "uvicorn",
        "fastapi",
        "temporalio",
        "mcp",
        "modal",
        "exa_py",
        "botocore",
        "zipfile",
    }
    for importer, path in sorted(modules.items()):
        if role_for_module(importer) not in pure_roles:
            continue
        for imported in sorted(graph.find_upstream_modules(importer)):
            if imported in acquisition or (
                imported in modules and role_for_module(imported) in {"edge", "shell"}
            ):
                chain = graph.find_shortest_chain(importer, imported)
                assert chain is not None
                details = graph.get_import_details(importer=chain[0], imported=chain[1])
                violations.append(
                    Violation(
                        ImportRef(path, details[0]["line_number"], importer, imported),
                        "ARCH008",
                        "pure roles cannot reach edge, shell, I/O, clock or environment "
                        f"acquisition: {' -> '.join(chain)}",
                    )
                )

    return tuple(violations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        nargs="?",
        type=Path,
        default=Path("src/nof1_causal_lab"),
    )
    args = parser.parse_args()

    role_inventory(args.source_root)
    violations = find_violations(args.source_root)
    for violation in violations:
        print(violation.diagnostic(args.source_root), file=sys.stderr)
    if violations:
        print(f"{len(violations)} architecture boundary violation(s)", file=sys.stderr)
        return 1

    print("Architecture boundaries: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
