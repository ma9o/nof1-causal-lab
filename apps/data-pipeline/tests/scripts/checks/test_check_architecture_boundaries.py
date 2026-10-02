"""Tests for the executable architecture-boundary checker."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract


@lru_cache(maxsize=1)
def _load_checker() -> Any:
    module_name = "check_architecture_boundaries_under_test"
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "checks"
        / "check_architecture_boundaries.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_module(source_root: Path, relative_path: str, source: str) -> None:
    path = source_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent
    while directory.is_relative_to(source_root):
        (directory / "__init__.py").touch()
        directory = directory.parent
    path.write_text(source, encoding="utf-8")


def test_new_modules_inherit_roles_and_unclassified_roots_fail(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(source_root, "artifacts/new_contract.py", "")
    inventory = checker.role_inventory(source_root)
    assert inventory[source_root / "artifacts/new_contract.py"] == "domain"
    assert set(inventory.values()) == {"domain"}
    _write_module(source_root, "unclassified.py", "")
    with pytest.raises(ValueError, match="Unclassified production module"):
        checker.role_inventory(source_root)


def test_type_only_dependencies_do_not_initialize_forbidden_layers(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "artifacts/new_value.py",
        """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.utils.storage import read_text
""",
    )
    _write_module(
        source_root,
        "models/new_compiler.py",
        """
import typing

if typing.TYPE_CHECKING:
    import time
""",
    )
    _write_module(source_root, "utils/storage.py", "")

    assert checker.find_violations(source_root) == ()


@pytest.mark.parametrize(
    "annotation",
    [
        "FitSettingsSpec",
        "dict[str, object]",
        "Options",
        "SamplerSpec | None",
        "ModelSpec",
        "ObservationLawSpec",
        "LikelihoodSpec",
        "pl.DataFrame",
        "SettingsAlias",
    ],
)
def test_execution_rejects_partial_sampler_inputs(tmp_path: Path, annotation: str) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/ssm/inference/new_sampler.py",
        f"""
from typing import TypedDict
import polars as pl
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.likelihood import ObservationLawSpec, LikelihoodSpec
from nof1_causal_lab.sampler_config import SamplerSpec
type SettingsAlias = FitSettingsSpec
class Options(TypedDict, total=False):
    num_samples: int

def run(*, request_values: {annotation}) -> None:
    pass
""",
    )
    assert [violation.code for violation in checker.find_violations(source_root)] == ["ARCH007"]


def test_execution_accepts_resolved_sampler_alias(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/ssm/inference/new_sampler.py",
        """
from nof1_causal_lab.sampler_config import SamplerSpec as Resolved

def run(*, sampler: Resolved) -> None:
    pass
""",
    )
    assert checker.find_violations(source_root) == ()


def test_pure_roles_reject_transitive_function_local_acquisition(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    producers = (
        "artifacts/new_value.py",
        "models/new_compiler.py",
        "models/ssm/new_engine.py",
        "workers/prompts/new_projection.py",
    )
    for relative in producers:
        _write_module(source_root, relative, "from nof1_causal_lab.utils.shared import read\n")
    _write_module(
        source_root,
        "utils/shared.py",
        "def read():\n    from nof1_causal_lab.utils.storage import read_text\n",
    )
    _write_module(source_root, "utils/storage.py", "import time\n")

    violations = checker.find_violations(source_root)

    for relative in producers:
        findings = [item for item in violations if item.ref.path == source_root / relative]
        assert {item.ref.imported for item in findings} == {"nof1_causal_lab.utils.storage", "time"}
        assert all(item.code == "ARCH008" for item in findings)
        assert all("nof1_causal_lab.utils.shared ->" in item.message for item in findings)
    assert all(
        "role=" in item.diagnostic(source_root) and "fix at" in item.diagnostic(source_root)
        for item in violations
    )
