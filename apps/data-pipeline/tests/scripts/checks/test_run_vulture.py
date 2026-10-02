"""Vulture reference generation and concurrent ownership passes."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.checks.run_vulture import _scan_string_type_node, _write_phantom

from scripts.checks import run_vulture

pytestmark = pytest.mark.contract


def test_keyword_field_alias_does_not_break_string_type_references(tmp_path):
    annotation = ast.parse('Annotated["ArtifactId", Field(alias="from")]', mode="eval").body
    refs: set[str] = set()
    _scan_string_type_node(annotation, refs)
    _write_phantom(tmp_path, "refs.py", refs)
    tree = ast.parse((tmp_path / "refs.py").read_text())
    assert {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} == {"_", "ArtifactId"}


@pytest.mark.parametrize("used_locally", [False, True])
def test_concurrent_passes_preserve_ownership_and_clean_caches(tmp_path, used_locally):
    script = tmp_path / "scripts/checks/run_vulture.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(run_vulture.__file__), script)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.vulture]\npaths = ["src"]\nmin_confidence = 60\n'
    )
    files = {
        "src/api.py": ("production_helper", ""),
        "evaluation/check.py": ("evaluation_helper", ""),
        "scripts/usage.py": (
            "scripts_helper",
            "from evaluation.check import evaluation_helper\nevaluation_helper()\n",
        ),
        "notebooks/demo.py": (
            "notebook_helper",
            "from scripts.usage import scripts_helper\nscripts_helper()\n",
        ),
        "tests/test_usage.py": (
            "fixture_helper",
            "from src.api import production_helper\n"
            "from notebooks.demo import notebook_helper\n"
            "production_helper()\nnotebook_helper()\n",
        ),
    }
    for relative, (helper, imports) in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        local_use = f"{helper}()\n" if used_locally else ""
        path.write_text(f"{imports}\ndef {helper}():\n    return 1\n\n{local_use}")

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == (0 if used_locally else 3), completed.stdout + completed.stderr
    assert not completed.stderr
    findings = [line for line in completed.stdout.splitlines() if "unused function" in line]
    expected_helpers = [] if used_locally else [helper for helper, _ in files.values()]
    assert len(findings) == len(expected_helpers)
    for finding, helper in zip(findings, expected_helpers, strict=True):
        assert f"unused function '{helper}'" in finding
    assert not list((tmp_path / ".vulture_cache").iterdir())
