"""Compiled profiling artifacts preserve JAX's native text and numeric outputs."""

import json

import jax
import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm.inference._profiling import dump_compiled_analysis

pytestmark = pytest.mark.contract


def test_compiled_analysis_writes_readable_hlo_and_numeric_costs(tmp_path):
    @jax.jit
    def affine(values):
        return values * 2.0 + 1.0

    dump_compiled_analysis(
        affine,
        jnp.arange(8, dtype=jnp.float32),
        profile_dir=tmp_path,
        label="affine",
    )

    assert "HloModule" in (tmp_path / "affine.hlo.txt").read_text()
    costs = json.loads((tmp_path / "affine.cost.json").read_text())
    assert costs["flops"] >= 8
    assert costs["bytes accessed"] > 0
    operations = json.loads((tmp_path / "affine.access_patterns.json").read_text())
    assert any(count > 0 for count in operations.values())
