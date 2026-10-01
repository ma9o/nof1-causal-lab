"""Shared fixtures for SSM contracts, inference, and predictive test suites."""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from tests.model_fixtures import default_static_state_sd_block, dense_matrix_dynamics_spec

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION FAMILY MATRIX
# ══════════════════════════════════════════════════════════════════════════════


def complex_mixed_family_config() -> tuple[list[str], list[str], list[int], list[str]]:
    manifest_dists = [
        "gaussian",
        "bernoulli",
        "poisson",
        "student_t",
        "gamma",
        "beta",
        "ordered_logistic",
        "categorical",
        "negative_binomial",
        "gaussian",
    ]
    manifest_links = [
        LinkFunction.IDENTITY.value,
        LinkFunction.LOGIT.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
        LinkFunction.LOG.value,
        LinkFunction.LOGIT.value,
        LinkFunction.CUMULATIVE_LOGIT.value,
        LinkFunction.SOFTMAX.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
    ]
    manifest_level_counts = [0, 0, 0, 0, 0, 0, 4, 4, 0, 0]
    manifest_names = [
        "stress_cont",
        "adherence_flag",
        "steps_count",
        "fatigue_t",
        "screen_gap",
        "sleep_efficiency",
        "symptom_severity",
        "coping_style",
        "rumination_count",
        "focus_cont",
    ]
    return manifest_dists, manifest_links, manifest_level_counts, manifest_names


# ══════════════════════════════════════════════════════════════════════════════
# PRIOR-PREDICTIVE RUNTIME SPEC
# ══════════════════════════════════════════════════════════════════════════════


def complex_mixed_runtime_spec() -> ModelSpec:
    return ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / '_support/complex_mixed_runtime_spec_model_fixture.json').read_text())
