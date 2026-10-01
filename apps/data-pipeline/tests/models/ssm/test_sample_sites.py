"""Fixed model parameters do not declare sample sites."""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from tests.dynamics_fixtures import decay_term
from tests.model_fixtures import default_static_state_sd_block

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec




def test_all_fixed_spec_yields_no_sites():
    spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'sample_sites/all_fixed_spec_yields_no_sites__all_fixed_spec.json').read_text())
    assert list(numeric.iter_sample_sites(spec)) == []
