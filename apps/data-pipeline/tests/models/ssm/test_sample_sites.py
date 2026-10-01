"""Fixed model parameters do not declare sample sites."""

from __future__ import annotations
from nof1_causal_lab.artifacts.model_spec import ModelSpec

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nof1_causal_lab.models.ssm import numerics as numeric

pytestmark = pytest.mark.contract




def test_all_fixed_spec_yields_no_sites():
    spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'sample_sites/all_fixed_spec_yields_no_sites__all_fixed_spec.json').read_text())
    assert list(numeric.iter_sample_sites(spec)) == []
