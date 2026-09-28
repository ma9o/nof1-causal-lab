"""Parse cache metadata before using it for restoration or waiting on warmup."""

import json

import pytest
from pydantic import ValidationError

from nof1_causal_lab.flows import model_spec_compile_cache as cache
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "state",
    [{"status": "ready"}, {"status": "pending", "function_call_id": "fc-123"}],
)
def test_loads_typed_cache_metadata(tmp_path, monkeypatch, state):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    payload = {"schema_version": 1, "topology_fingerprint": "topology", **state}
    storage.write_text(cache._metadata_path("workspace"), json.dumps(payload))

    metadata = cache.load_model_spec_compile_cache_metadata("workspace")

    assert metadata is not None
    assert metadata.status == state["status"]
    assert cache._metadata_matches(metadata, "topology")
    assert not cache._metadata_matches(metadata, "different")
    if metadata.status == "pending":
        assert metadata.function_call_id == "fc-123"


@pytest.mark.parametrize("function_call_id", [None, "", 42])
def test_pending_metadata_requires_a_call_id(tmp_path, monkeypatch, function_call_id):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    payload = {
        "schema_version": 1,
        "topology_fingerprint": "topology",
        "status": "pending",
        "function_call_id": function_call_id,
    }
    storage.write_text(cache._metadata_path("workspace"), json.dumps(payload))

    with pytest.raises(ValidationError, match="function_call_id"):
        cache.load_model_spec_compile_cache_metadata("workspace")
