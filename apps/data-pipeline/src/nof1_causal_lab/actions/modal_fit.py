"""GPU compute for local fits; the caller still owns the study and publication."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.posterior_diagnostics import ParticleMCMCEvidence
from nof1_causal_lab.utils.arrays import decode_array, encode_array

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import numpy as np
    import polars as pl

    from nof1_causal_lab.actions.fit import FitResult
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.json_types import JsonObject, JsonValue
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.numpyro_json import ArrayLoader
    from nof1_causal_lab.sampler_config import SamplerSpec


@dataclass(frozen=True)
class FitComputeInput:
    """Pinned, self-contained inputs; no remote workspace or storage credentials."""

    model_json: str
    outcome: ConstructId | None
    panel_parquet: bytes
    time_origin: datetime | None
    arrays: dict[str, bytes]
    sampler: SamplerSpec
    compute_loo_diagnostics: bool


@dataclass(frozen=True)
class FitComputeResult:
    """Complete fitted output to validate and persist through the caller's store."""

    model_json: str
    report_json: str
    engine_evidence: JsonObject
    arrays: dict[str, bytes]


def _array_references(value: JsonValue) -> set[str]:
    if isinstance(value, dict):
        if "array_ref" in value:
            return {TypeAdapter(str).validate_python(value["array_ref"])}
        return {identity for child in value.values() for identity in _array_references(child)}
    if isinstance(value, list):
        return {identity for child in value for identity in _array_references(child)}
    return set()


def execute_fit_compute(payload: FitComputeInput) -> FitComputeResult:
    """Use the ordinary fit/report/conditioning code with an in-memory array store."""
    import polars as pl

    from nof1_causal_lab.actions.fit import fit

    arrays = {identity: decode_array(identity, data) for identity, data in payload.arrays.items()}
    written: dict[str, bytes] = {}

    def write_array(values: np.ndarray) -> str:
        identity, data = encode_array(values)
        written[identity] = data
        arrays[identity] = decode_array(identity, data)
        return identity

    from nof1_causal_lab.models.model_structure import StructuralSelection

    model = ModelSpec.model_validate_json(
        payload.model_json, context={"distribution_array_loader": arrays.__getitem__}
    )
    result = fit(
        selection=StructuralSelection(model, payload.outcome),
        data_for_model=pl.read_parquet(io.BytesIO(payload.panel_parquet)),
        time_origin=payload.time_origin,
        sampler=payload.sampler,
        array_writer=write_array,
        array_loader=arrays.__getitem__,
        compute_loo_diagnostics=payload.compute_loo_diagnostics,
    )
    conditioned = result["_model"]
    evidence = result["engine_evidence"]
    report = result["report"]
    return FitComputeResult(
        model_json=conditioned.model_dump_json(),
        report_json=report.model_dump_json(),
        engine_evidence=evidence.model_dump(mode="json"),
        arrays=written,
    )


def _fit_gpu(payload: FitComputeInput) -> FitComputeResult:
    import jax
    import modal

    logging.basicConfig(level=logging.INFO)
    if jax.default_backend() != "gpu":
        raise RuntimeError("Modal fit requires a GPU backend")
    logging.getLogger(__name__).info("Fit compute devices: %s", jax.devices())
    result = execute_fit_compute(payload)
    modal.Volume.from_name("nof1-cached-fit-cache").commit()
    return result


def _dispatch_fit(payload: FitComputeInput, *, timeout: int = 10800) -> FitComputeResult:
    import modal

    from nof1_causal_lab.actions.modal_runners import GPU_A100_80GB, gpu_image

    # A separate ephemeral app per call supports concurrent local study workers
    # and ships their current source; no deployed production function is replaced.
    app = modal.App("nof1-causal-lab-pipeline")
    remote_fit = app.function(
        image=gpu_image,
        # Function environment overrides image defaults without adding a build
        # layer after its add_local_* source mounts (Modal forbids that order).
        env={
            "DEPLOYMENT_ENV": "development",
            "JAX_PLATFORMS": "cuda",
            "JAX_COMPILATION_CACHE_DIR": "/cache/jax",
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        },
        gpu=GPU_A100_80GB,
        cpu=8,
        memory=32768,
        timeout=timeout,
        retries=0,
        volumes={"/cache": modal.Volume.from_name("nof1-cached-fit-cache", create_if_missing=True)},
    )(_fit_gpu)
    with modal.enable_output(), app.run():
        return remote_fit.remote(payload)


def fit_on_modal(
    *,
    selection: StructuralSelection,
    data_for_model: pl.DataFrame,
    time_origin: datetime | None,
    sampler: SamplerSpec,
    array_writer: Callable[[np.ndarray], str],
    array_loader: ArrayLoader,
    compute_loo_diagnostics: bool,
) -> FitResult:
    """Transfer pinned values, compute once on Modal, then retain returned arrays locally."""
    inputs: dict[str, bytes] = {}
    for identity in _array_references(selection.model.model_dump(mode="json")):
        actual, data = encode_array(array_loader(identity))
        if actual != identity:
            raise ValueError("Fit input array does not match its content identity")
        inputs[identity] = data
    panel = io.BytesIO()
    data_for_model.write_parquet(panel)
    result = _dispatch_fit(
        FitComputeInput(
            model_json=selection.model.model_dump_json(),
            outcome=selection.outcome,
            panel_parquet=panel.getvalue(),
            time_origin=time_origin,
            arrays=inputs,
            sampler=sampler,
            compute_loo_diagnostics=compute_loo_diagnostics,
        )
    )
    # Validate the complete response before the first local write. Exceptions
    # propagate to the action's normal error path; there is no local retry.
    arrays = {identity: decode_array(identity, data) for identity, data in result.arrays.items()}
    report = InferenceReport.model_validate_json(result.report_json)
    conditioned = ModelSpec.model_validate_json(result.model_json)
    if _array_references(conditioned.model_dump(mode="json")) - (arrays.keys() | inputs.keys()):
        raise ValueError("Modal fit returned an unresolved numerical array reference")
    for identity, values in arrays.items():
        if array_writer(values) != identity:
            raise ValueError("Fit output array does not match its content identity")
    conditioned = ModelSpec.model_validate_json(
        result.model_json, context={"distribution_array_loader": array_loader}
    )
    return {
        "report": report,
        "_model": conditioned,
        "engine_evidence": ParticleMCMCEvidence.model_validate(result.engine_evidence),
    }
