"""Lossless reads of recorded data and draws for the model workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.identity import ParameterRef
from nof1_causal_lab.machine.visual_models import (
    EmpiricalPoint,
    ObservationHistory,
    ParameterDrawColumn,
    ParameterDraws,
    PathSeries,
    PredictiveHistory,
    RecordedPath,
    SimulationPaths,
)
from nof1_causal_lab.utils.time_coordinates import serialization_origin

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.identity import IndicatorId
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.machine.snapshots import ModelReader


def finite_values(values: np.ndarray) -> tuple[float | None, ...]:
    return tuple(float(value) if np.isfinite(value) else None for value in values)


def empirical_points(values: np.ndarray) -> tuple[EmpiricalPoint, ...]:
    unique, counts = np.unique(values[np.isfinite(values)], return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum() if counts.size else []
    return tuple(
        EmpiricalPoint(value=value, probability=probability, count=int(count))
        for value, probability, count in zip(unique, cumulative, counts, strict=True)
    )


def observation_history(
    reader: ModelReader, indicator_id: IndicatorId
) -> ObservationHistory | None:
    metadata = reader.data_metadata
    if metadata is None:
        return None
    variable = next((v for v in metadata.value.variables if v.id == indicator_id), None)
    if variable is None:
        return None
    panel = (
        reader.store.read_parquet_file(
            "panel", reader.state.current["panel"].revision, "panel.parquet"
        )
        .filter(pl.col("indicator_id") == indicator_id)
        .sort("anchor_time")
    )
    origin = serialization_origin(metadata.value.time_origin)

    def days(column):
        return tuple(
            (value - origin).total_seconds() / 86400 if value is not None else None
            for value in panel[column]
        )

    values = panel["value"].cast(pl.Float64).to_numpy()
    return ObservationHistory(
        indicator_id=indicator_id,
        label=variable.name,
        times=days("anchor_time"),
        values=finite_values(values),
        support_start=days("support_start"),
        support_end=days("support_end"),
        time_origin=metadata.value.time_origin,
        levels=variable.ordinal_levels or variable.categorical_levels,
        empirical=empirical_points(values),
    )


def predictive_history(reader: ModelReader, indicator_id: IndicatorId) -> PredictiveHistory | None:
    """Recover the saved check's exact schedule from its pinned inputs, without prediction."""
    from nof1_causal_lab.actions.data_checks import read_data_metadata
    from nof1_causal_lab.actions.predictive_checks import fitted_law_report
    from nof1_causal_lab.machine.store import read_model
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.observation_support import (
        augment_wide_data_with_support_boundaries,
    )
    from nof1_causal_lab.models.ssm.runtime import project_observation_data

    check = reader.state.checks.predictive if reader.state.checks else None
    if check is None or check.predictive_checks is None or check.panel_revision is None:
        return None
    overlay = next(
        (item for item in check.predictive_checks.overlays if item.indicator_id == indicator_id),
        None,
    )
    if overlay is None:
        return None
    model = read_model(reader.store, check.model_revision)
    origin = read_data_metadata(reader.store, check.panel_revision).time_origin
    if check.law.fitted_model_revision is not None:
        origin = fitted_law_report(
            reader.repository.attempts(), check.law.fitted_model_revision
        ).time_origin
    panel = reader.store.read_parquet_file("panel", check.panel_revision, "panel.parquet")
    wide, rows = project_observation_data(panel, model_spec=model, time_origin=origin)
    wide = augment_wide_data_with_support_boundaries(
        rows, wide, numeric.observation_names(model), time_origin=origin
    )
    times = tuple(float(value) for value in wide["time"])
    if len(times) != len(overlay.observed):
        raise ValueError("Saved predictive series do not match their pinned observation schedule")
    likelihood = next(
        law for indicator, law in model.iter_likelihoods() if indicator.id == indicator_id
    )
    return PredictiveHistory(
        times=times, time_origin=origin, standardized=likelihood.standardized, overlay=overlay
    )


def simulation_paths(reader: ModelReader, *, start: int, count: int) -> SimulationPaths | None:
    """Page original paired histories; never thin time or synthesize paths from quantiles."""
    saved = reader.simulation()
    if saved is None:
        return None
    return recorded_simulation_paths(saved.value, reader.store.read_array, start=start, count=count)


def recorded_simulation_paths(
    report: SimulationReport,
    read_array: Callable[[str], np.ndarray],
    *,
    start: int,
    count: int,
) -> SimulationPaths:
    """Project stored arrays into plot coordinates without reducing their histories."""
    if start >= report.draws:
        raise ValueError("Draw page starts past the saved simulation")
    stop = min(start + count, report.draws)
    latent = read_array(report.latent_paths)[start:stop]
    observed = read_array(report.observations)[start:stop]
    mask = read_array(report.observation_layout.mask)[start:stop]
    reference = (
        read_array(report.reference_latent_paths)[start:stop]
        if report.reference_latent_paths is not None
        else None
    )
    reference_observed = (
        read_array(report.reference_observations)[start:stop]
        if report.reference_observations is not None
        else None
    )

    def paths(values):
        return tuple(
            RecordedPath(draw=start + index, values=finite_values(row))
            for index, row in enumerate(values)
        )

    effect = None
    if report.causal_result is not None:
        assert reference is not None
        outcome = report.state_ids.index(report.causal_result.outcome)
        effect = PathSeries(
            label=report.causal_result.labels[report.causal_result.outcome],
            action=paths(latent[:, :, outcome] - reference[:, :, outcome]),
        )
    return SimulationPaths(
        times=report.times,
        time_origin=report.time_origin,
        total_draws=report.draws,
        start=start,
        count=stop - start,
        states={
            identity: PathSeries(
                label=report.predictive.states[identity].label,
                action=paths(latent[:, :, index]),
                reference=paths(reference[:, :, index]) if reference is not None else (),
            )
            for index, identity in enumerate(report.state_ids)
        },
        indicators={
            variable.id: PathSeries(
                label=variable.name,
                action=paths(np.where(mask[:, :, index], observed[:, :, index], np.nan)),
                reference=paths(
                    np.where(mask[:, :, index], reference_observed[:, :, index], np.nan)
                )
                if reference_observed is not None
                else (),
                levels=variable.ordinal_levels or variable.categorical_levels,
            )
            for index, variable in enumerate(report.observation_layout.variables)
        },
        effect=effect,
    )


def parameter_draws(reader: ModelReader) -> ParameterDraws:
    """Read the fitted joint law instead of the report's small selection of pair plots."""
    from nof1_causal_lab.actions.predictive_checks import law_provenance
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
    from nof1_causal_lab.numpyro_json import empirical_atoms

    model = reader.model
    if model is None:
        return ParameterDraws(columns=(), unavailable_reason="No model at this revision.")
    provenance = law_provenance(reader.store, reader.state.current["model"], model, None)
    if provenance.kind != "fitted":
        return ParameterDraws(
            columns=(),
            unavailable_reason="This revision has no complete retained joint posterior. Recorded summary plots cannot recover missing draws.",
        )
    bindings, _ = parameter_bindings(model)
    columns = []
    for identity in sorted(
        {
            parameter.distribution
            for parameter in model.execution_parameters
            if parameter.distribution
        }
    ):
        members = [b for b in bindings if model.parameter(b.parameter_id).distribution == identity]
        layout = JointLawLayout.from_bindings(
            members,
            parameters=[b.parameter_id for b in members],
            constructs=[c.id for c in model.constructs if c.distribution == identity],
            time_points=model.time_points,
        )
        atoms = empirical_atoms(model.distributions[identity])
        for binding in members:
            for element, label in binding.elements.items():
                columns.append(
                    ParameterDrawColumn(
                        label=label,
                        subject=ParameterRef(parameter_id=binding.parameter_id, element_id=element),
                        values=tuple(float(v) for v in atoms[:, layout.parameter_columns[element]]),
                        empirical=empirical_points(atoms[:, layout.parameter_columns[element]]),
                    )
                )
    return ParameterDraws(columns=tuple(columns))
