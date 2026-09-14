"""Predictive assessments and scientific posterior summaries."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .effects import HistogramBin
from .identity import IndicatorId, ParameterRef


class LOODiagnostics(BaseModel):
    """Leave-one-out diagnostics assess predictive fit and the reliability of its cross-
    validation estimate.

    Exact emission factors on joint parameter/state draws support holding out
    one measurement row. All other rows, including future rows, are available
    for interpolation. PSIS reliability is assessed with Pareto-k diagnostics.
    """

    elpd_loo: float
    p_loo: float
    se: float
    n_data_points: int
    observation_unit: Literal["measurement_row"] = "measurement_row"
    prediction_task: Literal["interpolation_given_other_measurements"] = (
        "interpolation_given_other_measurements"
    )
    likelihood_source: Literal["exact_emission_on_joint_particle_draws"] = (
        "exact_emission_on_joint_particle_draws"
    )
    pareto_k: list[float] | None = None
    n_bad_k: int | None = None
    loo_pit: list[float] | None = None


class PosteriorEstimate(BaseModel):
    """A posterior estimate reports a mean and a credible interval with explicit semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mean: float
    lower: float
    upper: float
    interval_kind: Literal["hdi", "equal_tail"]
    interval_mass: float = Field(
        gt=0, lt=1, description="Posterior probability mass of the interval."
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "PosteriorEstimate":
        if self.lower > self.upper:
            raise ValueError("Posterior interval lower bound must not exceed its upper bound")
        return self


class PosteriorMarginal(PosteriorEstimate):
    """A posterior marginal summarizes uncertainty in one scalar parameter and supplies its
    density plot.
    """

    parameter: str
    subject: ParameterRef
    x_values: list[float]
    density: list[float]
    sd: float = Field(ge=0)


class PosteriorPair(BaseModel):
    """A posterior pair supplies joint samples of two parameters to visualize their dependence."""

    param_x: str
    subject_x: ParameterRef
    param_y: str
    subject_y: ParameterRef
    x_values: list[float]
    y_values: list[float]
    divergent: list[bool] | None = None


class PPCWarning(BaseModel):
    """A predictive-check finding records whether one indicator passes a calibration,
    dependence, or variance check.
    """

    indicator_id: IndicatorId
    check_type: Literal["calibration", "autocorrelation", "variance"]
    message: str
    value: float
    passed: bool = True


class PPCOverlay(BaseModel):
    """A predictive overlay compares observed values with posterior predictive bands for one
    indicator.

    Provides the data for Gabry's ppc_dens_overlay / ppc_ribbon plots:
    observed time series vs posterior predictive quantile bands.
    Optionally includes individual y_rep draw lines for spaghetti plots.
    """

    indicator_id: IndicatorId
    observed: list[float | None]
    q025: list[float]
    q25: list[float]
    median: list[float]
    q75: list[float]
    q975: list[float]
    spaghetti_draws: list[list[float]] = Field(default_factory=list)


class PPCTestStat(BaseModel):
    """A predictive test statistic compares an observed summary with its distribution under
    replicated data.

    Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
    with a vertical line at T(y_observed).
    """

    indicator_id: IndicatorId
    stat_name: Literal["mean", "sd", "min", "max"]
    observed_value: float
    rep_values: list[float]
    p_value: float | None
    histogram: list[HistogramBin]


class PosteriorPredictiveChecks(BaseModel):
    """Posterior predictive checks report exact-model checks and their supporting plot data."""

    per_variable_warnings: list[PPCWarning] = Field(default_factory=list)
    checked: bool = False
    n_subsample: int = 0
    overlays: list[PPCOverlay] = Field(default_factory=list)
    test_stats: list[PPCTestStat] = Field(default_factory=list)
