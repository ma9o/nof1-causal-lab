"""Synthetic nonlinear SSM fixture for PG/RBPF recovery notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    linear_effect,
    restoring_force,
)
from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.likelihoods import observation_law, revise_law
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

LATENT_NAMES = [
    "affective_state",
    "sleep_quality",
    "physical_activity",
]

INPUT_NAMES = [
    "serotonergic_exposure",
    "seasonal_load",
]

MANIFEST_NAMES = [
    "state_of_mind_valence",
    "late_night_message_count",
    "total_sleep_hours",
    "sleep_efficiency_pct",
    "sleep_onset_latency_min",
    "nightly_hrv_ms",
    "daily_step_count",
    "exercise_minutes",
    "journal_affect_severity",
    "journal_rumination_intensity",
    "gp_clinical_severity",
]

MANIFEST_DISTS = [
    DistributionFamily.GAUSSIAN,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAMMA,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
]

MANIFEST_LINKS = [
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
]

TRUE_DECAY = np.asarray([0.28, 0.35, 0.32], dtype=np.float32)
TRUE_HILL_BY_SITE = {
    "vf_3_Emax": 0.06,  # diminishing returns: sleep_quality -> affective_state
    "vf_3_EC50": 0.70,
    "vf_3_n": 2.0,
    "vf_5_Emax": 0.045,  # diminishing returns: physical_activity -> affective_state
    "vf_5_EC50": 0.75,
    "vf_5_n": 2.0,
    "vf_6_Emax": 0.035,  # positive mood improves sleep with saturation
    "vf_6_EC50": 0.80,
    "vf_6_n": 2.0,
    "vf_7_Emax": 0.030,  # behavioral activation saturates
    "vf_7_EC50": 0.80,
    "vf_7_n": 2.0,
}
TRUE_MULTIPLICATIVE_BY_SITE = {
    "vf_4_weight": 0.025,  # sleep and activity reinforce mood together
    "vf_8_weight": 0.020,  # mood and sleep jointly support activity
}
TRUE_OBS_R = 12.0
TRUE_OBS_SHAPE = 8.0

TRUE_DRIFT = np.asarray(
    [
        [-0.28, 0.00, 0.00],
        [0.00, -0.35, 0.00],
        [0.00, 0.00, -0.32],
    ],
    dtype=np.float32,
)

TRUE_DIFFUSION_SD = np.asarray([0.18, 0.16, 0.20], dtype=np.float32)
TRUE_T0_MEAN = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
TRUE_T0_SD = np.asarray([0.55, 0.50, 0.55], dtype=np.float32)

TRUE_INPUT_EFFECT = np.asarray(
    [
        [0.08, 0.03],
        [0.02, 0.00],
        [0.04, 0.00],
    ],
    dtype=np.float32,
)
TRUE_INPUT_EFFECT_POSITIONS = (
    (0, 0),  # serotonergic_exposure -> affective_state
    (0, 1),  # seasonal_load -> affective_state
    (1, 0),  # serotonergic_exposure -> sleep_quality
    (2, 0),  # serotonergic_exposure -> physical_activity
)

TRUE_LOADINGS = np.asarray(
    [
        [0.75, 0.00, 0.00],
        [-0.35, 0.00, 0.00],
        [0.00, 0.35, 0.00],
        [0.00, 2.50, 0.00],
        [0.00, -0.25, 0.00],
        [0.00, 3.00, 0.00],
        [0.00, 0.00, 0.25],
        [0.00, 0.00, 5.00],
        [-0.70, 0.00, 0.00],
        [-1.00, 0.00, 0.00],
        [-1.20, 0.00, 0.00],
    ],
    dtype=np.float32,
)

TRUE_MANIFEST_MEANS = np.asarray(
    [
        0.20,
        np.log(2.0),
        7.10,
        87.0,
        np.log(18.0),
        45.0,
        np.log(80.0),
        28.0,
        2.0,
        4.0,
        5.0,
    ],
    dtype=np.float32,
)

TRUE_MANIFEST_SD = np.asarray(
    [0.25, 1.0, 0.20, 1.00, 1.0, 1.50, 1.0, 2.00, 0.30, 0.40, 0.35],
    dtype=np.float32,
)

_manifest_order = sorted(
    range(len(MANIFEST_NAMES)), key=lambda row: int(np.flatnonzero(TRUE_LOADINGS[row])[0])
)
MANIFEST_NAMES = [MANIFEST_NAMES[i] for i in _manifest_order]
MANIFEST_DISTS = [MANIFEST_DISTS[i] for i in _manifest_order]
MANIFEST_LINKS = [MANIFEST_LINKS[i] for i in _manifest_order]
TRUE_LOADINGS = TRUE_LOADINGS[_manifest_order]
TRUE_MANIFEST_MEANS = TRUE_MANIFEST_MEANS[_manifest_order]
TRUE_MANIFEST_SD = TRUE_MANIFEST_SD[_manifest_order]

EXACT_MEASUREMENT_MANIFEST_INDICES = tuple(
    idx
    for idx, (dist, link) in enumerate(zip(MANIFEST_DISTS, MANIFEST_LINKS, strict=True))
    if (dist == DistributionFamily.GAUSSIAN and link == LinkFunction.IDENTITY)
    or (dist == DistributionFamily.NEGATIVE_BINOMIAL and link == LinkFunction.LOG)
)
EXACT_MEASUREMENT_SUPPORT = np.asarray(
    [idx in EXACT_MEASUREMENT_MANIFEST_INDICES for idx in range(len(MANIFEST_NAMES))],
    dtype=bool,
)
# Gaussian/identity manifest means are PINNED at truth: their raw-scale
# locations (e.g. sleep_efficiency_pct ~ 87) sit tens of prior sd outside the
# canonical Normal(0, 2) manifest-mean prior, so leaving them free makes the
# truth unreachable and distorts every coupled posterior coordinate. Production
# removes the raw location via deterministic centering (prepare_model_runtime);
# this fixture bypasses that path, so the location is pinned instead — it is
# not a recovery axis here. Count intercepts (NegBin/log) stay free: their
# log-scale truths sit within the canonical prior.
MEASUREMENT_MEANS_FREE_SUPPORT = np.asarray(
    [
        MANIFEST_DISTS[idx] == DistributionFamily.NEGATIVE_BINOMIAL
        and MANIFEST_LINKS[idx] == LinkFunction.LOG
        for idx in range(len(MANIFEST_NAMES))
    ],
    dtype=bool,
)
ANCHOR_LOADING_POSITIONS = (
    (0, 0),  # state_of_mind_valence anchors affective_state
    (MANIFEST_NAMES.index("total_sleep_hours"), 1),  # sleep anchor
    (MANIFEST_NAMES.index("daily_step_count"), 2),  # activity anchor
)
MEASUREMENT_LOADINGS_FREE_SUPPORT = (
    (~np.isclose(TRUE_LOADINGS, 0.0))
    & EXACT_MEASUREMENT_SUPPORT[:, None]
    & np.asarray(
        [
            [(row, col) not in ANCHOR_LOADING_POSITIONS for col in range(TRUE_LOADINGS.shape[1])]
            for row in range(TRUE_LOADINGS.shape[0])
        ],
        dtype=bool,
    )
)
MEASUREMENT_MEANS_FREE_POSITIONS = tuple(
    idx for idx in range(len(MANIFEST_NAMES)) if bool(MEASUREMENT_MEANS_FREE_SUPPORT[idx])
)
MEASUREMENT_LOADINGS_FREE_POSITIONS = tuple(
    (row, col)
    for row in range(TRUE_LOADINGS.shape[0])
    for col in range(TRUE_LOADINGS.shape[1])
    if bool(MEASUREMENT_LOADINGS_FREE_SUPPORT[row, col])
)


@dataclass(frozen=True)
class SyntheticNonlinearData:
    observations: jnp.ndarray
    times: jnp.ndarray
    latent: jnp.ndarray
    observation_support: ObservationSupportRuntime


def build_synthetic_nonlinear_spec(*, diffusion_scale: float = 1.0) -> ModelSpec:
    """Author the nonlinear recovery fixture as one scientific definition."""
    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.expressions import coefficient
    from nof1_causal_lab.artifacts.identity import (
        ConstructRef,
        EdgeRef,
        IndicatorRef,
        MechanismRef,
        scientific_id,
    )
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.prior_planning import complete_model

    definitions = {}

    def quantity(kind, owners, value=None):
        identity = scientific_id("parameter", [kind.value, sorted(owner.id for owner in owners)])
        definitions[identity] = ParameterSpec(
            id=identity,
            name=identity,
            description="Synthetic fixture quantity",
            value=value,
        )
        return identity

    constructs = []
    for column, name in enumerate(LATENT_NAMES):
        identity = ConstructId(f"construct:{name}")
        owner = ConstructRef(id=identity)
        mechanism_id = f"mechanism:relaxation-{name}"
        decay = quantity(SiteKind.DYNAMICS_DECAY, [owner, MechanismRef(id=mechanism_id)])
        indicators = []
        for row, obs_name in enumerate(MANIFEST_NAMES):
            if TRUE_LOADINGS[row, column] == 0:
                continue
            indicator_id = f"indicator:{obs_name}"
            refs = [owner, IndicatorRef(id=indicator_id)]
            loading = quantity(
                SiteKind.LOADING,
                refs,
                None
                if MEASUREMENT_LOADINGS_FREE_SUPPORT[row, column]
                else float(TRUE_LOADINGS[row, column]),
            )
            intercept = quantity(
                SiteKind.MANIFEST_MEANS,
                refs,
                None if MEASUREMENT_MEANS_FREE_SUPPORT[row] else float(TRUE_MANIFEST_MEANS[row]),
            )
            scale = (
                quantity(SiteKind.MANIFEST_VAR_DIAG, refs)
                if MANIFEST_DISTS[row].uses_manifest_noise
                else 0
            )
            dtype = (
                "count"
                if MANIFEST_DISTS[row] == DistributionFamily.NEGATIVE_BINOMIAL
                else "continuous"
            )
            indicators.append(
                IndicatorSpec(
                    id=indicator_id,
                    name=obs_name,
                    how_to_measure="Read the synthetic observation",
                    construct_polarity="negative" if TRUE_LOADINGS[row, column] < 0 else "positive",
                    measurement_dtype=dtype,
                    aggregation="last",
                    likelihood=revise_law(
                        LikelihoodSpec(
                            law=observation_law(identity, MANIFEST_DISTS[row], MANIFEST_LINKS[row]),
                            reasoning="Synthetic measurement law",
                        ),
                        lambda node, loading=loading, intercept=intercept, scale=scale: (
                            node.model_copy(
                                update={
                                    "value": {
                                        "loading": loading,
                                        "observation_intercept": intercept,
                                        "observation_scale": scale,
                                    }[node.role]
                                }
                            )
                            if isinstance(node, CoefficientExpression)
                            and node.role
                            in {"loading", "observation_intercept", "observation_scale"}
                            else node
                        ),
                    ),
                )
            )
        constructs.append(
            ConstructSpec(
                id=identity,
                name=name,
                description="Synthetic latent state",
                role="endogenous",
                temporal_status="time_varying",
                indicators=tuple(indicators),
                coefficients=(
                    coefficient(quantity(SiteKind.DIFFUSION_DIAG, [owner]), "diffusion_scale"),
                    coefficient(
                        quantity(SiteKind.T0_MEANS, [owner], float(TRUE_T0_MEAN[column])),
                        "initial_mean",
                    ),
                    coefficient(
                        quantity(SiteKind.T0_VAR_DIAG, [owner], float(TRUE_T0_SD[column])),
                        "initial_scale",
                    ),
                ),
                dynamics=(
                    DynamicsMechanismSpec(
                        id=mechanism_id,
                        expression=restoring_force(
                            identity,
                            center=0,
                            stiffness=decay,
                            quartic=0,
                        ),
                    ),
                ),
            )
        )
    edges = {}
    terms = {}

    def edge(cause, effect, lagged=True):
        pair = (cause, effect)
        if pair not in edges:
            identity = f"edge:{cause}-{effect}"
            edges[pair] = CausalEdgeSpec(
                id=identity,
                cause=next(construct for construct in constructs if construct.name == cause),
                effect=next(construct for construct in constructs if construct.name == effect),
                lagged=lagged,
                description="Synthetic causal relationship",
            )
            terms[identity] = []
        return edges[pair]

    def refs(owner, mechanism_id):
        return [
            ConstructRef(id=owner.cause.id),
            ConstructRef(id=owner.effect.id),
            EdgeRef(id=owner.id),
            MechanismRef(id=mechanism_id),
        ]

    for source, target, prefix in [(1, 0, "vf_3"), (2, 0, "vf_5"), (0, 1, "vf_6"), (0, 2, "vf_7")]:
        owner = edge(LATENT_NAMES[source], LATENT_NAMES[target])
        mechanism_id = f"mechanism:hill-{source}-{target}"
        terms[owner.id].append(
            DynamicsMechanismSpec(
                id=mechanism_id,
                expression=expr_hill(
                    expr_state(owner.cause.id),
                    emax=quantity(SiteKind.HILL_EMAX, refs(owner, mechanism_id)),
                    ec50=TRUE_HILL_BY_SITE[prefix + "_EC50"],
                    n=TRUE_HILL_BY_SITE[prefix + "_n"],
                ),
            )
        )
    for source, moderator, target in [(1, 2, 0), (0, 1, 2)]:
        owner = edge(LATENT_NAMES[source], LATENT_NAMES[target])
        edge(LATENT_NAMES[moderator], LATENT_NAMES[target])
        mechanism_id = f"mechanism:interaction-{source}-{moderator}-{target}"
        terms[owner.id].append(
            DynamicsMechanismSpec(
                id=mechanism_id,
                expression=linear_effect(
                    owner.cause.id,
                    quantity(
                        SiteKind.DYNAMICS_WEIGHT,
                        [
                            *refs(owner, mechanism_id),
                            ConstructRef(id=f"construct:{LATENT_NAMES[moderator]}"),
                        ],
                    ),
                )
                * expr_state(ConstructId(f"construct:{LATENT_NAMES[moderator]}")),
            )
        )
    for name in INPUT_NAMES:
        indicator = IndicatorSpec(
            id=f"indicator:{name}",
            name=name,
            likelihood=LikelihoodSpec(
                law=observation_law(
                    ConstructId(f"construct:{name}"),
                    DistributionFamily.DELTA,
                    LinkFunction.IDENTITY,
                ),
                reasoning="Synthetic driver values are observed exactly at each anchor.",
            ),
            how_to_measure="Read the exact synthetic driver",
            construct_polarity="positive",
            measurement_dtype="continuous",
            aggregation="last",
        )
        constructs.append(
            ConstructSpec(
                id=f"construct:{name}",
                name=name,
                description="Known synthetic input",
                role="exogenous",
                temporal_status="time_varying",
                indicators=(indicator,),
                coefficients=(
                    coefficient(0, "initial_mean"),
                    coefficient(1, "initial_scale"),
                    coefficient(1, "diffusion_scale"),
                ),
                dynamics=(
                    DynamicsMechanismSpec(
                        id=f"mechanism:driver-{name}",
                        expression=coefficient(0, "intercept"),
                    ),
                ),
            )
        )
    for row, column in TRUE_INPUT_EFFECT_POSITIONS:
        owner = edge(INPUT_NAMES[column], LATENT_NAMES[row], False)
        mechanism_id = f"mechanism:input-{row}-{column}"
        terms[owner.id].append(
            DynamicsMechanismSpec(
                id=mechanism_id,
                expression=linear_effect(
                    owner.cause.id, quantity(SiteKind.DYNAMICS_WEIGHT, refs(owner, mechanism_id))
                ),
            )
        )
    model = ModelSpec(
        edges=tuple(
            owner.model_copy(update={"mechanisms": tuple(terms[owner.id])})
            for owner in sorted(
                edges.values(),
                key=lambda edge: (
                    [*LATENT_NAMES, *INPUT_NAMES].index(edge.cause.name),
                    [*LATENT_NAMES, *INPUT_NAMES].index(edge.effect.name),
                ),
            )
        ),
        parameters=tuple(definitions.values()),
        measurement_clock="1d",
    )
    model = complete_model(model)
    # Off-truth priors are owned by the same scientific quantities as every other law.
    overrides = {
        SiteKind.LOADING: dist.Normal(0, 2.5),
        SiteKind.DIFFUSION_DIAG: dist.HalfNormal(0.4 * float(diffusion_scale)),
        SiteKind.OBS_R: dist.LogNormal(float(np.log(6.0)), 0.6),
        SiteKind.OBS_SHAPE: dist.LogNormal(float(np.log(5.0)), 0.6),
    }
    from nof1_causal_lab.models.model_distributions import with_parameter_distributions

    return with_parameter_distributions(
        model,
        {
            parameter.id: overrides[model.parameter_context(parameter.id).quantity]
            for parameter in model.parameters
            if parameter.value is None
            and model.parameter_context(parameter.id).quantity in overrides
        },
    )


def build_synthetic_nonlinear_model(
    data: SyntheticNonlinearData | None = None,
    *,
    include_interval_support: bool = False,
    diffusion_scale: float = 1.0,
) -> SSMModel:
    model = SSMModel(build_synthetic_nonlinear_spec(diffusion_scale=diffusion_scale))
    if data is not None and include_interval_support:
        model.set_observation_support(data.observation_support)
    return model


def _build_transition_inputs(T: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    serotonergic = 1.0 / (1.0 + np.exp(-(t - 8.0) / 2.0))
    seasonal = np.sin(2.0 * np.pi * t / max(T - 1, 1))
    return np.column_stack([serotonergic, seasonal]).astype(np.float32)


def _build_gp_interval_support(T: int, gp_rows: np.ndarray, window: int = 3):
    n_manifest = len(MANIFEST_NAMES) + len(INPUT_NAMES)
    support_start_times = np.full((T, n_manifest), np.nan, dtype=np.float64)
    support_end_times = np.full((T, n_manifest), np.nan, dtype=np.float64)
    interval_prev_coeffs = np.zeros((T, n_manifest), dtype=np.float64)
    interval_curr_coeffs = np.zeros((T, n_manifest), dtype=np.float64)
    interval_weights = np.zeros((T, n_manifest), dtype=np.float64)
    emission_slot_indices = np.full((T, n_manifest), -1, dtype=np.int64)
    gp_idx = MANIFEST_NAMES.index("gp_clinical_severity")

    for end in gp_rows:
        start = max(0, int(end) - window)
        support_start_times[end, gp_idx] = float(start)
        support_end_times[end, gp_idx] = float(end)
        emission_slot_indices[end, gp_idx] = 0
        for step in range(start + 1, int(end) + 1):
            interval_prev_coeffs[step, gp_idx] += 0.5
            interval_curr_coeffs[step, gp_idx] += 0.5
            interval_weights[step, gp_idx] += 1.0

    support_kinds = [None] * n_manifest
    summary_operators = ["last"] * n_manifest
    observation_windows = [None] * n_manifest
    support_kinds[gp_idx] = "interval"
    summary_operators[gp_idx] = "mean"
    observation_windows[gp_idx] = f"{window}d"

    return _make_observation_support_runtime(
        anchor_times=np.arange(T, dtype=np.float64),
        manifest_names=[*MANIFEST_NAMES, *INPUT_NAMES],
        support_kinds=support_kinds,
        summary_operators=summary_operators,
        observation_windows=observation_windows,
        support_start_times=support_start_times,
        support_end_times=support_end_times,
        interval_prev_coeffs=interval_prev_coeffs,
        interval_curr_coeffs=interval_curr_coeffs,
        interval_weights=interval_weights,
        emission_slot_indices=emission_slot_indices,
    )


def _make_observation_support_runtime(**kwargs) -> ObservationSupportRuntime:
    """Build the runtime while accepting compact 2D interval coefficients."""
    support_kinds = kwargs["support_kinds"]
    kwargs.setdefault(
        "summary_operators",
        ["mean" if kind == "interval" else "last" for kind in support_kinds],
    )
    kwargs.setdefault(
        "anchor_policies",
        [
            "support_start" if operator == "first" else "support_end"
            for operator in kwargs["summary_operators"]
        ],
    )
    prev = np.asarray(kwargs["interval_prev_coeffs"], dtype=np.float64)
    curr = np.asarray(kwargs["interval_curr_coeffs"], dtype=np.float64)
    weights = np.asarray(kwargs["interval_weights"], dtype=np.float64)
    if prev.ndim == 2:
        prev = prev[..., None]
        curr = curr[..., None]
        weights = weights[..., None]
    kwargs["interval_prev_coeffs"] = prev
    kwargs["interval_curr_coeffs"] = curr
    kwargs["interval_weights"] = weights
    emission_slots = kwargs.get("emission_slot_indices")
    if emission_slots is None:
        support_end = np.asarray(kwargs["support_end_times"])
        emission_slots = np.where(np.isfinite(support_end), 0, -1).astype(np.int64)
    kwargs["emission_slot_indices"] = emission_slots
    return ObservationSupportRuntime(**kwargs)


def _sample_negative_binomial(rng: np.random.Generator, mean: np.ndarray, r: float) -> np.ndarray:
    gamma_rate = rng.gamma(shape=r, scale=mean / r)
    return rng.poisson(gamma_rate).astype(np.float32)


def _hill_effect(source_value: np.ndarray | float, prefix: str) -> np.ndarray | float:
    x = np.maximum(source_value, 0.0)
    emax = TRUE_HILL_BY_SITE[f"{prefix}_Emax"]
    ec50 = TRUE_HILL_BY_SITE[f"{prefix}_EC50"]
    n_hill = TRUE_HILL_BY_SITE[f"{prefix}_n"]
    x_n = x**n_hill
    return emax * x_n / (ec50**n_hill + x_n + 1e-12)


def _synthetic_nonlinear_drift(state: np.ndarray, transition_input: np.ndarray) -> np.ndarray:
    drift = TRUE_DRIFT @ state + TRUE_INPUT_EFFECT @ transition_input
    drift = drift.copy()
    drift[0] += _hill_effect(state[1], "vf_3")
    drift[0] += _hill_effect(state[2], "vf_5")
    drift[1] += _hill_effect(state[0], "vf_6")
    drift[2] += _hill_effect(state[0], "vf_7")
    drift[0] += TRUE_MULTIPLICATIVE_BY_SITE["vf_4_weight"] * state[1] * state[2]
    drift[2] += TRUE_MULTIPLICATIVE_BY_SITE["vf_8_weight"] * state[0] * state[1]
    return drift


def _sample_nonlinear_transition(
    rng: np.random.Generator,
    state: np.ndarray,
    transition_input: np.ndarray,
    *,
    diffusion_sd: np.ndarray,
    dt: float = 1.0,
    substeps: int = 8,
) -> np.ndarray:
    next_state = np.asarray(state, dtype=np.float32).copy()
    step = float(dt) / int(substeps)
    innovation_sd = np.asarray(diffusion_sd, dtype=np.float32) * np.sqrt(step)
    for _ in range(int(substeps)):
        innovation = rng.normal(0.0, innovation_sd).astype(np.float32)
        next_state = next_state + step * _synthetic_nonlinear_drift(next_state, transition_input)
        next_state = next_state + innovation
    return next_state.astype(np.float32)


def simulate_synthetic_nonlinear_data(
    T: int = 32, seed: int = 71, diffusion_scale: float = 1.0
) -> SyntheticNonlinearData:
    rng = np.random.default_rng(seed)
    transition_inputs = _build_transition_inputs(T)
    # Process-noise SD lever: scale=1 is the informative regime; >1 gives diffuse
    # dynamics where the prior-as-proposal advantage of cSMC should erode.
    diffusion_sd = np.asarray(TRUE_DIFFUSION_SD, dtype=np.float32) * float(diffusion_scale)
    latent = np.zeros((T, 3), dtype=np.float32)
    latent[0] = rng.normal(TRUE_T0_MEAN, TRUE_T0_SD).astype(np.float32)
    for time_idx in range(1, T):
        latent[time_idx] = _sample_nonlinear_transition(
            rng,
            latent[time_idx - 1],
            transition_inputs[time_idx],
            diffusion_sd=diffusion_sd,
        )

    linear_predictor = latent @ TRUE_LOADINGS.T + TRUE_MANIFEST_MEANS
    observations = np.full((T, len(MANIFEST_NAMES)), np.nan, dtype=np.float32)
    gaussian_indices = [
        i for i, family in enumerate(MANIFEST_DISTS) if family == DistributionFamily.GAUSSIAN
    ]
    for idx in gaussian_indices:
        observations[:, idx] = rng.normal(
            linear_predictor[:, idx],
            TRUE_MANIFEST_SD[idx],
        ).astype(np.float32)

    observations[:, MANIFEST_NAMES.index("late_night_message_count")] = _sample_negative_binomial(
        rng,
        np.exp(linear_predictor[:, MANIFEST_NAMES.index("late_night_message_count")]),
        TRUE_OBS_R,
    )
    observations[:, MANIFEST_NAMES.index("daily_step_count")] = _sample_negative_binomial(
        rng,
        np.exp(linear_predictor[:, MANIFEST_NAMES.index("daily_step_count")]),
        TRUE_OBS_R,
    )
    gamma_mean = np.exp(linear_predictor[:, MANIFEST_NAMES.index("sleep_onset_latency_min")])
    observations[:, MANIFEST_NAMES.index("sleep_onset_latency_min")] = rng.gamma(
        shape=TRUE_OBS_SHAPE,
        scale=gamma_mean / TRUE_OBS_SHAPE,
    ).astype(np.float32)

    dense_rows = np.arange(T)
    mood_rows = dense_rows[dense_rows % 4 == 0]
    journal_rows = np.asarray([3, 8, 14, 21, 27], dtype=np.int64)
    journal_rows = journal_rows[journal_rows < T]
    gp_rows = np.asarray([6, 18, 30], dtype=np.int64)
    gp_rows = gp_rows[gp_rows < T]

    sparse_keep = {
        0: mood_rows,
        MANIFEST_NAMES.index("journal_affect_severity"): journal_rows,
        MANIFEST_NAMES.index("journal_rumination_intensity"): journal_rows,
        MANIFEST_NAMES.index("gp_clinical_severity"): gp_rows,
    }
    for manifest_idx, rows in sparse_keep.items():
        mask = np.ones(T, dtype=bool)
        mask[rows] = False
        observations[mask, manifest_idx] = np.nan

    support = _build_gp_interval_support(T, gp_rows)
    gp_idx = MANIFEST_NAMES.index("gp_clinical_severity")
    for end in gp_rows:
        start = max(0, int(end) - 3)
        coeff_prev = np.zeros(T, dtype=np.float32)
        coeff_curr = np.zeros(T, dtype=np.float32)
        weights = np.zeros(T, dtype=np.float32)
        coeff_prev[start + 1 : int(end) + 1] = 0.5
        coeff_curr[start + 1 : int(end) + 1] = 0.5
        weights[start + 1 : int(end) + 1] = 1.0
        numerator = np.sum(
            coeff_prev
            * np.concatenate([[linear_predictor[0, gp_idx]], linear_predictor[:-1, gp_idx]])
            + coeff_curr * linear_predictor[:, gp_idx]
        )
        denominator = np.maximum(np.sum(weights), 1e-8)
        observations[end, gp_idx] = rng.normal(
            numerator / denominator,
            TRUE_MANIFEST_SD[gp_idx],
        )

    return SyntheticNonlinearData(
        observations=jnp.asarray(np.column_stack([observations, transition_inputs])),
        times=jnp.arange(T, dtype=jnp.float32),
        latent=jnp.asarray(np.column_stack([latent, transition_inputs])),
        observation_support=support,
    )


def _scalar_recovery_targets() -> dict[str, float]:
    """Bind retained truths through their mechanism identities to current sample sites."""
    from nof1_causal_lab.artifacts.identity import MechanismRef
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    truths = {
        **{
            f"mechanism:relaxation-{name}": float(TRUE_DECAY[index])
            for index, name in enumerate(LATENT_NAMES)
        },
        **{
            f"mechanism:hill-{source}-{target}": TRUE_HILL_BY_SITE[f"{prefix}_Emax"]
            for source, target, prefix in [
                (1, 0, "vf_3"),
                (2, 0, "vf_5"),
                (0, 1, "vf_6"),
                (0, 2, "vf_7"),
            ]
        },
        **{
            f"mechanism:input-{row}-{column}": float(TRUE_INPUT_EFFECT[row, column])
            for row, column in TRUE_INPUT_EFFECT_POSITIONS
        },
        "mechanism:interaction-1-2-0": TRUE_MULTIPLICATIVE_BY_SITE["vf_4_weight"],
        "mechanism:interaction-0-1-2": TRUE_MULTIPLICATIVE_BY_SITE["vf_8_weight"],
    }
    model = build_synthetic_nonlinear_spec()
    targets = {"obs_r": TRUE_OBS_R, "obs_shape": TRUE_OBS_SHAPE}
    for binding in parameter_bindings(model)[0]:
        if binding.component_index is not None:
            mechanism_id = next(
                ref.id
                for ref in model.parameter_context(binding.parameter_id).owners
                if isinstance(ref, MechanismRef)
            )
            targets[binding.site_name] = truths[mechanism_id]
    return targets


SCALAR_RECOVERY_TARGETS = _scalar_recovery_targets()

MEASUREMENT_MEAN_RECOVERY_TARGETS = {
    f"manifest_mean_{MANIFEST_NAMES[manifest_idx]}": {
        "site": "manifest_means_free",
        "index": idx,
        "true": TRUE_MANIFEST_MEANS[manifest_idx],
    }
    for idx, manifest_idx in enumerate(MEASUREMENT_MEANS_FREE_POSITIONS)
}

MEASUREMENT_LOADING_RECOVERY_TARGETS = {
    f"loading_{MANIFEST_NAMES[row]}_{LATENT_NAMES[col]}": {
        "site": "lambda_free",
        "index": idx,
        "true": TRUE_LOADINGS[row, col],
    }
    for idx, (row, col) in enumerate(MEASUREMENT_LOADINGS_FREE_POSITIONS)
}

RECOVERY_TARGETS = {
    **SCALAR_RECOVERY_TARGETS,
    **MEASUREMENT_MEAN_RECOVERY_TARGETS,
    **MEASUREMENT_LOADING_RECOVERY_TARGETS,
}
