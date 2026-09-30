"""Shared machinery for the blind case-study walkthroughs.

A case study authors only what its brief supports: the constructs, the causal DAG, one
indicator per observed construct, and each construct's self-relaxation timescale. This
module turns that into a scientific model, loads the observations into emission space,
applies the elicitation rules, and draws the findings of the production predictive battery
(:mod:`nof1_causal_lab.models.ssm.simulation_checks` and
:mod:`nof1_causal_lab.models.ssm.reachability`, which carry no plotting or notebook code).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
from model_mechanisms import declare_dynamics
from parameter_planning import complete_component_slots
from predictive_support import ConstructEditSpec, model_with_prior_payloads

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LikelihoodSpec, LinkFunction
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.model_parameters import referenced_parameter_ids
from nof1_causal_lab.models.ssm.simulation_checks import DesignInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from predictive_support import ConstructPredictiveReport

    from nof1_causal_lab.json_types import JsonObject

DATA_DIR = Path(__file__).parent / "data"
MODEL_CLOCK_DAYS = 1.0
# The library default for an off-diagonal diffusion coordinate.
COUPLING_PRIOR: JsonObject = {"distribution": "Normal", "params": {"mu": 0.0, "sigma": 0.5}}

# ---------------------------------------------------------------- the authored study


class Indicator(NamedTuple):
    """One observed channel: the construct it measures and its emission law."""

    name: str
    construct: str
    dtype: str  # measurement dtype: "continuous" or "count"
    family: str  # emission family: "gaussian", "beta", or "poisson"
    link: str  # emission link: "identity", "logit", or "log"


@dataclass(frozen=True)
class CaseStudy:
    """Everything a blind case study authors from its brief.

    Every edge contributes to state dynamics. Roots are exogenous; constructs without an indicator are
    unobserved, and the structural compiler marginalizes them.
    """

    directory: str
    constructs: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    indicators: tuple[Indicator, ...]
    timescales: Mapping[str, float]  # self-relaxation τ in days, per observed construct
    saturating: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    seed: int = 0

    def parents(self, construct: str) -> tuple[str, ...]:
        return tuple(cause for cause, effect in self.edges if effect == construct)

    @property
    def unobserved(self) -> frozenset[str]:
        return frozenset(self.constructs) - {item.construct for item in self.indicators}


def read_brief(case: CaseStudy) -> str:
    return (DATA_DIR / case.directory / "brief.md").read_text()


def scientific_model(case: CaseStudy) -> ModelSpec:
    """The brief's DAG as a scientific model with a 1-day measurement clock."""
    constructs = {
        name: {
            "id": f"construct:{name}",
            "name": name,
            "description": name,
            "role": "endogenous" if case.parents(name) else "exogenous",
            "temporal_status": "time_varying",
            "indicators": [
                {
                    "id": f"indicator:{indicator.name}",
                    "name": indicator.name,
                    "construct_polarity": "positive",
                    "measurement_dtype": indicator.dtype,
                    "aggregation": "last",
                }
                for indicator in case.indicators
                if indicator.construct == name
            ],
        }
        for name in case.constructs
    }
    return ModelSpec.model_validate(
        {
            "measurement_clock": "1d",
            "edges": [
                {
                    "id": f"edge:{cause}-{effect}",
                    "cause": constructs[cause],
                    "effect": constructs[effect],
                    "description": f"{cause} -> {effect}",
                }
                for cause, effect in case.edges
            ],
        }
    )


_WORD_BREAK = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")


def dag_figure(case: CaseStudy):
    """Constructs in columns by depth (the longest path from a root).

    Within a column, constructs are ordered by the mean height of their parents, which keeps
    most edges from crossing.
    """
    node_size = 3400  # marker area in pt², wide enough for the longest word at 6.5 pt
    node_radius = math.sqrt(node_size) / 2
    depth = {}
    for name in case.constructs:  # causal order: every parent is placed first
        parents = case.parents(name)
        depth[name] = 1 + max(depth[parent] for parent in parents) if parents else 0
    position = {}
    for level in range(max(depth.values()) + 1):
        names = [name for name in case.constructs if depth[name] == level]
        if level:
            names.sort(key=lambda name: np.mean([position[p][1] for p in case.parents(name)]))
        for row, name in enumerate(names):
            position[name] = (level * 1.9, (row - (len(names) - 1) / 2) * 1.7)

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for cause, effect in case.edges:
        saturating = (cause, effect) in case.saturating
        ax.annotate(
            "",
            xy=position[effect],
            xytext=position[cause],
            arrowprops=dict(
                arrowstyle="-|>",
                color="#e08a3c" if saturating else "#9a9a9a",
                lw=2.0 if saturating else 1.2,
                shrinkA=node_radius + 1,
                shrinkB=node_radius + 1,
                connectionstyle="arc3,rad=0.12",
            ),
        )
    for name, (x, y) in position.items():
        hidden = name in case.unobserved
        ax.scatter(
            [x],
            [y],
            s=node_size,
            facecolor="white" if hidden else "#3b6ea5",
            edgecolor="#c0504d" if hidden else "#3b6ea5",
            linewidth=2.2 if hidden else 1.5,
            zorder=3,
        )
        ax.text(
            x,
            y,
            _WORD_BREAK.sub("\n", name),
            ha="center",
            va="center",
            fontsize=6.5,
            color="#c0504d" if hidden else "white",
            fontweight="bold",
            zorder=4,
        )
    xs, ys = zip(*position.values(), strict=True)
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    legend = "hollow red = unobserved"
    if case.saturating:
        legend += " · orange = saturating (Hill) edge"
    ax.set_title(
        f"The posited DAG, columns by depth from a root ({legend})",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    return _as_html(fig)


# ---------------------------------------------------------------- observations


@dataclass(frozen=True)
class Observations:
    """Observation times (days) and each indicator's values in emission space."""

    times: np.ndarray
    values: dict[str, np.ndarray]


def load_observations(case: CaseStudy) -> Observations:
    """Bounded 0–100 indices become fractions in (0, 1), the support of their Beta law."""
    raw = np.genfromtxt(DATA_DIR / case.directory / "observations.csv", delimiter=",", names=True)
    values = {}
    for indicator in case.indicators:
        column = np.asarray(raw[indicator.name], dtype=float)
        values[indicator.name] = (
            np.clip(column / 100.0, 1e-3, 1 - 1e-3) if indicator.link == "logit" else column
        )
    return Observations(np.asarray(raw["t"], dtype=float), values)


def observation_summary(case: CaseStudy, observations: Observations):
    """Legitimate summaries: design, then location, spread and serial dependence per channel."""

    def lag1_rank_correlation(values):
        ranks = np.argsort(np.argsort(values))
        return float(np.corrcoef(ranks[:-1], ranks[1:])[0, 1])

    times = observations.times
    gap = float(np.median(np.diff(times)))
    span = float(np.ptp(times))
    rows = []
    for indicator in case.indicators:
        values = observations.values[indicator.name]
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        rows.append(
            f"| `{indicator.name}` | {indicator.family}/{indicator.link} | {values.mean():.2f} | "
            f"{values.std():.2f} | {q25:.2f} / {q50:.2f} / {q75:.2f} | "
            f"{lag1_rank_correlation(values):+.2f} |"
        )
    return mo.md(
        f"**{times.size} observations** over {span:.0f} days, median gap **{gap:.2f} d**, so "
        f"the design can resolve self-relaxation times of roughly "
        f"`[gap/3, span/4] = [{gap / 3:.2f}, {span / 4:.1f}] d`.\n\n"
        "| indicator | family/link | mean | sd | q25 / q50 / q75 | lag-1 rank correlation |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(rows)
    )


def design(case: CaseStudy, model: ModelSpec, observations: Observations) -> DesignInfo:
    """Evaluate the prior predictive exactly at the observation times."""
    index = np.arange(observations.times.size)
    return DesignInfo(
        t_grid=jnp.asarray(observations.times),
        obs_index_by_indicator={f"indicator:{item.name}": index for item in case.indicators},
        values_by_indicator={
            f"indicator:{item.name}": observations.values[item.name] for item in case.indicators
        },
        manifest_ids=tuple(model.manifest_indicator_order),
        n_draws=64,
        seed=case.seed,
    )


# ---------------------------------------------------------------- elicitation


def _normal(mu, sigma) -> JsonObject:
    return {"distribution": "Normal", "params": {"mu": mu, "sigma": sigma}}


def _lognormal(mu, sigma) -> JsonObject:
    return {"distribution": "LogNormal", "params": {"mu": mu, "sigma": sigma}}


def _halfnormal(sigma) -> JsonObject:
    return {"distribution": "HalfNormal", "params": {"sigma": sigma}}


def _unit_interval_normal(mu, sigma) -> JsonObject:
    return {
        "distribution": "TruncatedNormal",
        "params": {"mu": mu, "sigma": sigma, "lower": 0.0, "upper": 1.0},
    }


def _inverse_link(link, values):
    values = np.asarray(values, dtype=float)
    if link == "identity":
        return values
    if link == "logit":
        clipped = np.clip(values, 1e-3, 1 - 1e-3)
        return np.log(clipped / (1 - clipped))
    return np.log(np.maximum(values, 0.5))  # log link: counts, floored at half an event


def _scale_anchor(indicator, values):
    """Data-implied latent scale: the inverse-link IQR / 1.349 (unit reference loading)."""
    q75, q25 = _inverse_link(indicator.link, np.percentile(values, [75, 25]))
    return abs(float(q75 - q25)) / 1.349


def _intercept_prior(indicator, values) -> JsonObject:
    """The observation intercept, centred on the data mapped through the inverse link."""
    if indicator.link == "identity":
        return _normal(float(np.mean(values)), 0.3 * float(np.std(values)))
    if indicator.link == "logit":
        median = float(np.clip(np.median(values), 0.02, 0.98))
        return _normal(math.log(median / (1 - median)), 0.4)
    return _normal(math.log(max(float(np.median(values)), 0.5)), 0.4)


def elicit(
    case: CaseStudy, model: ModelSpec, observations: Observations, *, edge_base: float = 0.45
) -> list[ConstructEditSpec]:
    """Elicit every retained construct's priors from the brief's timescales and the data.

    ``edge_base`` is how far a parent may displace its child, in units of the child's own
    scale. Each construct is completed and elicited on its own copy of the template.
    """
    template = declare_dynamics(
        model,
        hill_edges=[
            edge.id
            for edge in model.edges
            if (edge.cause.name, edge.effect.name) in case.saturating
        ],
    )
    retained = {model.get_construct(identity).name for identity in model.state_order}
    anchors = {
        item.construct: _scale_anchor(item, observations.values[item.name])
        for item in case.indicators
    }

    def elicit_construct(name):
        tau = case.timescales[name]
        relaxation = MODEL_CLOCK_DAYS / tau  # the child's relaxation rate per clock step
        persistence = math.exp(-relaxation)
        anchor = anchors[name]
        priors: dict[str, JsonObject] = {
            f"rho_{name}": _unit_interval_normal(persistence, 0.35 * relaxation * persistence),
            f"sigma_{name}": _lognormal(math.log(anchor * math.sqrt(2.0 / tau)), 0.4),
        }
        linear_parents, hill_parents = [], []
        # A marginalized parent acts through the innovation coupling, not through an edge.
        for parent in (item for item in case.parents(name) if item in retained):
            if (parent, name) in case.saturating:
                hill_parents.append(parent)
                # A Hill drift saturates at emax and settles to an offset τ·emax, so emax
                # scales like β below. EC50 lives in the parent's latent units and is positive:
                # a LogNormal whose median is the parent's own scale anchor.
                priors[f"hill_emax_{parent}_{name}"] = _halfnormal(edge_base * relaxation * anchor)
                priors[f"hill_ec50_{parent}_{name}"] = _lognormal(
                    math.log(max(anchors[parent], 0.5)), 0.4
                )
                priors[f"hill_n_{parent}_{name}"] = _lognormal(math.log(2.0), 0.3)
            else:
                linear_parents.append(parent)
                # A steady drift β·anchor_parent settles to an offset τ·β·anchor_parent.
                # Scaling β by the relaxation rate keeps that offset at edge_base·anchor
                # whatever the child's timescale; otherwise a slow child integrates a modest
                # edge into an overwhelming offset (C4b) that also inflates its predictive
                # width through the link (C5b).
                priors[f"beta_{parent}_{name}"] = _normal(
                    0.0, edge_base * relaxation * anchor / max(anchors[parent], 0.25)
                )
        indicator = next(item for item in case.indicators if item.construct == name)
        priors[f"manifest_mean_{indicator.name}"] = _intercept_prior(
            indicator, observations.values[indicator.name]
        )

        construct = next(item for item in template.constructs if item.name == name)
        likelihood = LikelihoodSpec(
            law=observation_law(
                construct.id, DistributionFamily(indicator.family), LinkFunction(indicator.link)
            ),
            reasoning=f"{indicator.family}/{indicator.link} for {indicator.name}",
        )
        construct = construct.model_copy(
            update={
                "indicators": tuple(
                    item.model_copy(update={"likelihood": likelihood})
                    for item in construct.indicators
                )
            }
        )
        proposal = complete_component_slots(
            template.revised(
                edges=replace_constructs(
                    template.edges,
                    tuple(
                        construct if item.id == construct.id else item
                        for item in template.constructs
                    ),
                )
            )
        )
        couplings = {
            operand.value
            for operand in proposal.get_construct(construct.id).coefficients
            if operand.role == "diffusion_loading"
        }
        payloads: dict[str, JsonObject] = {
            parameter.id: {
                **(COUPLING_PRIOR if parameter.id in couplings else priors[parameter.name]),
                "reference_interval_days": MODEL_CLOCK_DAYS,
            }
            for parameter in proposal.parameters
            if parameter.id in couplings or parameter.name in priors
        }
        proposal = model_with_prior_payloads(proposal, payloads)
        construct = proposal.get_construct(construct.id)
        edges = tuple(edge for edge in proposal.edges if edge.effect.id == construct.id)
        referenced = referenced_parameter_ids(construct, *edges)
        parameters = tuple(item for item in proposal.parameters if item.id in referenced)
        return ConstructEditSpec(
            construct=construct,
            edges=edges,
            parameters=parameters,
            distributions={
                item.distribution: proposal.distributions[item.distribution]
                for item in parameters
                if item.distribution is not None
            },
            edge_parents=(*linear_parents, *hill_parents),
            hill_parents=tuple(hill_parents),
        )

    return [elicit_construct(name) for name in case.constructs if name in retained]


# ---------------------------------------------------------------- evidence figures


def _as_html(fig):
    html = mo.as_html(fig)
    plt.close(fig)
    return html


def _viz_confinement(ev):
    x, growth, t = ev["x"], ev["growth"], ev["times"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 3.0))
    for row in x[:25]:
        ax0.plot(t, row, color="#c5c5c5", lw=0.6)
    for i in np.argsort(np.nan_to_num(growth, nan=np.inf))[-5:]:
        ax0.plot(t, x[i], color="#c0504d", lw=1.2)
    finite = x[np.isfinite(x)]
    if finite.size:
        lo_y, hi_y = np.percentile(finite, [0.1, 99.9])
        pad = 0.25 * (hi_y - lo_y + 1e-9)
        ax0.set_ylim(lo_y - pad, hi_y + pad)
    ax0.set_title("prior draws (gray) vs the 5 highest-growth draws (red)", fontsize=9)
    ax0.set_xlabel("day")
    finite_g = growth[np.isfinite(growth)]
    ax1.hist(np.clip(finite_g, 0, 20), bins=40, color="#3b6ea5")
    ax1.axvline(ev["growth_ratio"], color="#c0504d", ls="--", label="growth criterion")
    ax1.set_title("late/early amplitude ratio per draw", fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_scale(ev):
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ax.hist(ev["marginal_scales"], bins=40, color="#3b6ea5")
    ax.axvline(ev["lo"], color="#c0504d", ls="--", label="band")
    ax.axvline(ev["hi"], color="#c0504d", ls="--")
    ax.axvline(ev["anchor"], color="#4a9d5b", lw=2, label="anchor")
    ax.set_title("Across-draw marginal scale vs the scale-anchor band", fontsize=9)
    ax.set_xlabel("Marginal scale at each late-window time")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_resolvability(ev):
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    if "observation_times" in ev:
        ax.scatter(ev["observation_times"], np.zeros_like(ev["observation_times"]))
        ax.set_title("Too few distinct times for a temporal contrast")
        return fig
    tau = ev["tau"]
    hi_x = float(np.percentile(tau, 99))
    ax.hist(np.clip(tau, 0, hi_x), bins=50, color="#3b6ea5", label="sampled τ = 1/decay")
    # Half the actual gaps must be no greater than 3τ; retain their discrete order statistic.
    gaps = np.sort(ev["gaps"])
    floor = float(gaps[int(np.ceil(len(gaps) / 2)) - 1]) / 3.0
    ceiling = ev["span"] / 4.0
    if floor < ceiling:
        ax.axvspan(
            floor, min(ceiling, hi_x), color="#4a9d5b", alpha=0.12, label="resolvable window"
        )
    ax.axvline(floor, color="#c0504d", ls="--", label="actual-gap floor")
    ax.axvline(ceiling, color="#7d6bb0", ls=":", label="span/4 ceiling")
    ax.set_xlabel("self-relaxation τ (days)")
    ax.set_title("prior timescale vs the design's resolvable window", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_edge(ev):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 3.0))
    idx = np.arange(ev["on"].size)
    ax0.plot(idx, ev["on"], color="#3b6ea5", lw=1.4, label="edge on")
    ax0.plot(idx, ev["off"], color="#e08a3c", lw=1.4, ls="--", label="edge off (same noise)")
    ax0.set_xlabel("observation #")
    ax0.set_title("high-displacement draw: how much the edges move the child", fontsize=9)
    ax0.legend(frameon=False, fontsize=8)
    hi = float(np.percentile(ev["e"], 99))
    ax1.hist(np.clip(ev["e"], 0, hi), bins=40, color="#3b6ea5")
    ax1.axvline(0.95, color="#7d6bb0", ls=":", label="overwhelm cap")
    ax1.set_title("per-draw displacement / child scale", fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_saturation(ev):
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.hist(ev["bend_mass"], bins=np.linspace(0, 1, 26).tolist(), color="#3b6ea5")
    ax.axvline(0.1, color="#c0504d", ls="--", label="minimum schedule share on bend")
    ax.set_xlabel("Fraction of each draw's schedule exercising the Hill bend")
    ax.set_title("Draw-paired Hill activation", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_coverage(ev):
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.hist(ev["pp"], bins=60, density=True, color="#c5c5c5", label="prior predictive (pooled)")
    ax.hist(ev["y_obs"], bins=20, density=True, color="#3b6ea5", alpha=0.6, label="observed")
    ax.set_title("prior predictive vs observed — location and width", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_transmission(ev):
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    if "undefined_moment_fraction" in ev:
        ax.bar(
            ["undefined conditional variance"], [ev["undefined_moment_fraction"]], color="#c0504d"
        )
        ax.set_ylim(0, 1)
        return fig
    ax.hist(
        ev["signal_fraction"],
        bins=np.linspace(0.0, 1.0, 51).tolist(),
        color="#4a9d5b",
        alpha=0.75,
        label="prior-draw signal share",
    )
    minimum = ev["min_signal_fraction"]
    ax.axvline(minimum, color="#c0504d", ls="--", label=f"minimum {minimum:.0%}")
    ax.set(
        xlabel="temporal signal variance / total predictive variance",
        ylabel="prior draws",
        title="latent signal share of predictive variation",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


CHECK_VIZ = {
    "C1a finiteness": _viz_confinement,
    "C1b confinement": _viz_confinement,
    "C2 latent scale": _viz_scale,
    "C3 resolvability": _viz_resolvability,
    "C4b edge overwhelm": _viz_edge,
    "C4c saturation": _viz_saturation,
    "C5a location reach": _viz_coverage,
    "C5b width": _viz_coverage,
    "C5c transmission": _viz_transmission,
}

# ---------------------------------------------------------------- report rendering


def _verdict(passed):
    return "not evaluated" if passed is None else "passed" if passed else "failed"


def render_report(title, report: ConstructPredictiveReport):
    """One construct's measurements, with an evidence figure per failed check family."""
    rows = "\n".join(
        f"| {result.check} | {result.target} | {result.value} | {result.band} | "
        f"{_verdict(result.passed)} |"
        for result in report.results
    )
    findings = [result for result in report.results if result.passed is not True]
    notes = "\n".join(f"- **{result.check}** — {result.note}" for result in findings)
    md = mo.md(
        f"### {title}\n\n| check | target | predictive value | band | finding |\n"
        "|---|---|---|---|---|\n" + rows + "\n\n" + notes
    )
    figures = []
    drawn = set()
    for result in findings:
        draw = CHECK_VIZ.get(result.check)
        if draw is not None and draw not in drawn and result.evidence is not None:
            figures.append(_as_html(draw(result.evidence)))
            drawn.add(draw)
    return mo.vstack([md, *figures])


def render_reports(
    case: CaseStudy,
    reports: Mapping[str, ConstructPredictiveReport],
    descriptions: Mapping[str, str],
):
    """Every construct in causal order; a marginalized construct gets a note, not a report."""
    blocks = []
    for name in case.constructs:
        title = f"{name} — {descriptions[name]}"
        if name in reports:
            blocks.append(render_report(title, reports[name]))
        else:
            blocks.append(
                mo.md(
                    f"### {title}\n\nThe structural compiler marginalizes this unobserved root "
                    "instead of keeping an unanchored latent state. Its shared-child dependence "
                    "lives in the compiled innovation coupling, so it has no report of its own."
                )
            )
    return mo.vstack(blocks)


def outcome_summary(
    case: CaseStudy, checked_model: ModelSpec, reports: Mapping[str, ConstructPredictiveReport]
):
    """The board, read straight off the reports, and the totals behind it."""
    rows = []
    for name in case.constructs:
        if name not in reports:
            rows.append(f"| {name} | — | — | marginalized |")
            continue
        results = reports[name].results
        resolvability = next(result for result in results if result.check == "C3 resolvability")
        failed = [result.check for result in results if result.passed is False]
        rows.append(
            f"| {name} | {resolvability.value.split(';')[0]} | {len(results)} | "
            f"{', '.join(failed) or 'none'} |"
        )
    outcomes = [result.passed for report in reports.values() for result in report.results]
    unevaluated = outcomes.count(None)
    return mo.md(
        "| construct | prior τ (C3) | checks | failed |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + f"\n\nThe checked model keeps {len(checked_model.state_order)} of "
        f"{len(case.constructs)} constructs as latent states. On its shared predictive batch, "
        f"**{outcomes.count(False)} of {len(outcomes)} checks failed**"
        + (f" and {unevaluated} were not evaluated" if unevaluated else "")
        + ". Findings guide the next revision; they certify neither recovery of the hidden "
        "truth nor any causal claim."
    )
