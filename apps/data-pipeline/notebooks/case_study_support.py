"""Rendering support for the blind case-study walkthroughs.

The case studies drive the *production* full-model checking engine
(:mod:`nof1_causal_lab.models.ssm.simulation_checks`) and its reachability
battery (:mod:`nof1_causal_lab.models.ssm.reachability`). Those modules are
deliberately free of any plotting or notebook dependency, so the notebook-facing
presentation lives here: a table of saved scientific findings and an evidence
figure per failed check family. Figures consume the current reducer evidence;
there are no acceptance modes or authoring decisions in this renderer.
"""

from __future__ import annotations

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- evidence figures


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

_PATTERN_HINTS = (
    (
        {"C2 latent scale", "C5c transmission"},
        "shared input — both depend on the latent scale and where its mass lands on the link. "
        "C5c additionally depends on conditional observation variance, so their joint failure "
        "calls for inspecting the scale anchor, loading/link geometry, and noise prior rather "
        "than diagnosing saturation alone.",
    ),
)

# ---------------------------------------------------------------- report rendering


def render_report(title, report):
    """Display measurements from the shared full-model batch."""
    rows = "\n".join(
        f"| {result.check} | {result.target} | {result.value} | {result.band} | "
        f"{'not evaluated' if result.passed is None else 'passed' if result.passed else 'failed'} |"
        for result in report.results
    )
    findings = [result for result in report.results if result.passed is not True]
    notes = "\n".join(f"- **{result.check}** — {result.note}" for result in findings)
    md = mo.md(
        f"### {title}\n\n| check | target | predictive value | band | finding |\n"
        "|---|---|---|---|---|\n" + rows + "\n\n" + notes
    )
    figures = []
    seen = set()
    for result in findings:
        fn = CHECK_VIZ.get(result.check)
        if fn is not None and id(fn) not in seen and result.evidence is not None:
            figures.append(mo.as_html(fn(result.evidence)))
            seen.add(id(fn))
    return mo.vstack([md, *figures])
