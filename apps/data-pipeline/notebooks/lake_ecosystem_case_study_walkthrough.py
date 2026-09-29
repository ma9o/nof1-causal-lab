import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # A second blind case study: a lake ecosystem through the predictive battery

    This notebook runs the protocol of `d10_case_study_walkthrough.py` on a fresh **blind**
    problem in a different domain. It exercises two features the D = 10 study does not:
    **saturating (Hill) edges**, and **timescales that span hours to weeks**. Like the D = 10
    study, it drives the production engine directly
    (`nof1_causal_lab.models.ssm.simulation_checks` and
    `nof1_causal_lab.models.ssm.reachability`), so every verdict is the shared predictive
    battery's.

    **What the battery measures before fitting.** Reachability, plus one design-observability
    screen:

    - **C1** finiteness and confinement, **C2** latent scale, **C5a/b/c** location, width and
      transmission — can the prior generate and cover this data at all?
    - **C3 design resolvability** — does the prior's self-relaxation τ fall inside the window
      this sampling design can resolve, roughly `[gap/3, span/4]`? It reads only the
      observation times and the prior; estimating τ from the data would confound a construct's
      own relaxation with persistence inherited through its edges, a split left to the fit.
    - **C4b edge overwhelm** (is a child slaved to a parent?) and **C4c saturation** (is a
      Hill edge's bend exercised over the parent's realized range, or is it a dead linear arm
      or a flat saturated response?).

    Whether the data will *pin* a parameter is deliberately not judged here; practical
    identifiability belongs after the fit (posterior contraction, power scaling).

    **The blind protocol.** A separate agent designed a hidden continuous-time **nonlinear,
    non-Gaussian** ground truth in a domain it chose, generated the data, and wrote the brief.
    Everything below is built from the brief and legitimate summaries of the observed data
    only; the generator under `data/lake_ecosystem_case_study/hidden/` is never opened.

    The shared case-study machinery lives in `case_study_support.py`, and the whole notebook
    runs in about a minute.
    """)
    return


@app.cell
def imports():
    import case_study_support as cs
    from predictive_support import evaluate_case_study

    return cs, evaluate_case_study


@app.cell(hide_code=True)
def brief_md(mo):
    mo.md(r"""
    ## 1. The brief

    The verbatim domain briefing: one lake, nine constructs, directions but not magnitudes of
    the causal edges, two edges flagged as saturating, and the irregular sampling design.
    """)
    return


@app.cell
def brief_text(CASE, cs, mo):
    mo.accordion({"brief.md": mo.md(cs.read_brief(CASE))})
    return


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## 2. The posited model

    Nine constructs and thirteen lagged edges, as the brief draws them. `CatchmentLoading` is
    the unmeasured storm-driven confounder of nitrate, turbidity, and dissolved color; the
    structural compiler marginalizes it. The two edges the brief expects to saturate — the
    light-limitation ceiling Turbidity → Phytoplankton and the grazer satiation ceiling
    Phytoplankton → Zooplankton — are authored as Hill terms. Oxygen saturation is a bounded
    0–100 index (Beta/logit on the fraction), the zooplankton tow is a count (Poisson/log),
    and the rest are Gaussian/identity. The timescales τ (days) follow the brief's physical
    account: hours for turbidity and oxygen, about a day for temperature and pH, days for
    nutrients, color, and algae, and about a week for the grazers.
    """)
    return


@app.cell
def case_spec(cs):
    CASE = cs.CaseStudy(
        directory="lake_ecosystem_case_study",
        constructs=(
            "CatchmentLoading",
            "WaterTemperature",
            "Nitrate",
            "Turbidity",
            "CDOM",
            "Phytoplankton",
            "DissolvedOxygen",
            "pH",
            "Zooplankton",
        ),
        edges=(
            ("CatchmentLoading", "Nitrate"),
            ("CatchmentLoading", "Turbidity"),
            ("CatchmentLoading", "CDOM"),
            ("WaterTemperature", "Nitrate"),
            ("Nitrate", "Phytoplankton"),
            ("WaterTemperature", "Phytoplankton"),
            ("Turbidity", "Phytoplankton"),
            ("CDOM", "Phytoplankton"),
            ("Phytoplankton", "DissolvedOxygen"),
            ("WaterTemperature", "DissolvedOxygen"),
            ("Phytoplankton", "pH"),
            ("Phytoplankton", "Zooplankton"),
            ("WaterTemperature", "Zooplankton"),
        ),
        indicators=(
            cs.Indicator("water_temp_C", "WaterTemperature", "continuous", "gaussian", "identity"),
            cs.Indicator("nitrate_mgL", "Nitrate", "continuous", "gaussian", "identity"),
            cs.Indicator("turbidity_NTU", "Turbidity", "continuous", "gaussian", "identity"),
            cs.Indicator("fdom_QSU", "CDOM", "continuous", "gaussian", "identity"),
            cs.Indicator("chl_a_ugL", "Phytoplankton", "continuous", "gaussian", "identity"),
            cs.Indicator("do_sat_pct", "DissolvedOxygen", "continuous", "beta", "logit"),
            cs.Indicator("ph", "pH", "continuous", "gaussian", "identity"),
            cs.Indicator("zoop_count", "Zooplankton", "count", "poisson", "log"),
        ),
        timescales={
            "WaterTemperature": 1.2,
            "Nitrate": 3.5,
            "Turbidity": 0.15,
            "CDOM": 6.0,
            "Phytoplankton": 4.5,
            "DissolvedOxygen": 0.25,
            "pH": 0.5,
            "Zooplankton": 8.0,
        },
        saturating=frozenset({("Turbidity", "Phytoplankton"), ("Phytoplankton", "Zooplankton")}),
        seed=7,
    )
    MODEL = cs.scientific_model(CASE)
    return CASE, MODEL


@app.cell
def dag_fig(CASE, cs):
    cs.dag_figure(CASE)
    return


@app.cell(hide_code=True)
def data_md(mo):
    mo.md(r"""
    ## 3. The observed data, in emission space

    One station, visited irregularly. The oxygen-saturation index is divided by 100 so its
    values live on the Beta likelihood's (0, 1) support. Compare the resolvable window in the
    first line with the timescales above: the design is expected to struggle with the fastest
    constructs, which is what C3 measures.
    """)
    return


@app.cell
def observed_data(CASE, cs):
    observations = cs.load_observations(CASE)
    cs.observation_summary(CASE, observations)
    return (observations,)


@app.cell(hide_code=True)
def elicitation_md(mo):
    mo.md(r"""
    ## 4. Elicitation strategy

    The D = 10 study's rules, with no hidden value consulted:

    - **Persistence from the brief's timescales.** τ sets each `rho` prior on the discrete-time
      persistence scale, centred on `exp(-Δt/τ)`, wide, and truncated to the unit interval.
      **C3** then checks τ against the sampling design rather than trying to pin it from data.
    - **Standardized latents via the diffusion** (`sigma`), targeting a stationary latent sd
      near the indicator's inverse-link scale anchor (the reference indicator carries unit
      loading).
    - **Location** from the data mapped through the inverse link (`manifest_mean`).
    - **Linear edges** `Normal(0, s)`, with `s` tied to the child's relaxation rate; **C4b**
      guards against a parent swamping its child.
    - **Saturating edges as Hill terms** (`hill_emax`, `hill_ec50`, `hill_n`). `emax` is scaled
      like a linear edge, and the EC50 prior is a LogNormal centred on the parent's latent
      scale anchor, so that **C4c** can confirm the bend is exercised where the parent lives.
    - **Residual coupling.** Marginalizing `CatchmentLoading` correlates the innovations of
      its three children; each pair's `cor_<a>_<b>` diffusion loading gets `Normal(0, 0.5)`,
      the library default for an off-diagonal diffusion coordinate.
    """)
    return


@app.cell
def elicitation(CASE, MODEL, cs, observations):
    edits = cs.elicit(CASE, MODEL, observations)
    return (edits,)


@app.cell(hide_code=True)
def checks_md(mo):
    mo.md(r"""
    ## 5. Whole-model predictive checks

    One exact predictive batch for the whole candidate model, plus one edge-knockout batch per
    checked edge, measured construct by construct.
    """)
    return


@app.cell
def run_checks(CASE, MODEL, cs, edits, evaluate_case_study, observations):
    checked_model, reports = evaluate_case_study(MODEL, edits, cs.design(CASE, MODEL, observations))
    return checked_model, reports


@app.cell
def construct_reports(CASE, cs, reports):
    cs.render_reports(
        CASE,
        reports,
        {
            "CatchmentLoading": "latent storm-driven confounder",
            "WaterTemperature": "physical root (τ ≈ a day)",
            "Nitrate": "loading and temperature drivers",
            "Turbidity": "sub-cadence settling (C3 reads as a design limit)",
            "CDOM": "colored dissolved organics (multi-day)",
            "Phytoplankton": "four drivers, including saturating Turbidity (Hill, C4c)",
            "DissolvedOxygen": "bounded index (Beta/logit), biology vs solubility",
            "pH": "tracks biology within a day",
            "Zooplankton": "weekly grazer lag, saturating grazing (Hill, C4c), Poisson counts",
        },
    )
    return


@app.cell(hide_code=True)
def outcome_md(mo):
    mo.md(r"""
    ## 6. Outcome

    Read straight off the report objects above. The C3 column shows the gradient of timescales
    that the design can and cannot resolve.
    """)
    return


@app.cell
def outcome(CASE, checked_model, cs, reports):
    cs.outcome_summary(CASE, checked_model, reports)
    return


if __name__ == "__main__":
    app.run()
