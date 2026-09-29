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
    # A blind D = 10 case study through the predictive battery

    A separate agent designed a hidden ten-construct ground truth — continuous-time,
    **nonlinear and non-Gaussian**, a single-subject behavioral and physiological story —
    generated 120 days of irregular observations from it, and wrote a study brief. This
    notebook works on the other side of that firewall. It posits a model and elicits its priors
    from the brief and from *legitimate summaries of the observed data only*, then runs the
    production predictive checks on the whole model. The generator and its parameters under
    `data/d10_case_study/hidden/` are never opened, so the priors are a genuine blind
    elicitation, not reverse-engineered from the answer.

    **How the checks run.** Every authored construct and mechanism is assembled into one model
    before a single exact predictive batch: the production generator draws the model's own
    prior laws, integrates the nonlinear dynamics with Diffrax, and applies the declared
    emission densities. The C1–C5 reducers that the scientific actions use then measure the
    simulated paths, and each checked edge adds one paired edge-knockout batch.
    `reachability_checks_walkthrough.py` shows each check failing in isolation.

    **Reading the findings.** They screen the prior design against the observed data. Passing
    them does not establish recovery of the hidden truth, practical parameter identification,
    or causal identification; those need separate evidence. Failed checks stay attached to the
    checked model and guide its next revision.

    The case-study machinery — model assembly, elicitation rules, evidence figures — lives in
    `case_study_support.py`, and the whole notebook runs in about a minute.
    `lake_ecosystem_case_study_walkthrough.py` runs the same protocol on a second blind
    problem.
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

    The verbatim study brief is everything the modeler may read: the constructs, the causal
    DAG, the indicator families, and the observation design. It deliberately gives **no**
    parameter values, scales, or timescales, and no hints about where the nonlinearities live.
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

    The brief's DAG, taken as given: ten constructs in causal order and fifteen lagged edges.
    Each observed construct has one indicator whose emission law follows from its response
    scale — the 0–100 sliders as Beta/logit on the fraction, the daily counts as Poisson/log,
    everything else Gaussian/identity. `AutonomicArousal` has no indicator: it is the
    unobserved common cause of stress, sleep, and pain, and the structural compiler
    marginalizes it. The self-relaxation timescales τ (days) come from what the brief says
    each construct is, not from the data.
    """)
    return


@app.cell
def case_spec(cs):
    CASE = cs.CaseStudy(
        directory="d10_case_study",
        constructs=(
            "CaffeineIntake",
            "AutonomicArousal",
            "PerceivedStress",
            "SleepQuality",
            "Fatigue",
            "MusculoskeletalPain",
            "PhysicalActivity",
            "NegativeMood",
            "CognitiveFocus",
            "SocialEngagement",
        ),
        edges=(
            ("CaffeineIntake", "SleepQuality"),
            ("AutonomicArousal", "PerceivedStress"),
            ("AutonomicArousal", "SleepQuality"),
            ("AutonomicArousal", "MusculoskeletalPain"),
            ("PerceivedStress", "SleepQuality"),
            ("PerceivedStress", "NegativeMood"),
            ("PerceivedStress", "Fatigue"),
            ("SleepQuality", "Fatigue"),
            ("Fatigue", "MusculoskeletalPain"),
            ("Fatigue", "PhysicalActivity"),
            ("Fatigue", "CognitiveFocus"),
            ("MusculoskeletalPain", "PhysicalActivity"),
            ("PhysicalActivity", "NegativeMood"),
            ("NegativeMood", "SocialEngagement"),
            ("NegativeMood", "CognitiveFocus"),
        ),
        indicators=(
            cs.Indicator("caffeine_servings", "CaffeineIntake", "count", "poisson", "log"),
            cs.Indicator("stress_vas", "PerceivedStress", "continuous", "beta", "logit"),
            cs.Indicator("sleep_quality_vas", "SleepQuality", "continuous", "beta", "logit"),
            cs.Indicator("fatigue_score", "Fatigue", "continuous", "gaussian", "identity"),
            cs.Indicator("pain_nrs", "MusculoskeletalPain", "continuous", "gaussian", "identity"),
            cs.Indicator(
                "active_minutes", "PhysicalActivity", "continuous", "gaussian", "identity"
            ),
            cs.Indicator(
                "irritability_index", "NegativeMood", "continuous", "gaussian", "identity"
            ),
            cs.Indicator(
                "reaction_time_ms", "CognitiveFocus", "continuous", "gaussian", "identity"
            ),
            cs.Indicator("social_contacts", "SocialEngagement", "count", "poisson", "log"),
        ),
        timescales={
            "CaffeineIntake": 0.7,
            "PerceivedStress": 3.3,
            "SleepQuality": 1.4,
            "Fatigue": 2.8,
            "MusculoskeletalPain": 2.5,
            "PhysicalActivity": 1.2,
            "NegativeMood": 2.5,
            "CognitiveFocus": 2.5,
            "SocialEngagement": 1.8,
        },
        seed=20260705,
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

    The battery compares the prior predictive with the observed indicators in each emission's
    own space. Gaussian channels stay as recorded and counts stay counts, but the two 0–100
    sliders are divided by 100 first: the fraction in (0, 1) is what the Beta likelihood and
    every slider check see. The lag-1 rank correlation is shown for context only; the
    elicitation below does not read timescales off it.
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

    Four rules turn the brief and the summaries into priors, keyed by the parameter names the
    authoring defaults create (`rho_<c>`, `sigma_<c>`, `manifest_mean_<ind>`, `beta_<p>_<c>`),
    with **no** reference to any hidden value:

    - **Persistence from the timescale.** Each construct's τ sets its `rho` prior on the
      discrete-time persistence scale at the 1-day model clock: centred on `exp(-Δt/τ)` with a
      standard deviation of `0.35·Δt/τ` times that centre, truncated to the unit interval. The
      compiler maps persistence to the continuous-time decay. τ is **not** read off indicator
      autocorrelation: a downstream indicator's serial dependence mixes the construct's own
      relaxation with persistence inherited from its parents, a split only the fit can make.
    - **Standardized latents via the diffusion.** `sigma` puts the stationary sd of the latent
      process near the construct's data-implied scale anchor, the indicator's inverse-link IQR
      / 1.349. The reference indicator carries unit loading, so the loading carries the
      physical scale; C2 checks this convention.
    - **Location from the data.** The observation intercept `manifest_mean` is the data's
      centre mapped through the inverse link: the mean for Gaussian channels, the logit of the
      median fraction for sliders, and the log of the median rate for counts.
    - **Edges scaled to the child.** Each edge prior is `Normal(0, s)`, with `s` tied to the
      child's relaxation rate and rescaled by the parent/child anchor ratio so the edge acts on
      standardized latents. A slow child integrates its parents' input over a long memory, so
      it needs a tighter edge prior to stay self-driven; C4b checks this.

    Marginalizing `AutonomicArousal` leaves correlated innovations among its three children,
    one `cor_<a>_<b>` diffusion loading per pair. Each gets `Normal(0, 0.5)`, the library
    default for an off-diagonal diffusion coordinate. The confounder's own edges have no
    mechanism, so they need no prior.
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

    The per-construct definitions are merged into one candidate model, simulated once at the
    observation times (64 prior draws), and measured construct by construct. Each block below
    lists every check with its measured value and band, the note for any check that did not
    pass, and an evidence figure per failed check family.
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
            "CaffeineIntake": "root, Poisson count indicator",
            "AutonomicArousal": "unobserved confounder",
            "PerceivedStress": "slider (Beta/logit), driven only by the confounder",
            "SleepQuality": "slider (Beta/logit), three parents",
            "Fatigue": "continuous indicator, two parents",
            "MusculoskeletalPain": "continuous indicator",
            "PhysicalActivity": "continuous indicator (tens scale)",
            "NegativeMood": "continuous indicator (near zero)",
            "CognitiveFocus": "continuous indicator (reaction time, ms)",
            "SocialEngagement": "Poisson count indicator",
        },
    )
    return


@app.cell(hide_code=True)
def outcome_md(mo):
    mo.md(r"""
    ## 6. Outcome

    The board is read straight off the report objects above, so every verdict is the
    production battery's own.
    """)
    return


@app.cell
def outcome(CASE, checked_model, cs, reports):
    cs.outcome_summary(CASE, checked_model, reports)
    return


if __name__ == "__main__":
    app.run()
