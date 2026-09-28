# Assumptions and limits

These are the scientific commitments behind every result: what the model can express, what identification assumes, and how far each output can be read. The [action charts](../README.md#documentation) show where each commitment is checked. [Likelihoods](reference/statistical-model-spec/likelihoods.md) and [parameters](reference/statistical-model-spec/parameters.md) list the supported laws.

## Model class

- Latent constructs evolve as a continuous-time stochastic differential equation. The drift sums expressions over states and coefficients. These include per-construct dynamics such as decay and restoring potentials, and per-edge mechanisms such as linear and saturating (Hill) effects. Products express moderation, so the drift is nonlinear in general. Diffusion is additive, and fitting and simulation support only Gaussian diffusion.
- Indicators are observed at discrete, possibly irregular times through indicator-specific [observation laws](reference/statistical-model-spec/likelihoods.md).
- `fit` targets this model with Euler–Maruyama transitions between consecutive prepared time points, using the true nonlinear drift and the true emission density. Linearization and Gaussian approximations only initialize the sampler, so the remaining approximation is time discretization.
- `simulate` and the automatic predictive checks integrate the same equation with a Diffrax solver. Fitted and simulated histories therefore carry different discretization error.
- Missing observations are masked out of the likelihood, which assumes the missing-data mechanism is ignorable.
- Each study is fitted on its own, with no pooling across subjects or studies. Static subject-level states ([A5](#a5-time-invariant-constructs-are-static-subject-level-states)) are therefore not random effects in the multilevel sense.

## Causal graph

The model is one connected DAG over constructs. A theorized common cause appears as an explicit latent confounder node; the graph has no bidirected edges. Indicators belong to their construct and never cause it.

| Role | Temporal status | Example |
| --- | --- | --- |
| Exogenous | Time-varying | Weather, day of week |
| Exogenous | Time-invariant | Age, person intercept |
| Endogenous | Time-varying | Mood, stress, sleep quality |
| Endogenous | Time-invariant | Baseline severity, a stable trait outcome |

An edge is contemporaneous (lag 0) or lagged by one tick of the measurement clock.

### A3. Markov dynamics

Endogenous time-varying constructs follow first-order Markov dynamics: the state at `t − 1` summarizes all earlier history. Higher-order lags are not modeled, and cross-lagged effects span one tick. Residual autocorrelation therefore points to missing cross-lags or confounders rather than higher-order dynamics. First-order within-person dynamics are the standard starting point in [dynamic SEM](https://doi.org/10.1080/10705511.2017.1406803).

### A4. Acyclic within a time slice

Lag-0 edges form a DAG, and feedback is modeled across time through lagged edges. Cyclic contemporaneous relations would not be identified without further constraints.

### A4b. Effects between evolving states act through the drift

A same-slice edge between two endogenous time-varying constructs is rejected. Such an effect is declared as a lagged edge and becomes a cross-term in the continuous-time drift. Same-time co-movement belongs to an explicit latent confounder or to diffusion correlation. This is a contract of the model class, not a claim about every dynamic latent-variable model.

### A5. Time-invariant constructs are static subject-level states

A time-invariant construct keeps its initial value over the modeled window. It may be exogenous or endogenous, but its causes must also be time-invariant, because a changing cause cannot determine a fixed child. Static states separate stable between-person differences from within-person dynamics; failing to separate them [biases lagged effects](https://doi.org/10.1037/a0038889).

## Measurement

### A1. Reflective indicators

Each construct causes its indicators. Formative measurement, where indicators define the construct, is not supported. Indicators of one construct should therefore correlate, and removing one does not change the construct's meaning. Formative models suit composite indices rather than theoretical constructs; see [Diamantopoulos & Siguaw (2006)](https://doi.org/10.1111/j.1467-8551.2006.00500.x).

### A6. Indicator count sets how measurement error is handled

With two or more indicators, shared variance separates measurement error from construct variance; see [Bollen (1989)](https://doi.org/10.1002/9781118619179). With one indicator, [A9](#a9-a-single-indicator-absorbs-measurement-error) applies. The distinction affects precision, not identifiability.

### A8. Indicator errors are independent over time

All temporal dependence belongs to the constructs, so indicators get no autoregressive structure. Serially correlated residuals signal misspecification, such as a clock that is too coarse, missing cross-loadings or dynamics in the measurement itself; see [Asparouhov, Hamaker & Muthén (2018)](https://doi.org/10.1080/10705511.2017.1406803). Correlated indicator residuals are not supported.

### A9. A single indicator absorbs measurement error

A construct with one indicator is identified with that indicator: its loading is fixed, and measurement error merges into the structural noise. This is the standard fallback described by [Bollen (1989)](https://doi.org/10.1002/9781118619179). Its coefficients can be attenuated toward zero. Prefer several indicators when attenuation matters.

## Time

| Concept | Meaning |
| --- | --- |
| `measurement_clock` | The model's shared tick and default lag unit, such as one day. |
| `observation_window` | The support interval that one indicator value summarizes. It defaults to the clock and may differ per indicator. |
| `aggregation` | How a window is summarized, which fixes the value's support and anchor (next table). |
| `anchor_time` | Where the value attaches to the latent path. |
| `dt` | The time between consecutive prepared time points, including window boundaries. It scales each Euler–Maruyama step in `fit`. |
| Simulation window | `simulate` runs from `start` to `end` in model days. The model clock and any intervention times define its output grid. |

| `aggregation` | What matters | Support | Anchor |
| --- | --- | --- | --- |
| `mean` | Average level over the window | Interval | Window end |
| `sum` | Cumulative amount | Interval | Window end |
| `count` | Event frequency | Interval | Window end |
| `std` | Within-window instability | Interval | Window end |
| `last` | Most recent state | Point | Window end |
| `first` | Earliest state | Point | Window start |

These choices are substantive: a daily mean mood and an end-of-day mood encode different theories of what matters. Neither a point summary nor an exact (`Delta`) observation implies that a value persists between observations.

A historical simulation start still uses the selected revision's current joint law, including later observations used to fit it. Changing `start` does not undo conditioning or restrict the simulation to information available at that date.

## Causal identification

Identification is checked separately for each treatment's effect on the model's default outcome, using nonparametric do-calculus and the ID algorithm of [Shpitser & Pearl (2006)](https://aaai.org/Papers/AAAI/2006/AAAI06-191.pdf). One unidentified effect does not affect the others. Linear instrumental-variable arguments are not accepted, because the nonlinear model does not make the linearity assumptions they need.

### A3a. Scope of the two-slice check

The identifier uses two time slices with zero-or-one-tick causal lags. Markov dynamics do not guarantee that these slices capture all relevant confounding. [Jahn, Karnik & Schulman (2025)](https://proceedings.mlr.press/v275/jahn25a.html) establish finite bounds depending on graph width and maximum lag, not universal two-slice sufficiency. The current result establishes identification in the truncated graph; extending it to the full temporal process requires an additional assumption that the implementation does not verify.

### A7. Identified measurement lets constructs count as observed

Under A1, A6 and A9 the construct covariance is identified from the indicators, so graph identification can treat constructs as observed. This is the two-step logic of [Anderson & Gerbing (1988)](https://doi.org/10.1037/0033-2909.103.3.411). It requires the graph and the measurements to come from the same model revision.

The graph users see stays a DAG with explicit latent confounders. Internally, the check unrolls it to two slices, projects it to an ADMG as in [Richardson & Spirtes (2002)](https://doi.org/10.1214/aos/1031689015), and runs y0's identification algorithm. The projection never leaves the check.

## Parameter anchors

Every retained construct has exactly one location anchor and one scale anchor. Any shift or rescaling of a latent state that the anchors leave free must be absorbed by exactly one free parameter group. Two such groups would create an exact likelihood ridge, and none would over-constrain the model. `edit_model` reports a model that breaks this invariant as not executable.

- **Location** is anchored by the first of these that applies:
  1. A standardized channel: a mean-centered Gaussian or Student-t identity-link indicator whose intercept is fixed at zero.
  2. A direct exact state observation, `Delta(v=state(x))`, with additive-location support.
  3. For a time-invariant construct, a fixed initial mean.
  4. For an evolving construct, a complete restoring expression with a fixed center.

  Fixed means and centers need not be zero. Observation intercepts and thresholds are identified relative to the anchor; free latent-side locations require a standardized or exact state-observation anchor. The need to constrain competing location parameters follows the ctsem discussion in [Driver, Oud & Voelkle (2017)](https://doi.org/10.18637/jss.v077.i05).
- **Scale and sign:** the reference indicator's loading is fixed to ±1 by its polarity, the marker-variable convention of [Bollen (1989)](https://doi.org/10.1002/9781118619179). A construct measured only by categorical indicators instead pins its first non-baseline slope to +1, following the nominal-response model of [Bock (1972)](https://doi.org/10.1007/BF02291411). Reference indicators are chosen in this order: continuous, ordinal, binary or count, then categorical.

Some parameter pairs are only weakly separated, and the invariant deliberately doesn't gate them. No diagnostic currently flags them either:

- transient versus equilibrium location when dynamics are slow;
- static baseline-factor SDs in a single-subject fit;
- observation dispersion versus latent volatility;
- diffusion correlation versus fast reciprocal edges under coarse sampling;
- Hill `Emax` versus `EC50` when the input never spans the half-saturation point.

## Numeric causal claims

`simulate` reports a numeric effect only when all of these hold:

- The treatment's effect on the model's default outcome is identified within the [implemented graph check's scope](#a3a-scope-of-the-two-slice-check).
- The simulated model was conditioned by a committed production fit, using the particle engine with the nonlinear Euler–Maruyama target, and it matches that fit's record exactly.
- The simulated paths are finite.

Otherwise it still returns the generated histories and records why no effect is reported. Interventions are dated state assignments, after which the model's dynamics resume. Effects are reported for the default outcome only. Certification covers the identified treatment and outcome; it does not add a separate identification proof for a general timed joint intervention.

Certification does not look at convergence (R-hat, ESS), predictive checks or LOO. Read those before trusting a reported effect.

## Reading results

- **LOO**, when configured, estimates how well each measurement row is predicted from all other rows, including later ones; see [Vehtari, Gelman & Gabry (2017)](https://doi.org/10.1007/s11222-016-9696-4). It measures interpolation, not forecasting; forecasting needs [leave-future-out validation](https://doi.org/10.1080/00949655.2020.1783262), which is not implemented. LOO is omitted when the data contain exact observations, and LOO-PIT is not computed.
- **Predictive checks** of fitted laws against the panel they were fitted on are in-sample. A different panel revision does not prove the observations were held out. Neither case establishes held-out calibration.
- **Sampler diagnostics** cover parameters only: split R-hat, ESS and MCSE. Latent paths get no convergence diagnostics. All chains start near one Pathfinder mode, so R-hat cannot reveal a missed mode.
- **Calibration and recovery** are not automated. A `simulate` → `prepare_data` → `fit` round trip mixes sampler error with the gap between the Diffrax solver and Euler–Maruyama.
- **Prior sensitivity** is not measured: there is no power-scaling or posterior-contraction check.
- **The DT-to-CT diagnostic** compares the elementwise conversion of a linear reference matrix, built from the priors, with its matrix logarithm. Its warning says nothing about the full nonlinear system.
- **C3 resolvability** is a screen that compares `1 / decay` with the observation gaps and span. It is not a nonlinear relaxation analysis or an identification test.
- **Data pattern warnings**, such as dominant duplicate values or arithmetic sequences, flag possible fabrication; they do not prove it.
