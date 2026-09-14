# Likelihoods

Defines the observation-model vocabulary for [`LikelihoodSpec`](../../pipeline/statistical-model-spec.md#likelihoodspec) entries in a [`ModelSpec`](../../pipeline/statistical-model-spec.md#model-statistical-choices).

## Conditional Expressions

`LikelihoodSpec.law` contains a native `distribution` name and its `arguments`. Arguments use scientific state references, coefficient operands, arithmetic, and bounded functions. In the table, `p = a + Σ bᵢ state(xᵢ)` is an affine predictor; each loading refers to its actual construct. `a`, `bᵢ`, and the auxiliary operands are fixed coefficients or references to `ModelSpec.parameters`.

| Native law | Argument expressions |
|---|---|
| `Delta` | `v=state(x)` or `v=p` |
| `Normal` | `loc=p`, `scale=s` |
| `StudentT` | `df=ν`, `loc=p`, `scale=s` |
| `Poisson` | `rate=exp(p)` |
| `Gamma` | `concentration=k`, `rate=k / exp(p)` or `k / (1 / p)` |
| `Bernoulli` | `logits=p` or `probs=normal_cdf(p)` |
| `NegativeBinomial2` | `mean=exp(p)`, `concentration=r` |
| `Beta` | `concentration1=m * c`, `concentration0=(1 - m) * c`, where `m=sigmoid(p)` or `normal_cdf(p)` |
| `OrderedLogistic` | `predictor=p`, `cutpoints=ordered_cutpoints(base, gaps)` |
| `Categorical` | `logits=category_logits(p, intercepts, slopes)` |

`ordered_cutpoints` starts at the base threshold and cumulatively adds positive gaps in declared ordinal-level order. For two levels, its gap argument is literal zero. `category_logits` prepends the reference category's zero logit to `intercepts + slopes * p`; the indicator supplies category order and the existing anchor rules apply. These structured arguments retain the existing category-specific parameter identities.

`Delta(v=state(x))` declares an exact measurement without a loading, intercept, or noise parameter to author. Its numerical view derives a unit loading and zero intercept. The existing [aggregation and support semantics](../../pipeline/measurement-structure.md#indicator) determine what is exact: a point value for `first`/`last`, or the declared window summary for interval observations. An exact window mean does not force a constant trajectory. Missing observations impose no equality and do not fill gaps or turn the construct into a fixed input.

Delta densities and predictive draws preserve exact equality and zero measurement variance. The current particle backend rejects Delta observations before parameter initialization: its continuous proposals cannot condition on exact equalities. Fitting these laws requires constraint-preserving proposals; replacing Delta with a small Gaussian variance would change the authored model.

The numerical backend derives its family and response from these expressions. It rejects unsupported formulas before fitting. The same declaration supplies displayed equations and parameter references. Earlier family/link/slot records require the explicit offline converter at `apps/data-pipeline/scripts/migrate_likelihood_expressions.py`; runtime schemas accept the conditional law form.

> The sections below are generated from `nof1_causal_lab.distributions`.
> Edit the Python catalog and re-run `uv run python scripts/export_distribution_docs.py` instead of editing them manually.

## Dtype-to-Distribution Mapping

Each indicator's [`measurement_dtype`](../../pipeline/measurement-structure.md#indicator) selects the default conditional law. The family and link names below describe its numerical lowering. Where the dtype admits only one valid combination, the likelihood is locked by [component authoring](../../pipeline/statistical-model-spec.md). Where alternatives exist, the LLM chooses via a decision card.

| `measurement_dtype` | Default distribution | Link | Alternatives |
|---|---|---|---|
| `continuous` | `gaussian` | `identity` | `student_t` (`identity`), `gamma` (`log` or `inverse`), `beta` (`logit` or `probit`), `delta` (`identity`) |
| `binary` | `bernoulli` | `logit` | `bernoulli` with `probit`, `delta` (`identity`) |
| `count` | `poisson` | `log` | `negative_binomial` (`log`), `delta` (`identity`) |
| `ordinal` | `ordered_logistic` | `cumulative_logit` | `delta` (`identity`) |
| `categorical` | `categorical` | `softmax` | `ordered_logistic` (`cumulative_logit`) when categories are substantively ordered, `delta` (`identity`) |

## Distribution Families

`DistributionFamily` names the internal numerical emission kernels: `gaussian`, `student_t`, `poisson`, `gamma`, `bernoulli`, `negative_binomial`, `beta`, `ordered_logistic`, `categorical`, and `delta`.

## Link Functions

`LinkFunction` names the internal responses derived from conditional expressions: `identity`, `log`, `inverse`, `logit`, `probit`, `cumulative_logit`, and `softmax`.
