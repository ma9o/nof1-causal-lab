"""Executable statistical-model-spec artifact models and validation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nof1_causal_lab.distributions import (
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
    PARAMETER_ROLE_SPECS,
    VALID_LIKELIHOODS_FOR_DTYPE,
    DistributionFamily,
)
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

from .base import ArtifactPayload
from .evidence import LiteratureSource  # noqa: TC001
from .identity import (  # noqa: TC001
    ConstructId,
    EntityRef,
    IndicatorId,
    ParameterElementId,
    ParameterId,
)
from .mechanism import DynamicsMechanism, EstimatedCoefficient, mechanism_coefficients
from .parameter import PriorAuthoringTransform, SiteKind
from .prior_proposal import PriorProposal  # noqa: TC001


class LinkFunction(StrEnum):
    """A link function connects an observation distribution to the model's predictor."""

    IDENTITY = "identity"
    LOG = "log"
    INVERSE = "inverse"
    LOGIT = "logit"
    PROBIT = "probit"
    CUMULATIVE_LOGIT = "cumulative_logit"
    SOFTMAX = "softmax"


class InitializationPolicy(StrEnum):
    """This policy selects stationary-derived or freely estimated initial conditions for
    dynamic states.
    """

    STATIONARY = "stationary"
    FREE = "free"


class ObservationInterceptPolicy(StrEnum):
    """This policy determines whether eligible observation intercepts are fixed or freely
    estimated.
    """

    FIXED = "fixed"
    FREE = "free"


class ParameterRole(StrEnum):
    """A parameter role identifies which part of the statistical model a parameter controls."""

    FIXED_EFFECT = "fixed_effect"
    AR_COEFFICIENT = "ar_coefficient"
    DYNAMICS_PARAMETER = "dynamics_parameter"  # noqa: V107 - public StrEnum value construction
    DYNAMICS_PARAMETER_POSITIVE = "dynamics_parameter_positive"
    RESIDUAL_SD = "residual_sd"
    STATE_INTERCEPT = "state_intercept"
    OBSERVATION_INTERCEPT = "observation_intercept"
    INITIAL_STATE_MEAN = "initial_state_mean"
    INITIAL_STATE_SD = "initial_state_sd"
    STATIC_STATE_SD = "static_state_sd"
    CORRELATION = "correlation"
    INITIAL_STATE_CORRELATION = "initial_state_correlation"
    LOADING = "loading"
    MEASUREMENT_ERROR_SD = "measurement_error_sd"
    OBSERVATION_HYPERPARAMETER = "observation_hyperparameter"  # noqa: V107 - public StrEnum value construction
    OBSERVATION_HYPERPARAMETER_POSITIVE = "observation_hyperparameter_positive"  # noqa: V107 - public StrEnum value construction


class ParameterConstraint(StrEnum):
    """A parameter constraint specifies the permitted range of a model parameter."""

    NONE = "none"  # noqa: V107 - consumed through StrEnum value construction
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNIT_INTERVAL = "unit_interval"
    CORRELATION = "correlation"


VALID_LINKS_FOR_DISTRIBUTION: dict[DistributionFamily, set[LinkFunction]] = {
    family: {LinkFunction(link) for link in links}
    for family, links in OBSERVATION_LINK_VALUES_BY_DISTRIBUTION.items()
}

EXPECTED_CONSTRAINT_FOR_ROLE: dict[ParameterRole, ParameterConstraint] = {
    ParameterRole(spec.role): ParameterConstraint(spec.constraint) for spec in PARAMETER_ROLE_SPECS
}
EXPECTED_CONSTRAINT_FOR_ROLE[ParameterRole.INITIAL_STATE_CORRELATION] = (
    ParameterConstraint.CORRELATION
)


class LikelihoodSpec(BaseModel):
    """A likelihood specification defines how an indicator's observed values follow from the
    model.
    """

    indicator_id: IndicatorId = Field(description="Persistent identity of the observed indicator")
    distribution: DistributionFamily = Field(description="Distribution family for this variable")
    link: LinkFunction = Field(description="Link function mapping linear predictor to mean")
    standardized: bool = Field(
        default=False,
        description=(
            "Whether deterministic standardization (mean-centering and unit-scaling) "
            "is applied to the observed values before fitting"
        ),
    )
    reasoning: str = Field(description="Why this distribution/link was chosen for this variable")
    sources: list[LiteratureSource] = Field(
        default_factory=list,
        description="Literature sources supporting this likelihood choice",
    )

    @model_validator(mode="after")
    def validate_distribution_link_pair(self) -> LikelihoodSpec:
        """Reject recognized links that are invalid for this distribution."""
        allowed_links = VALID_LINKS_FOR_DISTRIBUTION[self.distribution]
        if self.link not in allowed_links:
            expected = ", ".join(sorted(link.value for link in allowed_links))
            raise ValueError(
                f"link '{self.link.value}' is invalid for {self.distribution.value}; "
                f"expected one of {{{expected}}}"
            )
        return self


class ParameterSpec(BaseModel):
    """A parameter specification declares a named model quantity, its role, and its allowed
    values.
    """

    id: ParameterId
    owners: list[EntityRef]
    quantity: SiteKind
    name: str = Field(description="Authored parameter label; relationships use its persistent ID")
    role: ParameterRole = Field(description="Role of this parameter in the model")
    constraint: ParameterConstraint = Field(description="Constraint on parameter values")
    description: str = Field(
        description="Human-readable description of what this parameter represents"
    )
    prior_transform: PriorAuthoringTransform = PriorAuthoringTransform.IDENTITY
    elements: dict[ParameterElementId, str] = Field(
        default_factory=dict,
        description="Logical scalar components and their labels, declared during compilation.",
    )


class StatisticalModelSpec(BaseModel):
    """A statistical model specification defines likelihoods, parameter roles, and estimation
    policies.
    """

    model_config = ConfigDict(extra="forbid")

    likelihoods: list[LikelihoodSpec] = Field(
        description="Likelihood specifications for each observed indicator"
    )
    parameters: list[ParameterSpec] = Field(description="All parameters requiring priors")
    mechanisms: list[DynamicsMechanism] = Field(
        description="Explicit state and edge dynamics; coefficients reference parameter IDs."
    )
    initialization_policy: InitializationPolicy = Field(
        default=InitializationPolicy.STATIONARY,
        description="Whether dynamic-state initial conditions are stationary-derived or free",
    )
    observation_intercept_policy: ObservationInterceptPolicy = Field(
        default=ObservationInterceptPolicy.FREE,
        description="Whether eligible manifest intercepts remain free or are fixed",
    )

    @model_validator(mode="after")
    def validate_references(self) -> StatisticalModelSpec:
        for label, identities in (
            ("parameter", [parameter.id for parameter in self.parameters]),
            ("likelihood", [likelihood.indicator_id for likelihood in self.likelihoods]),
        ):
            if len(identities) != len(set(identities)):
                raise ValueError(f"Duplicate {label} identities")
        parameter_ids = {parameter.id for parameter in self.parameters}
        for mechanism in self.mechanisms:
            for coefficient in mechanism_coefficients(mechanism).values():
                if (
                    isinstance(coefficient, EstimatedCoefficient)
                    and coefficient.parameter_id not in parameter_ids
                ):
                    raise ValueError(
                        f"Mechanism references undeclared parameter {coefficient.parameter_id!r}"
                    )
        return self


def validate_statistical_model_spec_dict(
    data: UncheckedJsonObject,
    indicators: list[UncheckedJsonObject] | None = None,
) -> tuple[StatisticalModelSpec | None, list[str]]:
    """Validate a statistical model spec dict, collecting all errors in one pass."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    valid_roles = {entry.value for entry in ParameterRole}
    valid_constraints = {entry.value for entry in ParameterConstraint}
    valid_distributions = {entry.value for entry in DistributionFamily}
    valid_links = {entry.value for entry in LinkFunction}
    valid_initialization_policies = {entry.value for entry in InitializationPolicy}
    valid_observation_intercept_policies = {entry.value for entry in ObservationInterceptPolicy}

    likelihoods = data.get("likelihoods", [])
    if not isinstance(likelihoods, list):
        errors.append("'likelihoods' must be a list")
        likelihoods = []

    likelihood_variables = [
        item.get("indicator_id", "") for item in likelihoods if isinstance(item, dict)
    ]
    seen_likelihood_variables: set[str] = set()
    for variable in likelihood_variables:
        if variable and variable in seen_likelihood_variables:
            errors.append(f"duplicate likelihood for variable '{variable}'")
        if variable:
            seen_likelihood_variables.add(variable)

    initialization_policy = data.get(
        "initialization_policy",
        InitializationPolicy.STATIONARY.value,
    )
    if initialization_policy not in valid_initialization_policies:
        errors.append(
            "'initialization_policy' invalid; must be one of "
            f"{sorted(valid_initialization_policies)}"
        )

    observation_intercept_policy = data.get(
        "observation_intercept_policy",
        ObservationInterceptPolicy.FREE.value,
    )
    if observation_intercept_policy not in valid_observation_intercept_policies:
        errors.append(
            "'observation_intercept_policy' invalid; must be one of "
            f"{sorted(valid_observation_intercept_policies)}"
        )

    indicator_dtype: dict[str, str] = {}
    if indicators:
        indicator_dtype = {
            indicator["id"]: indicator.get("measurement_dtype", "continuous")
            for indicator in indicators
        }
        missing = set(indicator_dtype) - seen_likelihood_variables
        for variable in sorted(missing):
            errors.append(f"indicator '{variable}' has no likelihood specification")

    for index, likelihood in enumerate(likelihoods):
        if not isinstance(likelihood, dict):
            errors.append(f"likelihoods[{index}]: must be a dictionary")
            continue

        variable = likelihood.get("indicator_id", "")
        distribution = likelihood.get("distribution", "")
        link = likelihood.get("link", "")

        if distribution and distribution not in valid_distributions:
            errors.append(
                f"likelihoods[{index}] '{variable}': distribution '{distribution}' invalid; "
                f"must be one of {sorted(valid_distributions)}"
            )
        if link and link not in valid_links:
            errors.append(
                f"likelihoods[{index}] '{variable}': link '{link}' invalid; "
                f"must be one of {sorted(valid_links)}"
            )

        if distribution in valid_distributions and link in valid_links:
            distribution_enum = DistributionFamily(distribution)
            link_enum = LinkFunction(link)
            allowed_links = VALID_LINKS_FOR_DISTRIBUTION.get(distribution_enum)
            if allowed_links is not None and link_enum not in allowed_links:
                errors.append(
                    f"likelihoods[{index}] '{variable}': link '{link}' invalid for {distribution}; "
                    f"expected one of {{{', '.join(sorted(item.value for item in allowed_links))}}}"
                )

        if distribution in valid_distributions and variable in indicator_dtype:
            dtype = indicator_dtype[variable]
            allowed_distributions = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype)
            if (
                allowed_distributions is not None
                and DistributionFamily(distribution) not in allowed_distributions
            ):
                errors.append(
                    f"likelihoods[{index}] '{variable}': distribution '{distribution}' invalid for dtype '{dtype}'; "
                    f"expected one of {{{', '.join(sorted(item.value for item in allowed_distributions))}}}"
                )

    parameters = data.get("parameters", [])
    if not isinstance(parameters, list):
        errors.append("'parameters' must be a list")
        parameters = []

    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, dict):
            errors.append(f"parameters[{index}]: must be a dictionary")
            continue

        name = parameter.get("name", f"[{index}]")
        role = parameter.get("role", "")
        constraint = parameter.get("constraint", "")

        if role and role not in valid_roles:
            errors.append(
                f"parameters[{index}] '{name}': role '{role}' invalid; "
                f"must be one of {sorted(valid_roles)}"
            )
        if constraint and constraint not in valid_constraints:
            errors.append(
                f"parameters[{index}] '{name}': constraint '{constraint}' invalid; "
                f"must be one of {sorted(valid_constraints)}"
            )

        if role in valid_roles and constraint in valid_constraints:
            role_enum = ParameterRole(role)
            constraint_enum = ParameterConstraint(constraint)
            expected = EXPECTED_CONSTRAINT_FOR_ROLE.get(role_enum)
            if role_enum == ParameterRole.LOADING:
                if constraint_enum not in {
                    ParameterConstraint.POSITIVE,
                    ParameterConstraint.NEGATIVE,
                }:
                    errors.append(
                        f"parameters[{index}] '{name}': constraint '{constraint}' unexpected "
                        "for role 'loading'; expected 'positive' or 'negative'"
                    )
            elif expected is not None and constraint_enum != expected:
                errors.append(
                    f"parameters[{index}] '{name}': constraint '{constraint}' unexpected "
                    f"for role '{role}'; expected '{expected.value}'"
                )

    if not errors:
        try:
            spec = StatisticalModelSpec.model_validate(data)
            return spec, []
        except ValidationError as exc:
            return None, [f"Unexpected validation error: {exc}"]

    return None, errors


__all__ = [
    "EXPECTED_CONSTRAINT_FOR_ROLE",
    "LinkFunction",
    "LikelihoodSpec",
    "StatisticalModelSpec",
    "ParameterConstraint",
    "ParameterRole",
    "ParameterSpec",
    "VALID_LINKS_FOR_DISTRIBUTION",
    "validate_statistical_model_spec_dict",
]


class PriorPredictiveDiagnostic(BaseModel):
    """A prior predictive diagnostic records the result of one exact model-admission check."""

    model_config = ConfigDict(extra="forbid")

    check: str
    construct_id: ConstructId
    value: str
    band: str
    passed: bool
    note: str
    diagnosis: list[str] = Field(default_factory=list)
    mode: str


class StatisticalModelSpecArtifact(ArtifactPayload):
    """This artifact combines the statistical specification with prior proposals and admission
    diagnostics.
    """

    statistical_model_spec: StatisticalModelSpec
    authored_priors: dict[ParameterId, PriorProposal]
    resolved_priors: list[PriorProposal]
    search_queries: dict[str, str] | None = None
    validation_warnings: list[str] | None = None
    prior_predictive_samples: dict[IndicatorId, list[float]] | None = None
    prior_predictive_diagnostics: list[PriorPredictiveDiagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_prior_references(self) -> StatisticalModelSpecArtifact:
        parameters = {parameter.id for parameter in self.statistical_model_spec.parameters}
        for key, proposal in self.authored_priors.items():
            if key != proposal.parameter_id or key not in parameters:
                raise ValueError("Authored prior does not reference its declared parameter")
        resolved_ids = [proposal.parameter_id for proposal in self.resolved_priors]
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError("Duplicate resolved prior parameter references")
        return self
