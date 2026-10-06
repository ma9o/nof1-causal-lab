"""The seven public actions, each with its input and successful output body.

HTTP inputs use ``RevisionSelector`` (an exact revision or ``"latest"``);
saved inputs use resolved ``GitOid`` values. ``prepare_data.source`` is a
``SourceFolder`` over HTTP and a captured ``FileSourceRef`` in saved inputs.

Request envelopes live in ``actions.contracts``. Each output becomes the
``body`` of the polling envelope in ``actions.results``.
"""

from collections.abc import Mapping

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.checks import SpecificationAssessment
from nof1_causal_lab.artifacts.data_preparation import (
    ExtractionSpec,
    ExtractionWorkerResult,
    PreparedDataMetadata,
)
from nof1_causal_lab.artifacts.data_ref import DataRef, DataSelection
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.execution import StructuralItemDisposition
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import (
    ConstructId,
    ConstructRef,
    EdgeId,
    EdgeRef,
    GitOid,
    GitRef,
    IndicatorId,
    ParameterId,
)
from nof1_causal_lab.artifacts.model_checks import (
    ModelCheckReport,
    ModelPredictiveReport,
    QuestionCheckReport,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.artifacts.posterior import (
    FitSettingsSpec,
    InferenceReport,
    InferenceReportCore,
    ModelFitResult,
)
from nof1_causal_lab.artifacts.posterior_diagnostics import DensityCurve, PPCOverlay
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.artifacts.simulation import (
    ModelSimulationResult,
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.json_types import JsonValue
from nof1_causal_lab.study.snapshot_models import FitSummary, ModelGraphView
from nof1_causal_lab.study.view_models import (
    Change,
    DataVariableDiff,
    MeasurementsData,
    RawDataData,
    Unchanged,
)
from nof1_causal_lab.study.visual_models import ObservationData, ParameterDraws, SimulationPaths

# edit_question


class EditQuestionInput(Value):
    """The study question, its required outcome, and any named intervention queries."""

    question: QuestionSpec = Field(
        description=(
            "The replacement study question, including the required outcome and any "
            "intervention queries to assess against candidate models."
        )
    )


class EditQuestionOutput(Value):
    """The question saved by this action."""

    question: QuestionSpec = Field(
        description="The saved question, including its outcome and intervention queries."
    )


# edit_model


class EditModelInput[RevisionT](Value):
    """Create or revise a model from a question or model parent."""

    parent_ref: RevisionT = Field(
        description=(
            "Question or model revision to start from. A model parent supplies its pinned "
            "question. 'latest' selects the current model, otherwise the current question."
        )
    )
    model: ModelSpec = Field(
        description=(
            "Complete authored scientific definition, replacing the selected model rather than "
            "applying a field patch. Endogenous constructs are modeled, with or without parents, "
            "and include every latent construct. Exogenous constructs are given by direct exact "
            "Delta readings and have no dynamics, diffusion, initial coefficients or trajectory law."
        )
    )


class EditModelOutput(Value):
    """The produced model, its checks, and backend-computed display values.

    Optional findings are absent when no corresponding report was retained.
    """

    model: ModelSpec = Field(description="Saved scientific definition.")
    can_simulate: bool = Field(
        description="Whether this model compiles to a supported forward simulator."
    )
    checks: ModelCheckReport | None = Field(
        description="Combined findings retained by the action that produced the model."
    )
    identification: IdentificationReport | None = Field(
        description="Causal identification findings."
    )
    dispositions: tuple[StructuralItemDisposition, ...] | None = Field(
        description=(
            "How each structural entity is retained, marginalized, or rejected by the execution"
            " representation."
        )
    )
    graph: ModelGraphView = Field(
        description=(
            "Construct and edge identities, dynamic membership, and execution status for the "
            "graph view."
        )
    )
    validation_report: ValidationReportArtifact | None = Field(
        description=(
            "Data findings combined with model-dependent preflight checks, when a report is "
            "available."
        )
    )
    specification: tuple[SpecificationAssessment, ...] | None = Field(
        description=(
            "Individual assessments of whether the model can be compiled and used with its "
            "selected inputs."
        )
    )
    question_checks: QuestionCheckReport | None = Field(
        description="Findings about the model's ability to address the selected study question."
    )
    predictive: ModelPredictiveReport | None = Field(
        description="Predictive check findings for the model's implied behavior."
    )
    predictive_overlays: Mapping[IndicatorId, PPCOverlay] = Field(
        description="Observed and generated summaries keyed by indicator."
    )
    entity_failures: Mapping[ConstructId | EdgeId | IndicatorId, tuple[str, ...]] = Field(
        description=(
            "Messages attributed to constructs, edges, or indicators from the available "
            "scientific reports."
        )
    )
    confounder_equations: Mapping[ConstructId, str] = Field(
        description="Rendered equations for marginalized confounding terms, keyed by construct."
    )
    state_equations: Mapping[ConstructId, str] = Field(
        description="Rendered latent state dynamics keyed by construct."
    )
    observation_equations: Mapping[IndicatorId, str] = Field(
        description="Rendered measurement laws keyed by indicator."
    )
    likelihood_diagnostics: Mapping[IndicatorId, tuple[HistogramBin, ...]] = Field(
        description=(
            "Histograms of recorded indicator values for comparison with their declared "
            "likelihoods."
        )
    )
    authoring_prior_densities: Mapping[ParameterId, DensityCurve] = Field(
        description="Density curves for authored parameter laws, before conditioning on observations."
    )


# prepare_data


class PrepareDataInput[RevisionT, SourceT](Value):
    """Extract observations using the definitions of the selected model."""

    model_ref: RevisionT = Field(
        description="Model revision owning the clock and observation definitions."
    )
    source: SourceT = Field(
        description=(
            "Folder under `data/{workspace_id}/` at the HTTP boundary; the captured file "
            "reference in the saved request."
        )
    )
    extraction: Mapping[IndicatorId, ExtractionSpec] = Field(
        min_length=1,
        description=(
            "Computed rules or semantic extraction instructions keyed by the model's "
            "observation IDs."
        ),
    )
    context: str = Field(
        default="", description="Additional background for interpreting the uploaded source tables."
    )


class PrepareDataOutput(Value):
    """Observations retained by a preparation run and evidence of how they were made.

    A missing summary means its artifact or report is unavailable; it does not
    mean the corresponding observation count is zero.
    """

    workers: tuple[ExtractionWorkerResult, ...] = Field(
        description=(
            "Outcomes of the semantic extraction chunks, including retained observation and "
            "window counts and any chunk failures."
        )
    )
    extraction_reused: int | None = Field(  # noqa: FIELD003 -- External callers inspect how many extraction workers reused retained observations.
        description=(
            "Number of extraction chunks served from retained results; null when the attempt "
            "did not record this count."
        )
    )
    raw_data: RawDataData | None = Field(
        description=(
            "Uploaded table dimensions, sample rows, column descriptions, and date bounds when "
            "the source provides dates."
        )
    )
    measurements: MeasurementsData | None = Field(
        description=(
            "Retained observation counts, broken down by indicator, with a sample of extracted "
            "records."
        )
    )
    metadata: PreparedDataMetadata | None = Field(
        description=(
            "Source reference, preparation recipe, resolved observation schema, and the "
            "calendar origin used to interpret model time."
        )
    )
    profile: DataProfileArtifact | None = Field(
        description=(
            "Data quality findings retained for the prepared panel, when a matching profile "
            "report exists."
        )
    )
    data: ObservationData = Field(
        description=(
            "Full observation histories keyed by indicator ID, including measurement support "
            "intervals and missing values."
        )
    )


# fit


class FitInput[RevisionT](Value):
    """Condition one model on one selected observation history."""

    model_ref: RevisionT = Field(
        description="Model revision whose parameter law will be conditioned."
    )
    data_ref: RevisionT = Field(
        description="Revision containing the observations used for conditioning."
    )
    replicate_index: int = Field(
        ge=0,
        description=(
            "Zero-based selection of a single history within that data source; histories are "
            "not pooled."
        ),
    )
    settings: FitSettingsSpec = Field(
        default_factory=FitSettingsSpec,
        description="Sampler, initialization, and diagnostic settings for the run.",
    )


class FitOutput(Value):
    """The fitted model, inference findings, parameter draws, and numerical arrays."""

    model: EditModelOutput = Field(
        description="Model definition and scientific findings associated with this fit."
    )
    inference: ModelFitResult | None = Field(  # noqa: FIELD003 -- External callers locate this fit's input model, selected history and numerical evidence.
        description=(
            "Retained fit result, including input references and array references; absent when "
            "the attempt retained no numerical result."
        )
    )
    summary: FitSummary | None = Field(
        description="Convergence summary and display findings for the recorded fit."
    )
    inference_report: InferenceReport | None = Field(
        description="Full diagnostic report, when one was retained."
    )
    parameter_draws: ParameterDraws = Field(
        description=(
            "Per-parameter draws and empirical distributions, or a typed explanation of why "
            "draws are unavailable."
        )
    )
    arrays: Mapping[str, JsonValue] = Field(  # noqa: FIELD003 -- External callers consume this action's full numerical evidence without another read route.
        description=(
            "Numerical evidence keyed by its stored array reference. Non-finite elements are "
            "represented by JSON nulls."
        )
    )


# simulate


class SimulateInput[RevisionT](Value):
    """Generate histories from a selected model and an explicit simulation design."""

    model_ref: RevisionT = Field(
        description="Revision supplying the authored or fitted generative law."
    )
    panel_ref: RevisionT | None = Field(
        default=None,
        description=(
            "Optional exact panel supplying the calendar origin for an authored law; "
            "a fitted law retains the origin of its conditioning history."
        ),
    )
    simulation: SimulationSpec = Field(
        description="Requested start, horizon, replicate count, and interventions."
    )


class SimulateOutput(Value):
    """The generated observation histories, paths, report, and retained arrays."""

    simulation: ModelSimulationResult = Field(  # noqa: FIELD003 -- External callers read generated coordinates and provenance even when no simulation report is retained.
        description=(
            "Generation result with the input references, coordinates, and references to the "
            "retained numerical evidence."
        )
    )
    report: SimulationReport | None = Field(
        description="Simulation findings, when a report was retained."
    )
    data: tuple[ObservationData, ...] = Field(
        description="One observation history per retained replicate, in replicate order."
    )
    paths: SimulationPaths | None = Field(
        description=(
            "Recorded state and indicator trajectories, including a reference arm when present;"
            " absent if path evidence is unavailable."
        )
    )
    arrays: Mapping[str, JsonValue] = Field(  # noqa: FIELD003 -- External callers consume this action's full numerical evidence without another read route.
        description=(
            "Retained numerical evidence keyed by its stored array reference. Non-finite "
            "elements are represented by JSON nulls."
        )
    )


# data_diff


class DataDiffInput[RevisionT](Value):
    """Compare two selections of saved observation histories."""

    left_ref: DataSelection[RevisionT] = Field(
        description=(
            "References on the left side. Each reference selects one replicate by index, or all"
            " its retained histories when the index is omitted."
        )
    )
    right_ref: DataSelection[RevisionT] = Field(
        description="References on the right side, using the same selection rule."
    )


class DataDiffOutput(Value):
    """Per-variable comparisons and the exact history references on both sides."""

    left: tuple[DataRef[GitOid, int], ...] = Field(
        description="Resolved revision and replicate index for every left-side history."
    )
    right: tuple[DataRef[GitOid, int], ...] = Field(
        description="Resolved revision and replicate index for every right-side history."
    )
    variables: tuple[DataVariableDiff, ...] = Field(
        description=(
            "Series, point changes, statistics, compatibility issues, and any predictive "
            "comparison for each compared indicator."
        )
    )


# model_diff


class ModelDiffInput[RevisionT](Value):
    """The before and after model revisions or study checkpoints to compare."""

    before_ref: RevisionT = Field(
        description="Earlier model revision or checkpoint used as the comparison base."
    )
    after_ref: RevisionT = Field(
        description="Later model revision or checkpoint to compare with the base."
    )


class ModelDiffOutput(Value):
    """Compare model definitions and scientific evidence at two selections."""

    before: GitRef | None = Field(
        description="Artifact reference for the earlier model, if one is selected."
    )
    after: GitRef | None = Field(
        description="Artifact reference for the later model, if one is selected."
    )
    before_model: ModelSpec | None = Field(
        description="Earlier scientific definition, or null without a model."
    )
    after_model: ModelSpec | None = Field(
        description="Later scientific definition, or null without a model."
    )
    parameters: tuple[Change[ParameterSpec], ...] = Field(
        description="Added, removed, and revised parameter definitions."
    )
    constructs: tuple[Change[ConstructRef] | Unchanged[ConstructRef], ...] = Field(
        description="Construct identity changes, including unchanged identities."
    )
    edges: tuple[Change[EdgeRef] | Unchanged[EdgeRef], ...] = Field(
        description="Edge identity changes, including unchanged identities."
    )
    before_dispositions: tuple[StructuralItemDisposition, ...] = Field(
        description="Execution treatment of structural entities before the change."
    )
    after_dispositions: tuple[StructuralItemDisposition, ...] = Field(
        description="Execution treatment of structural entities after the change."
    )
    before_dynamic_construct_ids: tuple[ConstructId, ...] = Field(
        description="Constructs with state dynamics in the earlier selection."
    )
    after_dynamic_construct_ids: tuple[ConstructId, ...] = Field(
        description="Constructs with state dynamics in the later selection."
    )
    changed_inputs: tuple[str, ...] = Field(
        description=(
            "Names of scientific input dependencies whose selections differ between the two "
            "checkpoints."
        )
    )
    before_checks: tuple[SpecificationAssessment, ...] = Field(
        description="Specification findings retained for the earlier model."
    )
    after_checks: tuple[SpecificationAssessment, ...] = Field(
        description="Specification findings retained for the later model."
    )
    before_fit: InferenceReportCore | None = Field(
        description="Earlier inference summary, when available."
    )
    after_fit: InferenceReportCore | None = Field(
        description="Later inference summary, when available."
    )
    before_simulation: SimulationReport | None = Field(
        description="Earlier simulation findings, when available."
    )
    after_simulation: SimulationReport | None = Field(
        description="Later simulation findings, when available."
    )
