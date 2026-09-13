"""Canonical JSON payload contracts for every file-backed machine artifact.

Panel is a Parquet table, so it has a storage contract but no JSON payload.
The machine file catalog declares all JSON, Parquet, and binary members.
"""

from pydantic import BaseModel

from .baseline_report import BaselineReportArtifact, SavedScenariosArtifact
from .causal_design import CausalDesignArtifact, IdentificationReport
from .compiled_ssm import CompiledSSMArtifact
from .identity import ArtifactId
from .latent_structure import LatentStructureArtifact
from .measurement_structure import MeasurementStructureArtifact
from .measurements import MeasurementsArtifact
from .posterior import PosteriorArtifact
from .question import QuestionArtifact
from .raw_data import RawDataArtifact
from .statistical_model_spec import StatisticalModelSpecArtifact
from .structural_plan import StructuralPlanArtifact
from .validation_report import ValidationReportArtifact

ARTIFACT_CONTRACTS: dict[ArtifactId, type[BaseModel]] = {
    "question": QuestionArtifact,
    "raw_data": RawDataArtifact,
    "latent_structure": LatentStructureArtifact,
    "measurement_structure": MeasurementStructureArtifact,
    "causal_design": CausalDesignArtifact,
    "structural_plan": StructuralPlanArtifact,
    "identification_report": IdentificationReport,
    "measurements": MeasurementsArtifact,
    "validation_report": ValidationReportArtifact,
    "statistical_model_spec": StatisticalModelSpecArtifact,
    "compiled_ssm": CompiledSSMArtifact,
    "posterior": PosteriorArtifact,
    "baseline_report": BaselineReportArtifact,
    "saved_scenarios": SavedScenariosArtifact,
}
