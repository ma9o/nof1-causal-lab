import modelArtifact from "../../../../../data/DEMO/fixture/artifacts/model.json";
import admissionArtifact from "../../../../../data/DEMO/fixture/artifacts/admission_report.json";
import baselineReportArtifact from "../../../../../data/DEMO/fixture/artifacts/baseline_report.json";
import inferenceLog from "../../../../../data/DEMO/fixture/inference.json";
import validationReportArtifact from "../../../../../data/DEMO/fixture/artifacts/validation_report.json";

export const demoArtifactSources = {
  model: modelArtifact,
  admission_report: admissionArtifact,
  validation_report: validationReportArtifact,
  inference_report: inferenceLog.report,
  baseline_report: baselineReportArtifact,
} as const;
