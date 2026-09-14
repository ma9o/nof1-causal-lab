import modelArtifact from "../../../../../data/DEMO/fixture/artifacts/model.json";
import authoringLog from "../../../../../data/DEMO/fixture/model_authoring.json";
import inferenceLog from "../../../../../data/DEMO/fixture/inference.json";
import validationReportArtifact from "../../../../../data/DEMO/fixture/artifacts/validation_report.json";

export const demoArtifactSources = {
  model: modelArtifact,
  prior_predictive: authoringLog.prior_predictive,
  validation_report: validationReportArtifact,
  inference_report: inferenceLog.report,
} as const;
