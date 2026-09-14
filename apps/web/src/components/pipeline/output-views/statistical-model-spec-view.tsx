import { modelConstructs } from "@/lib/model-accessors";
import { FunctionalSpecLink } from "@/components/analysis-widgets/statistical-model-spec/functional-spec-link";
import { MeasurementTable } from "@/components/analysis-widgets/statistical-model-spec/measurement-table";
import { PriorTable } from "@/components/analysis-widgets/statistical-model-spec/prior-table";
import { SSMEquationDisplay } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import type { PriorPredictiveDiagnostic, ModelSnapshot } from "@nof1-causal-lab/api-types";

function PriorPredictiveDiagnostics({
  diagnostics,
  names,
}: {
  diagnostics: PriorPredictiveDiagnostic[];
  names: Record<string, string>;
}) {
  if (diagnostics.length === 0) return null;
  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">Prior-predictive reachability</h3>
        <p className="text-sm text-muted-foreground">
          Persisted checks from exact construct admission, including feedback-component rechecks.
        </p>
      </div>
      <ul className="grid gap-2 md:grid-cols-2">
        {diagnostics.map((diagnostic, index) => (
          <li
            key={`${diagnostic.check}:${diagnostic.construct_id}:${index}`}
            className={
              diagnostic.passed
                ? "rounded-md border border-success/25 bg-success/5 p-3"
                : "rounded-md border border-warning/30 bg-warning/10 p-3"
            }
          >
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="font-medium">{diagnostic.check}</span>
              <span className="text-muted-foreground">{diagnostic.passed ? "PASS" : "REVIEW"}</span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {names[diagnostic.construct_id]}: {diagnostic.value}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Target: {diagnostic.band}</p>
            {!diagnostic.passed && diagnostic.diagnosis.length > 0 && (
              <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                {diagnostic.diagnosis.map((line) => (
                  <p key={line}>{line}</p>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function StatisticalModelSpecView({ data }: { data: ModelSnapshot }) {
  const model = data.model?.value;
  if (!model) return null;
  const indicators = modelConstructs(model).flatMap((construct) => construct.indicators);
  const diagnostics = data.findings.diagnostics;
  const hasLikelihoodDiagnostics = Object.values(diagnostics?.likelihood_diagnostics ?? {}).some(
    (diagnostics) => (diagnostics?.histogram.length ?? 0) > 0,
  );

  return (
    <div className="space-y-4">
      {diagnostics && (
        <SSMEquationDisplay
          equations={diagnostics.state_equations}
          confounderEquations={diagnostics.confounder_equations}
          observationEquations={diagnostics.observation_equations}
          parameters={model.parameters}
          indicators={indicators}
          model={model}
        />
      )}
      {hasLikelihoodDiagnostics && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Likelihoods Diagnostics</h3>
            <FunctionalSpecLink />
          </div>
          <MeasurementTable
            diagnostics={diagnostics?.likelihood_diagnostics ?? {}}
            indicators={indicators ?? []}
          />
        </div>
      )}
      <PriorPredictiveDiagnostics
        diagnostics={data.findings.prior_predictive?.value.diagnostics ?? []}
        names={Object.fromEntries(
          modelConstructs(model).map((construct) => [construct.id, construct.name]),
        )}
      />
      {model.parameters.length > 0 && (
        <div className="space-y-3">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold">Parameter Distributions</h3>
            <p className="text-sm text-muted-foreground">
              Distributions are shown on each parameter’s declared scale. The state equations show
              any conversion to the continuous-time model scale.
            </p>
          </div>
          <PriorTable
            parameters={model.parameters}
            distributions={model.distributions}
            densities={diagnostics?.prior_densities ?? {}}
          />
        </div>
      )}
    </div>
  );
}
