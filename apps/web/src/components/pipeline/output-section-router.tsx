"use client";

import { modelConstructs } from "@/lib/model-accessors";
import type {
  PipelineSectionId,
  ModelSnapshot,
  BaselineReportArtifact,
  TransitionMeta,
} from "@nof1-causal-lab/api-types";
import { lazy, memo, Suspense, useMemo } from "react";
import { constructStatuses } from "@/components/dag/construct-statuses";
import { createSimulateDispatch } from "@/components/dag/interactive/dispatch-simulate";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisTransitionRun } from "@/lib/api/analysis";
import { useWorkspaceView } from "@/lib/contexts/workspace-view-context";
import { useLLMTrace } from "@/lib/hooks/use-llm-trace";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import type { TransitionRunStatus, TransitionTiming } from "@/lib/hooks/use-run-events";
import { resolveTransitionObservedStatus } from "@/lib/transition-runtime";
import { OutputPresentationShell } from "./output-presentation-shell";
import {
  buildBaselineReportScenarios,
  buildEdgePosteriors,
  buildPersistencePosteriors,
} from "./output-views/baseline-report-scenarios";

const RawDataView = lazy(() => import("./output-views/raw-data-view"));
const LatentStructureView = lazy(() => import("./output-views/latent-structure-view"));
const MeasurementStructureView = lazy(() => import("./output-views/measurement-structure-view"));
const MeasurementsView = lazy(() => import("./output-views/measurements-view"));
const MeasurementsRunningOutputView = lazy(
  () => import("./output-views/measurements-running-view"),
);
const StatisticalModelSpecRunningOutputView = lazy(
  () => import("./output-views/statistical-model-spec-running-view"),
);
const ValidationReportView = lazy(() => import("./output-views/validation-report-view"));
const StatisticalModelSpecView = lazy(() => import("./output-views/statistical-model-spec-view"));
const PosteriorView = lazy(() => import("./output-views/posterior-view"));
const LLMTracePanel = lazy(() =>
  import("@/components/ui/custom/llm-trace-panel").then((module) => ({
    default: module.LLMTracePanel,
  })),
);
const SimulationViewer = lazy(() =>
  import("@/components/dag/simulation-viewer").then((module) => ({
    default: module.SimulationViewer,
  })),
);

type OutputSectionRouterProps = {
  output: TransitionMeta;
  workspaceId: string;
  status: TransitionRunStatus;
  timing?: TransitionTiming;
  transitionRun?: AnalysisTransitionRun;
  errorMessage?: string;
  staleArtifactIds?: string[];
};

function transitionRunsEqual(
  previous?: AnalysisTransitionRun,
  next?: AnalysisTransitionRun,
): boolean {
  return (
    previous?.execution?.stateType === next?.execution?.stateType &&
    previous?.execution?.startTime === next?.execution?.startTime &&
    previous?.execution?.endTime === next?.execution?.endTime
  );
}

function OutputSectionRouterInner({
  output,
  workspaceId,
  status,
  timing,
  transitionRun,
  errorMessage,
  staleArtifactIds,
}: OutputSectionRouterProps) {
  const effectiveStatus = resolveTransitionObservedStatus(status, transitionRun);
  const isCompleted = effectiveStatus === "completed";
  const elapsedMs =
    timing?.completedAt && timing?.startedAt ? timing.completedAt - timing.startedAt : undefined;

  // Read context + trace from the artifact data once the output has completed.
  const { data: artifactData } = useModelSnapshot(workspaceId);
  const { data: llmTrace } = useLLMTrace(workspaceId, output.id, isCompleted);

  return (
    <OutputPresentationShell
      output={output}
      status={effectiveStatus}
      elapsedMs={elapsedMs}
      context={output.description}
      errorMessage={errorMessage}
      staleArtifactIds={staleArtifactIds}
      loadingHint={output.loadingHint}
      runningContent={
        output.id === "measurements" && effectiveStatus === "running" ? (
          <Suspense fallback={null}>
            <MeasurementsRunningOutputView workspaceId={workspaceId} />
          </Suspense>
        ) : output.id === "statistical_model_spec" &&
          (effectiveStatus === "running" || effectiveStatus === "failed") ? (
          <Suspense fallback={null}>
            <StatisticalModelSpecRunningOutputView
              workspaceId={workspaceId}
              showError={effectiveStatus !== "failed"}
            />
          </Suspense>
        ) : undefined
      }
      panelContent={
        llmTrace ? (
          <Suspense fallback={null}>
            <LLMTracePanel trace={llmTrace} />
          </Suspense>
        ) : undefined
      }
    >
      {isCompleted && (
        <ErrorBoundary>
          <Suspense fallback={null}>
            <OutputView artifactId={output.id} workspaceId={workspaceId} data={artifactData} />
          </Suspense>
        </ErrorBoundary>
      )}
    </OutputPresentationShell>
  );
}

export const OutputSectionRouter = memo(
  OutputSectionRouterInner,
  (previous, next) =>
    previous.workspaceId === next.workspaceId &&
    previous.output.id === next.output.id &&
    previous.status === next.status &&
    previous.timing?.startedAt === next.timing?.startedAt &&
    previous.timing?.completedAt === next.timing?.completedAt &&
    previous.errorMessage === next.errorMessage &&
    (previous.staleArtifactIds?.join("|") ?? "") === (next.staleArtifactIds?.join("|") ?? "") &&
    transitionRunsEqual(previous.transitionRun, next.transitionRun),
);

function BaselineReportConnectedContent({
  workspaceId,
  data,
  snapshot,
}: {
  workspaceId: string;
  snapshot: ModelSnapshot;
  data: BaselineReportArtifact;
}) {
  const { selectedScenarioKey, selectScenario, readOnly } = useWorkspaceView();
  const model = snapshot.model?.value;
  const { data: llmTrace } = useLLMTrace(workspaceId, "baseline_report", true);

  // The scientific DAG remains the stable base. Fitted edge posteriors and simulation
  // trajectories appear only where the backend materialized them; marginalized
  // constructs stay visible as subdued theory context.
  const graph = useMemo(
    () => ({
      constructs: modelConstructs(model) ?? [],
      edges: model?.edges ?? [],
      indicators: modelConstructs(model).flatMap((construct) => construct.indicators),
      edgePosteriors: buildEdgePosteriors({
        latentStructure: model,
        estimates: snapshot.findings.fit?.value.edge_estimates ?? {},
      }),
      persistencePosteriors: buildPersistencePosteriors({
        latentStructure: model,
        estimates: snapshot.findings.fit?.value.decay_estimates ?? {},
      }),
      identifiableTreatments: data.intervention_results.map(({ treatment }) => treatment),
      nodeStatuses: constructStatuses(snapshot),
    }),
    [data.intervention_results, model, snapshot],
  );
  const scenarios = useMemo(() => buildBaselineReportScenarios({ trace: llmTrace }), [llmTrace]);
  const onSimulate = useMemo(
    () => (readOnly ? undefined : createSimulateDispatch(workspaceId)),
    [readOnly, workspaceId],
  );

  return (
    <SimulationViewer
      scenarios={scenarios}
      graph={graph}
      selectedKey={selectedScenarioKey}
      onSelect={selectScenario}
      rankingResults={data.intervention_results}
      onSimulate={onSimulate}
    />
  );
}

function OutputView({
  artifactId,
  workspaceId,
  data,
}: {
  artifactId: PipelineSectionId;
  workspaceId: string;
  data?: ModelSnapshot;
}) {
  if (!data) return null;
  const indicators =
    modelConstructs(data.model?.value).flatMap((construct) => construct.indicators) ?? [];
  switch (artifactId) {
    case "raw_data":
      return (
        data.data.raw_data && (
          <RawDataView workspaceId={workspaceId} data={data.data.raw_data.value} />
        )
      );
    case "latent_structure":
      return data.model && <LatentStructureView data={data.model.value} />;
    case "measurement_structure":
      return data.model && <MeasurementStructureView data={data} />;
    case "statistical_model_spec":
      return data.model && <StatisticalModelSpecView data={data} />;
    case "measurements":
      return (
        data.data.measurements && (
          <MeasurementsView
            workspaceId={workspaceId}
            data={data.data.measurements.value}
            indicators={indicators}
          />
        )
      );
    case "validation_report":
      return (
        data.findings.validation_report && (
          <ValidationReportView
            data={data.findings.validation_report.value}
            indicators={indicators}
          />
        )
      );
    case "posterior":
      return (
        data.findings.fit && (
          <PosteriorView data={data.findings.fit.value.report} indicators={indicators} />
        )
      );
    case "baseline_report":
      return (
        data.findings.baseline_report && (
          <BaselineReportConnectedContent
            workspaceId={workspaceId}
            snapshot={data}
            data={data.findings.baseline_report.value}
          />
        )
      );
  }
}
