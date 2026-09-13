"use client";

import type {
  ArtifactViewId,
  ArtifactViewData as ArtifactViewPayload,
  BaselineReportArtifact,
  Indicator,
  LatentStructureArtifact,
  MeasurementStructureViewData,
  MeasurementsData,
  PosteriorArtifact,
  RawDataData,
  StatisticalModelSpecData,
  TransitionMeta,
  ValidationReportArtifact,
} from "@nof1-causal-lab/api-types";
import { type ComponentType, lazy, memo, type ReactNode, Suspense, useMemo } from "react";
import { deriveConstructStatuses } from "@/components/dag/construct-statuses";
import { createSimulateDispatch } from "@/components/dag/interactive/dispatch-simulate";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisTransitionRun } from "@/lib/api/analysis";
import { useWorkspaceView } from "@/lib/contexts/workspace-view-context";
import { useArtifactView } from "@/lib/hooks/use-artifact-view";
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

type ArtifactViewData = ArtifactViewPayload & {
  context?: string;
};

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
  const { data: artifactData } = useArtifactView<ArtifactViewData>(
    workspaceId,
    output.id,
    isCompleted,
  );
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

type OutputViewAdapterProps = {
  workspaceId: string;
  data: ArtifactViewData;
};

function createArtifactDataAdapter<TData>(Component: ComponentType<{ data: TData }>) {
  return function ArtifactDataAdapter({ data }: OutputViewAdapterProps) {
    return <Component data={data as TData} />;
  };
}

function ModelSpecConnectedContent({
  workspaceId,
  data,
}: {
  workspaceId: string;
  data: StatisticalModelSpecData;
}) {
  const { data: measurementStructure } = useArtifactView<MeasurementStructureViewData>(
    workspaceId,
    "measurement_structure",
    true,
  );
  return (
    <StatisticalModelSpecView
      data={data}
      indicators={measurementStructure?.causal_design.measurement.indicators}
    />
  );
}

function createIndicatorDataAdapter<TData>(
  Component: ComponentType<{ data: TData; workspaceId: string; indicators: Indicator[] }>,
) {
  return function IndicatorDataAdapter({ workspaceId, data }: OutputViewAdapterProps) {
    const { data: measurement } = useArtifactView<MeasurementStructureViewData>(
      workspaceId,
      "measurement_structure",
      true,
    );
    if (!measurement) return null;
    return (
      <Component
        workspaceId={workspaceId}
        data={data as TData}
        indicators={measurement.causal_design.measurement.indicators}
      />
    );
  };
}

function BaselineReportConnectedContent({
  workspaceId,
  data,
}: {
  workspaceId: string;
  data: BaselineReportArtifact;
}) {
  const { selectedScenarioKey, selectScenario, readOnly } = useWorkspaceView();
  const snapshot = useModelSnapshot(workspaceId);
  const { data: latentStructure } = useArtifactView<LatentStructureArtifact>(
    workspaceId,
    "latent_structure",
    true,
  );
  const { data: measurementStructure } = useArtifactView<MeasurementStructureViewData>(
    workspaceId,
    "measurement_structure",
    true,
  );
  const model = snapshot.data;
  const { data: llmTrace } = useLLMTrace(workspaceId, "baseline_report", true);

  // The scientific DAG remains the stable base. Fitted edge posteriors and simulation
  // trajectories appear only where the backend materialized them; marginalized
  // constructs stay visible as subdued theory context.
  const graph = useMemo(() => {
    const design = measurementStructure?.causal_design;
    return {
      constructs: latentStructure?.latent_structure.constructs ?? [],
      edges: latentStructure?.latent_structure.edges ?? [],
      indicators: design?.measurement.indicators,
      knownInputs: design?.known_inputs,
      edgePosteriors: buildEdgePosteriors({
        latentStructure,
        estimates: model?.fit?.value.edge_estimates ?? {},
      }),
      persistencePosteriors: buildPersistencePosteriors({
        latentStructure,
        estimates: model?.fit?.value.decay_estimates ?? {},
      }),
      identifiableTreatments: data.intervention_results.map(({ treatment }) => treatment),
      nodeStatuses:
        design && measurementStructure
          ? deriveConstructStatuses(design, measurementStructure.structural_plan)
          : undefined,
    };
  }, [data.intervention_results, latentStructure, measurementStructure, model]);
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

const outputViewAdapters = {
  raw_data: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <RawDataView workspaceId={workspaceId} data={data as RawDataData} />
  ),
  latent_structure: createArtifactDataAdapter<LatentStructureArtifact>(LatentStructureView),
  measurement_structure:
    createArtifactDataAdapter<MeasurementStructureViewData>(MeasurementStructureView),
  measurements: createIndicatorDataAdapter<MeasurementsData>(MeasurementsView),
  validation_report: createIndicatorDataAdapter<ValidationReportArtifact>(ValidationReportView),
  statistical_model_spec: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <ModelSpecConnectedContent workspaceId={workspaceId} data={data as StatisticalModelSpecData} />
  ),
  posterior: createIndicatorDataAdapter<PosteriorArtifact>(PosteriorView),
  baseline_report: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <BaselineReportConnectedContent
      workspaceId={workspaceId}
      data={data as BaselineReportArtifact}
    />
  ),
} satisfies Record<ArtifactViewId, (props: OutputViewAdapterProps) => ReactNode>;

function OutputView({
  artifactId,
  workspaceId,
  data,
}: {
  artifactId: ArtifactViewId;
  workspaceId: string;
  data?: ArtifactViewData;
}) {
  if (!data) return null;
  const renderOutputView = outputViewAdapters[artifactId];
  return renderOutputView ? renderOutputView({ workspaceId, data }) : null;
}
