"use client";

import { modelConstructs } from "@/lib/model-accessors";
import type { PipelineSectionId, ModelSnapshot, TransitionMeta } from "@nof1-causal-lab/api-types";
import { lazy, memo, Suspense } from "react";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisTransitionRun } from "@/lib/api/analysis";
import { useLLMTrace } from "@/lib/hooks/use-llm-trace";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import type { TransitionRunStatus, TransitionTiming } from "@/lib/hooks/use-run-events";
import { resolveTransitionObservedStatus } from "@/lib/transition-runtime";
import { OutputPresentationShell } from "./output-presentation-shell";

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
  }
}
