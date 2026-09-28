import { RefreshCw } from "lucide-react";
import type { PipelineProgress } from "@/lib/hooks/pipeline-progress";

export function StaleResultsBannerView({ staleTransitionCount }: { staleTransitionCount: number }) {
  return (
    <div
      role="status"
      className="mx-auto flex w-full max-w-[1600px] items-center gap-2 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm"
    >
      <RefreshCw className="h-4 w-4 shrink-0 text-warning-foreground" />
      <span>
        {staleTransitionCount} artifact{staleTransitionCount === 1 ? "" : "s"} have stale results —
        inputs changed since they last ran.
      </span>
    </div>
  );
}

export function countStaleTransitions(progress: PipelineProgress): number {
  return progress.transitionOrder.filter(
    (artifactId) =>
      (progress.staleArtifactsByProducer[artifactId]?.length ?? 0) > 0 &&
      progress.artifacts[artifactId] !== "running",
  ).length;
}

export function StaleResultsBanner({ progress }: { progress: PipelineProgress }) {
  const staleTransitionCount = countStaleTransitions(progress);
  return staleTransitionCount > 0 ? (
    <StaleResultsBannerView staleTransitionCount={staleTransitionCount} />
  ) : null;
}
