"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { use, useEffect } from "react";
import { CausalModelAsset } from "@/components/model/causal-model-asset";
import { getAnalysisManifest, getAnalysisManifestQueryKey } from "@/lib/api/analysis";
import { useEpisodeProgress } from "@/lib/hooks/use-episode-progress";
import { usePipelineStatus } from "@/lib/hooks/use-pipeline-status";
import { useRunEvents } from "@/lib/hooks/use-run-events";

/** One persistent causal model per workspace: the asset view. */
export default function ModelPage({ params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = use(params);
  const manifestQuery = useQuery({
    queryKey: getAnalysisManifestQueryKey(workspaceId),
    queryFn: () => getAnalysisManifest(workspaceId),
    enabled: !!workspaceId,
    staleTime: Infinity,
    retry: false,
  });
  const manifest = manifestQuery.data;
  useRunEvents(workspaceId, manifest?.transitionOrder);
  const progress = usePipelineStatus(workspaceId);
  const episode = useEpisodeProgress(workspaceId);

  useEffect(() => {
    document.title = manifest?.question
      ? `${manifest.question.slice(0, 60)} | nof1-causal-lab`
      : "Causal model | nof1-causal-lab";
  }, [manifest?.question]);

  if (manifestQuery.error && !manifest) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6">
        <div className="max-w-md space-y-3 rounded-lg border bg-card p-6 text-center">
          <h1 className="text-lg font-semibold">Workspace unavailable</h1>
          <p className="text-sm text-muted-foreground">{manifestQuery.error.message}</p>
          <Link
            href="/"
            className="inline-flex rounded-md border px-3 py-2 text-sm font-medium transition-colors hover:bg-secondary"
          >
            Return Home
          </Link>
        </div>
      </div>
    );
  }

  if (!manifest || !progress || !episode) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">
        Reading the journal…
      </div>
    );
  }

  return (
    <CausalModelAsset
      workspaceId={workspaceId}
      question={manifest.question}
      readOnly={manifest.readOnly}
      progress={progress}
      episode={episode}
    />
  );
}
