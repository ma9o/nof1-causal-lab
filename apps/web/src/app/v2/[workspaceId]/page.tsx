"use client";

import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import Link from "next/link";
import { use, useEffect } from "react";
import { CausalModelAsset } from "@/components/model/causal-model-asset";
import { useRunEvents } from "@/lib/hooks/use-run-events";

const ACTIVITY_SECTIONS = TRANSITIONS.map((transition) => transition.id);

/** Inspect the canonical study from its first, possibly empty, checkpoint. */
export default function ModelPage({ params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = use(params);
  const episodeQuery = useRunEvents(workspaceId, ACTIVITY_SECTIONS);
  const episode = episodeQuery.data;

  useEffect(() => {
    document.title = `${workspaceId} | Model workbench | nof1-causal-lab`;
  }, [workspaceId]);

  if (episodeQuery.error && !episode) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6">
        <div className="max-w-md space-y-3 rounded-lg border bg-card p-6 text-center">
          <h1 className="text-lg font-semibold">Workspace unavailable</h1>
          <p className="text-sm text-muted-foreground">{episodeQuery.error.message}</p>
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

  if (!episode) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">
        Reading the journal…
      </div>
    );
  }

  return <CausalModelAsset workspaceId={workspaceId} question={undefined} episode={episode} />;
}
