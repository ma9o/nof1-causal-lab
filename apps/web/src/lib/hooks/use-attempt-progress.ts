"use client";

import type { RunningAction } from "@nof1-causal-lab/api-types";
import { applyProgressEvents, EMPTY_ATTEMPT_PROGRESS } from "@/lib/attempt-progress";

/** Timeline messages own the structured progress entries. */
export function useAttemptProgress(_workspaceId: string, running: RunningAction) {
  return {
    data: applyProgressEvents(
      EMPTY_ATTEMPT_PROGRESS,
      running.messages.flatMap((message) =>
        message.kind === "progress" ? [message.progress] : [],
      ),
    ),
  };
}
