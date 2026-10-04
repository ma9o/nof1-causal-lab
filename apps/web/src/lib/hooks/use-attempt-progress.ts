"use client";

import type { RunningAction } from "@nof1-causal-lab/api-types";
import { applyProgressEvents, EMPTY_ATTEMPT_PROGRESS } from "@/lib/attempt-progress";

/** Timeline polling owns progress. Replaying a completed failure would start another run. */
export function useAttemptProgress(_workspaceId: string, running: RunningAction) {
  return { data: applyProgressEvents(EMPTY_ATTEMPT_PROGRESS, running.events) };
}
