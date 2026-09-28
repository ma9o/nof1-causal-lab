import type { RuntimeEvent } from "@nof1-causal-lab/api-types";

export const EXTRACTION_EVENT_PREFIX = "nof1-causal-lab.extraction.";
const EXTRACTION_RPM_WINDOW_MS = 60_000;

export type ExtractionWorkerState = "pending" | "running" | "completed" | "failed";

export interface ExtractionPlan {
  total_workers: number;
  max_concurrent_workers: number | null;
  max_rpm: number | null;
}

export interface ExtractionWorkerRecord {
  worker_id: number;
  state: ExtractionWorkerState;
  n_windows: number;
  n_extractions: number | null;
  n_llm_calls: number | null;
  error: string | null;
  completed_at: string | null;
}

export interface ExtractionSnapshot {
  total_workers: number;
  pending_workers: number;
  running_workers: number;
  completed_workers: number;
  failed_workers: number;
  llm_requests_last_60s: number;
}

export type ExtractionEventRecord = RuntimeEvent & { occurred?: string | null };

export type ExtractionEvent =
  | { type: "plan"; plan: ExtractionPlan }
  | { type: "snapshot"; snapshot: ExtractionSnapshot }
  | { type: "worker"; worker: ExtractionWorkerRecord };

export interface ExtractionReplayState {
  plan: ExtractionPlan | null;
  snapshot: ExtractionSnapshot | null;
  workers: Record<string, ExtractionWorkerRecord>;
}

export interface ExtractionSummary {
  total: number;
  pending: number;
  running: number;
  completed: number;
  failed: number;
}

export const EMPTY_EXTRACTION_REPLAY_STATE: ExtractionReplayState = {
  plan: null,
  snapshot: null,
  workers: {},
};

export function getExtractionStateQueryKey(workspaceId: string) {
  return ["pipeline", workspaceId, "extraction-state"] as const;
}

function stateRank(state: ExtractionWorkerState): number {
  switch (state) {
    case "pending":
      return 0;
    case "running":
      return 1;
    case "completed":
    case "failed":
      return 2;
  }
}

function createPendingWorker(workerId: number): ExtractionWorkerRecord {
  return {
    worker_id: workerId,
    state: "pending",
    n_windows: 0,
    n_extractions: null,
    n_llm_calls: null,
    error: null,
    completed_at: null,
  };
}

function mergeWorker(
  existing: ExtractionWorkerRecord | undefined,
  incoming: ExtractionWorkerRecord,
): ExtractionWorkerRecord {
  if (!existing) {
    return incoming;
  }

  const state =
    stateRank(incoming.state) >= stateRank(existing.state) ? incoming.state : existing.state;

  return {
    ...existing,
    ...incoming,
    state,
    n_windows: incoming.n_windows || existing.n_windows,
    n_extractions: incoming.n_extractions ?? existing.n_extractions,
    n_llm_calls: Math.max(existing.n_llm_calls ?? 0, incoming.n_llm_calls ?? 0) || null,
    error: incoming.error ?? existing.error,
    completed_at: incoming.completed_at ?? existing.completed_at,
  };
}

export function parseExtractionEvent(event: ExtractionEventRecord): ExtractionEvent | null {
  switch (event.event) {
    case "nof1-causal-lab.extraction.plan":
      return {
        type: "plan",
        plan: {
          total_workers: event.total_workers,
          max_concurrent_workers: event.max_concurrent_workers ?? null,
          max_rpm: event.max_rpm ?? null,
        },
      };
    case "nof1-causal-lab.extraction.snapshot":
      return {
        type: "snapshot",
        snapshot: {
          total_workers: event.total_workers,
          pending_workers: event.pending_workers,
          running_workers: event.running_workers,
          completed_workers: event.completed_workers,
          failed_workers: event.failed_workers,
          llm_requests_last_60s: event.llm_requests_last_60s,
        },
      };
    case "nof1-causal-lab.extraction.worker":
      return {
        type: "worker",
        worker: {
          worker_id: event.worker_id,
          state: event.state,
          n_windows: event.n_windows,
          n_extractions: event.n_extractions ?? null,
          n_llm_calls: event.n_llm_calls ?? null,
          error: event.error ?? null,
          completed_at:
            event.state === "completed" || event.state === "failed"
              ? (event.occurred ?? null)
              : null,
        },
      };
    default:
      return null;
  }
}

export function applyExtractionEvent(
  state: ExtractionReplayState | undefined,
  event: ExtractionEvent,
): ExtractionReplayState {
  const next = state ?? EMPTY_EXTRACTION_REPLAY_STATE;

  if (event.type === "plan") {
    const workers = { ...next.workers };
    for (let workerId = 0; workerId < event.plan.total_workers; workerId += 1) {
      const key = String(workerId);
      workers[key] = workers[key] ?? createPendingWorker(workerId);
    }
    return {
      ...next,
      plan: event.plan,
      workers,
    };
  }

  if (event.type === "snapshot") {
    return {
      ...next,
      snapshot: event.snapshot,
    };
  }

  const workerKey = String(event.worker.worker_id);
  return {
    ...next,
    workers: {
      ...next.workers,
      [workerKey]: mergeWorker(next.workers[workerKey], event.worker),
    },
  };
}

export function listExtractionWorkers(
  state: ExtractionReplayState | null | undefined,
): ExtractionWorkerRecord[] {
  return Object.values(state?.workers ?? {}).sort(
    (left, right) => left.worker_id - right.worker_id,
  );
}

export function summarizeExtractionState(
  state: ExtractionReplayState | null | undefined,
): ExtractionSummary {
  const workers = listExtractionWorkers(state);
  if (state?.snapshot) {
    return {
      total: state.snapshot.total_workers,
      pending: state.snapshot.pending_workers,
      running: state.snapshot.running_workers,
      completed: state.snapshot.completed_workers,
      failed: state.snapshot.failed_workers,
    };
  }
  const summary: ExtractionSummary = {
    total: state?.plan?.total_workers ?? workers.length,
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
  };

  for (const worker of workers) {
    summary[worker.state] += 1;
  }

  return summary;
}

export function getExtractionRequestsPerMinute(
  state: ExtractionReplayState | null | undefined,
  now = Date.now(),
): number {
  if (typeof state?.snapshot?.llm_requests_last_60s === "number") {
    return state.snapshot.llm_requests_last_60s;
  }

  let total = 0;
  for (const worker of Object.values(state?.workers ?? {})) {
    if (!worker.completed_at || !worker.n_llm_calls) {
      continue;
    }

    const completedAt = Date.parse(worker.completed_at);
    if (!Number.isFinite(completedAt)) {
      continue;
    }

    if (now - completedAt < EXTRACTION_RPM_WINDOW_MS) {
      total += worker.n_llm_calls;
    }
  }
  return total;
}
