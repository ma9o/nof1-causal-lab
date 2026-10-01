import type { StudyRevision } from "@nof1-causal-lab/api-types";

/**
 * A journal attempt; only applied attempts install a new scientific version.
 */
export interface JournalTick {
  seq: number;
  commitId: string;
  parentIds: string[];
  branch: string;
  ts: string;
  action: StudyRevision["action"];
  inputs: StudyRevision["inputs"];
  status: StudyRevision["status"];
  produced: StudyRevision["produced"];
  error: string | null;
  traceIds: string[];
  checks: StudyRevision["checks"];
  messages: StudyRevision["messages"];
}

export function journalTicks(attempts: readonly StudyRevision[]): JournalTick[] {
  const ticks: JournalTick[] = [];
  for (const record of attempts) {
    ticks.push({
      seq: record.seq,
      commitId: record.commit_id,
      parentIds: record.parent_ids,
      branch: record.branch,
      ts: record.ts,
      action: record.action,
      inputs: record.inputs,
      status: record.status,
      produced: record.produced,
      error: record.reason ?? record.error_message ?? record.error_type ?? null,
      traceIds: record.trace_ids,
      checks: record.checks,
      messages: record.messages,
    });
  }
  return ticks;
}

export function latestSeq(attempts: readonly StudyRevision[]): number {
  return attempts.reduce(
    (max, record) =>
      record.status === "applied" && record.action !== "data_diff"
        ? Math.max(max, record.seq)
        : max,
    0,
  );
}
