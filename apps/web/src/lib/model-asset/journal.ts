import {
  MACHINE_DESCRIPTION,
  type ArtifactId,
  type StudyRevision,
} from "@nof1-causal-lab/api-types";

/** Resolve operation outputs through the backend's declared machine graph. */
export function primaryArtifact(record: StudyRevision): ArtifactId | null {
  const action = MACHINE_DESCRIPTION.actions.find((entry) => entry.action_id === record.action)!;
  return (
    [...action.produces, ...action.produces_optional].find((id) =>
      record.produced.some((item) => item.artifact_id === id),
    ) ?? null
  );
}

/**
 * A journal attempt; only applied attempts install a new scientific version.
 */
export interface JournalTick {
  seq: number;
  attemptId: string | null;
  commitId: string;
  parentIds: string[];
  branch: string;
  ts: string;
  action: StudyRevision["action"];
  inputs: StudyRevision["inputs"];
  status: StudyRevision["status"];
  produced: StudyRevision["produced"];
  /** Version this action installed for its own artifact; null for raised actions. */
  revision: string | null;
  /** Artifacts installed alongside the action's primary artifact (derived co-outputs). */
  derived: ArtifactId[];
  retracted: ArtifactId[];
  error: string | null;
  errorType: string | null;
  traceIds: string[];
  diagnostics: StudyRevision["diagnostics"];
  messages: StudyRevision["messages"];
}

export function journalTicks(transitions: readonly StudyRevision[]): JournalTick[] {
  const ticks: JournalTick[] = [];
  for (const record of transitions) {
    const own = record.produced.find((info) => info.artifact_id === primaryArtifact(record));
    ticks.push({
      seq: record.seq,
      attemptId: record.attempt_id ?? null,
      commitId: record.commit_id,
      parentIds: record.parent_ids,
      branch: record.branch,
      ts: record.ts,
      action: record.action,
      inputs: record.inputs,
      status: record.status,
      produced: record.produced,
      revision: own?.revision ?? null,
      derived: record.produced
        .filter((info) => info.artifact_id !== primaryArtifact(record))
        .map((info) => info.artifact_id),
      retracted: record.retracted.map((entry) => entry.artifact_id),
      error: record.reason ?? record.error_message ?? record.error_type ?? null,
      errorType: record.error_type ?? null,
      traceIds: record.trace_ids,
      diagnostics: record.diagnostics,
      messages: record.messages,
    });
  }
  return ticks;
}

export function latestSeq(transitions: readonly StudyRevision[]): number {
  return transitions.reduce(
    (max, record) => (record.status === "applied" ? Math.max(max, record.seq) : max),
    0,
  );
}
