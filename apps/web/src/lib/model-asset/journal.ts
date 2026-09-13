import type { ArtifactId, Move, TransitionRecord } from "@nof1-causal-lab/api-types";

/**
 * One tick of the version scrubber: an applied or raised move. Rejected attempts never
 * executed, so they stay off the rail and live only in the conversation as activity.
 */
export interface JournalTick {
  seq: number;
  ts: string;
  move: Move;
  status: "applied" | "raised";
  /** Version this move installed for its own artifact; null for raised moves. */
  version: number | null;
  /** Artifacts installed alongside the move's own artifact (derived co-outputs). */
  derived: ArtifactId[];
  retracted: ArtifactId[];
  error: string | null;
  traceIds: string[];
}

export function journalTicks(transitions: readonly TransitionRecord[]): JournalTick[] {
  const ticks: JournalTick[] = [];
  for (const record of transitions) {
    if (record.status === "rejected") {
      continue;
    }
    const own = record.produced.find((info) => info.artifact_id === record.move.artifact_id);
    ticks.push({
      seq: record.seq,
      ts: record.ts,
      move: record.move,
      status: record.status,
      version: own?.version ?? null,
      derived: record.produced
        .filter((info) => info.artifact_id !== record.move.artifact_id)
        .map((info) => info.artifact_id),
      retracted: record.retracted.map((entry) => entry.artifact_id),
      error: record.error_message ?? record.error_type ?? null,
      traceIds: record.trace_ids,
    });
  }
  return ticks;
}

/** The asset as it stood once journal seq `playhead` had been applied. */
export interface AssetSnapshot {
  playhead: number;
  /** Version in force per artifact; absent artifacts are absent, retracted ones removed. */
  versions: Partial<Record<ArtifactId, number>>;
  /** Journal seq that installed the version in force. */
  installedAt: Partial<Record<ArtifactId, number>>;
  /** Artifacts retracted by the playhead and not produced again since. */
  retracted: ArtifactId[];
}

export function modelPosition(
  model: import("@nof1-causal-lab/api-types").ModelSnapshot,
): AssetSnapshot {
  return {
    playhead: model.seq,
    versions: Object.fromEntries(
      Object.entries(model.state.current).map(([id, info]) => [id, info.version]),
    ),
    installedAt: model.installed_at,
    retracted: model.retracted,
  };
}

export function latestSeq(transitions: readonly TransitionRecord[]): number {
  return transitions.reduce(
    (max, record) => (record.status === "applied" ? Math.max(max, record.seq) : max),
    0,
  );
}
