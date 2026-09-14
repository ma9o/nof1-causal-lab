import type {
  AnalysisManifest,
  AnalysisTransitionRun,
  AnalysisTransitionRuns,
} from "@/lib/api/analysis";
import {
  getMachineDescription,
  getEpisodeStatus,
  getEpisodeTimeline,
  type EpisodeStatus,
  type MachineDescription,
  type TransitionRecord,
} from "@/lib/server/episode-runs";
import { readArtifactJson } from "@/lib/server/artifacts";
import { TRANSITIONS, type ModelSpec, type PipelineSectionId } from "@nof1-causal-lab/api-types";

function emptyTransitionRun(): AnalysisTransitionRun {
  return { execution: null };
}

function isPipelineSectionId(value: unknown): value is PipelineSectionId {
  return typeof value === "string" && TRANSITIONS.some((transition) => transition.id === value);
}

function artifactViewOrder(machine: MachineDescription): PipelineSectionId[] {
  const ordered: PipelineSectionId[] = machine.topological_transition_order.flatMap((id) =>
    id === "measurements" ? [id, "validation_report"] : [id],
  );
  const missing = TRANSITIONS.filter((transition) => !ordered.includes(transition.id)).map(
    (transition) => transition.id,
  );
  if (missing.length > 0) {
    throw new Error(
      `Machine description omits artifact views from topological order: ${missing.join(", ")}`,
    );
  }
  return ordered;
}

async function readEpisodeQuestion(
  workspaceId: string,
  status: EpisodeStatus,
): Promise<string | undefined> {
  const model = status.state.current.model;
  if (model == null) {
    return undefined;
  }

  const parsed = await readArtifactJson<ModelSpec>(workspaceId, "model", "model", model.version);
  return parsed.question ?? undefined;
}

/**
 * Per-artifact execution summaries from the episode journal: the latest
 * completed run attempt per artifact wins (applied -> completed, raised ->
 * failed; rejected attempts never executed).
 */
function summarizeTimelineTransitionRuns(transitions: TransitionRecord[]): AnalysisTransitionRuns {
  const transitionRuns = Object.fromEntries(
    TRANSITIONS.map((transition) => [transition.id, emptyTransitionRun()]),
  ) as AnalysisTransitionRuns;

  for (const record of transitions) {
    if (record.move.kind !== "run") {
      continue;
    }
    const artifactId = record.move.operation_id;
    if (!isPipelineSectionId(artifactId)) {
      continue;
    }
    if (record.status === "rejected") {
      continue;
    }

    transitionRuns[artifactId] = {
      execution: {
        stateType: record.status === "applied" ? "COMPLETED" : "FAILED",
        startTime: record.ts,
        endTime: record.ts,
      },
    };
  }

  return transitionRuns;
}

/**
 * The manifest comes straight from the episode journal — a published
 * (read-only) workspace carries its journal along, so there is no separate
 * curated-demo path.
 */
export async function buildAnalysisManifest(
  workspaceId: string,
): Promise<Omit<AnalysisManifest, "readOnly"> | null> {
  const [status, timeline, machine] = await Promise.all([
    getEpisodeStatus(workspaceId),
    getEpisodeTimeline(workspaceId),
    getMachineDescription(),
  ]);
  if (timeline.transitions.length === 0) {
    return null;
  }

  const question = await readEpisodeQuestion(workspaceId, status);

  return {
    workspaceId,
    createdAt: timeline.transitions[0].ts,
    question,
    transitionOrder: artifactViewOrder(machine),
    transitionRuns: summarizeTimelineTransitionRuns(timeline.transitions),
  };
}
