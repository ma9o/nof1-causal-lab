import { modelConstructs } from "@/lib/model-accessors";
/**
 * analysis scenario model.
 *
 * Every analysis "scenario" is a materialized `simulate` tool result — a start
 * state + a list of timed latent clamps — sourced from runtime tool responses,
 * carrying per-construct reference and action means on a shared time grid. The
 * production tool contract requires one or more `do()` clamps; each result also
 * carries the corresponding reference rollout for visual comparison.
 *
 * Each scenario also carries the natural-language **blurb** the LLM produced
 * alongside it (the assistant text co-located with the `simulate` tool call),
 * which explains the reasoning behind the intervention and its result.
 */

import type {
  EffectSummary,
  ModelSpec,
  LLMTrace,
  PosteriorEstimate,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { formatScenarioActionDescription } from "@/lib/dag/simulation";
import type { AnalysisSimulationResult } from "@/lib/dag/simulation-types";
import { parseSimulationResult } from "@/lib/simulation-result";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";

// ── scenario types ──────────────────────────────────────────────────────────

export type ScenarioProvenance = "intervention";

export interface SimulationScenario {
  /** Stable selection key — the `simulate` tool-call id. */
  key: string;
  provenance: ScenarioProvenance;
  /** Concise label for the rail card. */
  title: string;
  outcome: string;
  summary: EffectSummary;
  manifestEffects: Record<string, number> | null;
  result: AnalysisSimulationResult;
  requestedHorizonDays?: number;
  /** The user prompt that minted this scenario. */
  userQuery?: string;
  /** LLM-authored explanation produced with this scenario (assistant text beside the tool call). */
  blurb?: string;
}

// ── simulation sourcing ─────────────────────────────────────────────────────

const SIMULATION_TOOLS = new Set(["simulate"]);

interface RawSimulation {
  toolCallId: string;
  result: AnalysisSimulationResult;
  userQuery?: string;
  blurb?: string;
  order: number;
}

/**
 * Walk a UI-message stream and record every materialized `simulate` result,
 * keyed by tool-call id. The assistant text co-located with a simulate call is
 * captured as that scenario's `blurb`. `order` increases with recency; later
 * sources (and later occurrences) overwrite earlier ones with a higher order.
 */
function collectSimulations(
  messages: UIMessage[],
  into: Map<string, RawSimulation>,
  startOrder: number,
): number {
  let order = startOrder;
  let lastUserQuery: string | undefined;
  for (const message of messages) {
    if (message.role === "user") {
      const textPart = message.parts.find((part) => part.type === "text");
      if (textPart?.type === "text") {
        lastUserQuery = textPart.text;
      }
      continue;
    }
    if (message.role !== "assistant") {
      continue;
    }
    const blurb =
      message.parts
        .filter((part): part is Extract<typeof part, { type: "text" }> => part.type === "text")
        .map((part) => part.text)
        .join("\n\n")
        .trim() || undefined;
    for (const part of message.parts) {
      if (
        part.type !== "dynamic-tool" ||
        part.state !== "output-available" ||
        !SIMULATION_TOOLS.has(part.toolName)
      ) {
        continue;
      }
      const result = parseSimulationResult(part.output);
      if (!result) {
        continue;
      }
      into.set(part.toolCallId, {
        toolCallId: part.toolCallId,
        result,
        userQuery: lastUserQuery,
        blurb,
        order: order++,
      });
    }
  }
  return order;
}

function toScenario(raw: RawSimulation): SimulationScenario {
  return {
    key: raw.toolCallId,
    provenance: "intervention",
    title: formatScenarioActionDescription(raw.result),
    outcome: raw.result.labels[raw.result.request.outcome],
    summary: raw.result.summary,
    manifestEffects: raw.result.manifest_effects ?? null,
    result: raw.result,
    requestedHorizonDays: raw.result.request.readout.horizon_days,
    userQuery: raw.userQuery,
    blurb: raw.blurb,
  };
}

/**
 * Build the intervention scenario list newest first.
 */
export function buildSimulationScenarios(args: {
  trace?: LLMTrace | null;
  /** Additional live UI-message streams to merge with the supplied trace. */
  extraMessages?: UIMessage[];
}): SimulationScenario[] {
  const { trace, extraMessages } = args;

  const simulations = new Map<string, RawSimulation>();
  let order = 0;
  if (trace) {
    order = collectSimulations(traceToUIMessages(trace), simulations, order);
  }
  collectSimulations(extraMessages ?? [], simulations, order);

  return [...simulations.values()].sort((left, right) => right.order - left.order).map(toScenario);
}

// ── edge posteriors (graph-level, scenario-independent) ──────────────────────

/** Translate canonical edge IDs to graph display labels; estimates are computed in Python. */
export function buildEdgePosteriors({
  latentStructure,
  estimates,
}: {
  latentStructure?: ModelSpec | null;
  estimates: import("@nof1-causal-lab/api-types").FitSummary["edge_estimates"];
}): Record<string, PosteriorEstimate> {
  const names = new Map(
    modelConstructs(latentStructure).map((construct) => [construct.id, construct.name]),
  );
  return Object.fromEntries(
    (latentStructure?.edges ?? []).flatMap((edge) => {
      const estimate = estimates[edge.id];
      return estimate
        ? [[`${names.get(edge.cause.id)}→${names.get(edge.effect.id)}`, estimate]]
        : [];
    }),
  );
}

/** Decay estimates retain their runtime rate units. */
export function buildPersistencePosteriors({
  latentStructure,
  estimates,
}: {
  latentStructure?: ModelSpec | null;
  estimates: import("@nof1-causal-lab/api-types").FitSummary["decay_estimates"];
}): Record<string, PosteriorEstimate> {
  return Object.fromEntries(
    (modelConstructs(latentStructure) ?? []).flatMap((construct) => {
      const estimate = estimates[construct.id];
      return estimate ? [[construct.name, estimate]] : [];
    }),
  );
}
