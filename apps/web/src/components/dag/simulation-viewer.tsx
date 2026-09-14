"use client";

import { ManifestProjection } from "@/components/analysis-widgets/posterior/treatment-effect-visuals";
import type { SimulationScenario } from "@/components/dag/simulation-results";
import type {
  CausalEdgeSpec,
  ConstructSpec,
  PosteriorEstimate,
  IndicatorSpec,
} from "@nof1-causal-lab/api-types";
import { Bot, TriangleAlert } from "lucide-react";
import { useMemo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { InteractiveDag } from "./interactive/interactive-dag";
import type { SimulateFn } from "./interactive/simulate-input";
import { ScenarioRail } from "./scenario-rail";
import type { ConstructStatus } from "./structure-dag";

export interface SimulationViewerGraph {
  constructs: ConstructSpec[];
  edges: CausalEdgeSpec[];
  indicators?: IndicatorSpec[];
  edgePosteriors?: Record<string, PosteriorEstimate>;
  persistencePosteriors?: Record<string, PosteriorEstimate>;
  identifiableTreatments?: string[];
  nodeStatuses?: Record<string, ConstructStatus>;
  indicatorsVisible?: boolean;
}

export interface SimulationViewerProps {
  scenarios: SimulationScenario[];
  graph: SimulationViewerGraph;
  selectedKey?: string | null;
  onSelect?: (key: string) => void;
  /** Live simulate seam; when present, do() editing is enabled on the DAG. */
  onSimulate?: SimulateFn;
  onNodeClick?: (constructName: string) => void;
}

/**
 * The LLM's explanation produced with the focused scenario — reasoning behind the
 * intervention and what the simulation shows. Sits directly under the carousel.
 */
function ScenarioBlurb({ scenario }: { scenario: SimulationScenario }) {
  if (!scenario.blurb?.trim()) {
    return null;
  }
  return (
    <div className="flex gap-3 rounded-lg border bg-muted/20 p-4">
      <Bot className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="prose prose-sm max-w-none text-sm [&_p]:my-1.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0">
        <Markdown remarkPlugins={[remarkGfm]}>{scenario.blurb}</Markdown>
      </div>
    </div>
  );
}

function SimulationWarnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-400/40 bg-amber-400/5 p-3 text-xs text-amber-700 dark:text-amber-400">
      <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <ul className="space-y-0.5">
        {warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
    </div>
  );
}

function ScenarioDetail({
  scenario,
  graph,
  onSimulate,
  onNodeClick,
}: {
  scenario: SimulationScenario;
  graph: SimulationViewerGraph;
  onSimulate?: SimulateFn;
  onNodeClick?: (constructName: string) => void;
}) {
  return (
    <div className="space-y-3">
      <InteractiveDag
        constructs={graph.constructs}
        edges={graph.edges}
        indicators={graph.indicators}
        edgePosteriors={graph.edgePosteriors}
        persistencePosteriors={graph.persistencePosteriors}
        identifiableTreatments={graph.identifiableTreatments}
        indicatorsVisible={graph.indicatorsVisible}
        nodeStatuses={graph.nodeStatuses}
        result={scenario.result}
        onSimulate={onSimulate}
        onNodeClick={onNodeClick}
      />
      <SimulationWarnings warnings={scenario.result.warnings} />
      {scenario.manifestEffects ? (
        <ManifestProjection
          manifestEffects={scenario.manifestEffects}
          className="rounded-lg border bg-muted/20 p-3"
        />
      ) : null}
    </div>
  );
}

/**
 * analysis simulation viewer. The left output column: a rail of
 * backend-materialized intervention scenarios,
 * the LLM's blurb for the focused scenario, and the living DAG. The chat that
 * mints new scenarios lives in the shell's trace pane and shares selection via
 * RefinementContext.
 */
export function SimulationViewer({
  scenarios,
  graph,
  selectedKey,
  onSelect,
  onSimulate,
  onNodeClick,
}: SimulationViewerProps) {
  const selected = useMemo(
    () => scenarios.find((scenario) => scenario.key === selectedKey) ?? scenarios[0] ?? null,
    [scenarios, selectedKey],
  );

  return (
    <div className="space-y-4">
      {scenarios.length === 0 ? (
        <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
          No scenarios available yet. Ask in chat to simulate an intervention.
        </div>
      ) : (
        <>
          <ScenarioRail
            scenarios={scenarios}
            selectedKey={selected?.key ?? null}
            onSelect={(key) => onSelect?.(key)}
          />
          {selected ? <ScenarioBlurb scenario={selected} /> : null}
          {selected ? (
            <ScenarioDetail
              scenario={selected}
              graph={graph}
              onSimulate={onSimulate}
              onNodeClick={onNodeClick}
            />
          ) : null}
        </>
      )}
    </div>
  );
}
