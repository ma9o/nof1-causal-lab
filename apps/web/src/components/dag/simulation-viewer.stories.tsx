import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { buildSimulationScenarios } from "@/lib/dag/simulation-results";
import { withContainer } from "@/components/story-decorators";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import {
  constructs,
  edgePosteriors,
  edges,
  identifiableTreatments,
  indicators,
  demoSimulationTrace,
  nodeStatuses,
  persistencePosteriors,
} from "@/components/dag/__fixtures__/simulation-fixture";
import { SimulationViewer } from "./simulation-viewer";

/**
 * analysis simulation viewer driven by the ideal materialized artifact.
 *
 * Two layers (per the output view design):
 *  - Generative — direct `simulate` dispatch mints new scenarios. Disabled read-only.
 *  - Presentational — the viewer (left) shows persisted intervention scenarios,
 *    the LLM's blurb for the focused scenario,
 *    and the living DAG. Selection is shared: chat "View" ↔ rail.
 *
 * Scenarios come entirely from the persisted DEMO trace, exercising the real
 * string→object coercion and the production simulation visualization payload
 * without a component-local world.
 */

const scenarios = buildSimulationScenarios({ trace: demoSimulationTrace });

const graph = {
  constructs,
  edges,
  indicators,

  edgePosteriors,
  persistencePosteriors,
  identifiableTreatments,
  nodeStatuses,
};

function SimulationViewerWithChat() {
  const [selectedKey, setSelectedKey] = useState<string | null>(scenarios[0]?.key ?? null);

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(360px,1fr)]">
      <SimulationViewer
        scenarios={scenarios}
        graph={graph}
        selectedKey={selectedKey}
        onSelect={setSelectedKey}
      />
      <div className="flex h-[760px] min-h-0 flex-col rounded-lg border bg-muted/30 p-3">
        <LLMTracePanelView
          trace={demoSimulationTrace}
          selectedSimulationKey={selectedKey ?? undefined}
          onSelectSimulation={(key) => setSelectedKey(key)}
        />
      </div>
    </div>
  );
}

const meta = {
  title: "Analysis/Simulation Viewer",
  component: SimulationViewer,
  decorators: [withContainer("max-w-6xl")],
} satisfies Meta<typeof SimulationViewer>;

export default meta;

export const Materialized: StoryObj = {
  name: "Materialized fixture scenarios",
  render: () => <SimulationViewerWithChat />,
};
