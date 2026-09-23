import { demoModelSnapshot, demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { demoSimulationTrace } from "../__fixtures__/simulation-fixture";
import { buildSimulationScenarios } from "@/lib/dag/simulation-results";
import type { ConstructId } from "@nof1-causal-lab/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { LayeredCausalGraph, type LayeredCausalGraphProps } from "./layered-causal-graph";

/** Stories own the selection the way the asset view does. */
function SelectableLayeredCausalGraph(
  props: Omit<LayeredCausalGraphProps, "selectedNode" | "onSelectNode">,
) {
  const [selectedNode, setSelectedNode] = useState<ConstructId | null>(null);
  return (
    <LayeredCausalGraph {...props} selectedNode={selectedNode} onSelectNode={setSelectedNode} />
  );
}

const structureModel = demoSnapshotAt(3);
const measurementModel = demoSnapshotAt(4);
const designModel = demoSnapshotAt(4);
const specificationModel = demoSnapshotAt(7);
const fitModel = demoSnapshotAt(8);
const simulationResult = buildSimulationScenarios({ trace: demoSimulationTrace })[0].result;

const meta = {
  title: "DAG/Layered Causal Graph",
  component: SelectableLayeredCausalGraph,
  tags: ["svg-materialize"],
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Six cumulative artifact layers over the backend-selected graph. Early checkpoints show authored structure; established execution dispositions select the retained constructs and edges. Every story uses the canonical DEMO fixture.",
      },
    },
    svgMaterializer: {
      selector: 'svg[aria-label="Layered causal graph"]',
      auto: true,
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof SelectableLayeredCausalGraph>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Structure: Story = {
  name: "1 · Structure",
  args: { model: structureModel },
};

export const Measurement: Story = {
  name: "2 · + Measurement",
  args: { model: measurementModel },
};

export const Design: Story = {
  name: "3 · + Design",
  args: { model: designModel },
};

export const Specification: Story = {
  name: "4 · + Specification",
  args: { model: specificationModel },
};

export const Fit: Story = {
  name: "5 · + Fit",
  args: { model: fitModel },
};

export const Simulation: Story = {
  name: "6 · + Simulation",
  args: { model: demoModelSnapshot, simulation: simulationResult },
};
