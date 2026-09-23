import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  demoSimulationTrace,
  edgePosteriors,
  identifiableTreatments,
  nodeStatuses,
  persistencePosteriors,
} from "@/components/dag/__fixtures__/simulation-fixture";
import { constructs, edges, indicators } from "../__fixtures__/dag-base-fixtures";
import { buildSimulationScenarios } from "@/lib/dag/simulation-results";
import { InteractiveDag } from "./interactive-dag";

const scenarios = buildSimulationScenarios({ trace: demoSimulationTrace });
const firstScenario = scenarios[0]?.result;
const comparisonScenario = scenarios[scenarios.length - 1]?.result;

if (!firstScenario || !comparisonScenario) {
  throw new Error("The DEMO trace must contain materialized simulation scenarios.");
}

const graphArgs = {
  constructs,
  edges,
  indicators,

  edgePosteriors,
  persistencePosteriors,
  identifiableTreatments,
  nodeStatuses,
};

const meta: Meta<typeof InteractiveDag> = {
  title: "DAG/Interactive/Living DAG",
  component: InteractiveDag,
  parameters: { layout: "fullscreen" },
  decorators: [
    (Story) => (
      <div style={{ background: "#fafbfc", minHeight: "100vh" }}>
        <div style={{ maxWidth: 1320, margin: "0 auto", padding: "18px 20px 60px" }}>
          <Story />
        </div>
      </div>
    ),
  ],
};
export default meta;

type Story = StoryObj<typeof InteractiveDag>;

/** A production-shaped result with its backend reference and action rollouts. */
export const ReferenceAndAction: Story = {
  args: { ...graphArgs, result: firstScenario },
};

/** Read-only living DAG with an intervention applied (no do() editing). */
export const Intervention: Story = {
  args: { ...graphArgs, result: comparisonScenario },
};
