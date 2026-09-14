import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { buildSimulationScenarios } from "@/components/dag/simulation-results";
import { withContainer } from "@/components/story-decorators";
import { demoSimulationTrace } from "@/components/dag/__fixtures__/simulation-fixture";
import { ScenarioRail } from "./scenario-rail";

const scenarios = buildSimulationScenarios({ trace: demoSimulationTrace });

function RailDemo() {
  const [selected, setSelected] = useState<string | null>(scenarios[0]?.key ?? null);
  return <ScenarioRail scenarios={scenarios} selectedKey={selected} onSelect={setSelected} />;
}

const meta = {
  title: "Analysis/Scenario Rail",
  component: ScenarioRail,
  decorators: [withContainer("max-w-4xl")],
} satisfies Meta<typeof ScenarioRail>;

export default meta;

export const Default: StoryObj = {
  render: () => <RailDemo />,
};
