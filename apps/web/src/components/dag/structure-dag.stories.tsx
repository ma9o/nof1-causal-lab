import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { constructs, edges, indicators } from "./__fixtures__/dag-base-fixtures";
import { constructStatuses } from "./construct-statuses";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { StructureDag } from "./structure-dag";

const nodeStatuses = constructStatuses(demoModelSnapshot);

const meta: Meta<typeof StructureDag> = {
  title: "DAG/Structure DAG",
  component: StructureDag,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof StructureDag>;

/** latent-structure — latent structure: clickable constructs, no indicators or statuses. */
export const LatentStructureStory: Story = {
  render: () => {
    const [selected, setSelected] = useState<string | null>(null);
    return (
      <div className="p-4">
        <StructureDag constructs={constructs} edges={edges} onNodeClick={setSelected} />
        <p className="mt-2 text-sm text-muted-foreground">selected: {selected ?? "—"}</p>
      </div>
    );
  },
};

/** Canonical causal-design fixture with measurement and backend dispositions overlaid. */
export const MeasurementStructureStory: Story = {
  render: () => (
    <div className="p-4">
      <StructureDag
        constructs={constructs}
        edges={edges}
        indicators={indicators}
        nodeStatuses={nodeStatuses}
      />
    </div>
  ),
};
