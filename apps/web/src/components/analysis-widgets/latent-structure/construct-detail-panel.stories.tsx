import type { LatentStructureArtifact } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { demoLatentStructure } from "../../__fixtures__/demo-artifacts";
import { ConstructDetailPanel } from "./construct-detail-panel";

const data = demoLatentStructure as LatentStructureArtifact;
const constructs = data.latent_structure.constructs;
const endogenous = constructs.find((c) => c.role === "endogenous")!;
const exogenous = constructs.find((c) => c.role === "exogenous")!;
const outcome = constructs.find((c) => c.id === data.latent_structure.default_outcome?.id)!;

const meta = {
  title: "Pipeline/Outputs/Latent Structure/ConstructDetailPanel",
  component: ConstructDetailPanel,
  decorators: [withContainer("max-w-md")],
} satisfies Meta<typeof ConstructDetailPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Endogenous: Story = {
  args: { construct: endogenous },
};

export const Exogenous: Story = {
  args: { construct: exogenous },
};

export const Outcome: Story = {
  args: { construct: outcome, isOutcome: true },
};
