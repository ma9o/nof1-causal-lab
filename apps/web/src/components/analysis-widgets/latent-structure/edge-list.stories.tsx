import type { LatentStructureArtifact } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { demoLatentStructure } from "../../__fixtures__/demo-artifacts";
import { EdgeList } from "./edge-list";

const data = demoLatentStructure as LatentStructureArtifact;
const edges = data.latent_structure.edges;

const meta = {
  title: "Pipeline/Outputs/Latent Structure/EdgeList",
  component: EdgeList,
  args: { constructs: data.latent_structure.constructs },
  decorators: [withContainer("max-w-md")],
} satisfies Meta<typeof EdgeList>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { edges },
};

export const Empty: Story = {
  args: { edges: [] },
};
