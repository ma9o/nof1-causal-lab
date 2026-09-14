import { modelConstructs } from "@/lib/model-accessors";
import type { ModelSpec } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { demoModel } from "../../__fixtures__/demo-artifacts";
import { EdgeList } from "./edge-list";

const data = demoModel as ModelSpec;
const edges = data.edges;

const meta = {
  title: "Pipeline/Outputs/Latent Structure/EdgeList",
  component: EdgeList,
  args: { constructs: modelConstructs(data) },
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
