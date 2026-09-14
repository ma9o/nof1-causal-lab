import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { PriorTable } from "./prior-table";
import { parameters, priorDensities } from "./__fixtures__/statistical-model-spec-fixtures";

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/PriorTable",
  component: PriorTable,
  decorators: [withContainer()],
} satisfies Meta<typeof PriorTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithSearchContext: Story = {
  args: { parameters, densities: priorDensities },
};

export const WithoutSearchContext: Story = {
  args: { parameters, densities: priorDensities },
};
