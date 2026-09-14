import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { ObsModelTable } from "./obs-model-table";
import {
  observationEquations,
  indicators,
  constructs,
  parameters,
  distributions,
} from "./__fixtures__/statistical-model-spec-fixtures";

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/ObsModelTable",
  component: ObsModelTable,
  args: { constructs, observationEquations },
  decorators: [withContainer()],
} satisfies Meta<typeof ObsModelTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithIndicators: Story = {
  args: { parameters, distributions, indicators },
};

export const WithoutIndicators: Story = {
  args: { parameters, distributions, indicators: [] },
};
