import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import {
  observationEquations,
  equations,
  indicators,
  parameters,
  model,
  confounderEquations,
} from "./__fixtures__/statistical-model-spec-fixtures";
import { SSMEquationDisplay } from "./ssm-equation-display";

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/SSMEquationDisplay",
  component: SSMEquationDisplay,
  args: { model, equations, confounderEquations, observationEquations },
  decorators: [withContainer()],
} satisfies Meta<typeof SSMEquationDisplay>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { parameters, indicators },
};

export const WithoutIndicators: Story = {
  args: { parameters, indicators: [] },
};
