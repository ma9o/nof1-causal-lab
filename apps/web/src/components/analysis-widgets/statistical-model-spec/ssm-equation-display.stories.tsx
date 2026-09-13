import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import {
  equations,
  indicators,
  likelihoods,
  parameters,
  priors,
  structuralPlan,
} from "./__fixtures__/statistical-model-spec-fixtures";
import { SSMEquationDisplay } from "./ssm-equation-display";

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/SSMEquationDisplay",
  component: SSMEquationDisplay,
  args: { structuralPlan, equations },
  decorators: [withContainer()],
} satisfies Meta<typeof SSMEquationDisplay>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { likelihoods, parameters, priors, indicators },
};

export const WithoutIndicators: Story = {
  args: { likelihoods, parameters, priors },
};
