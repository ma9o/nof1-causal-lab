import { withContainer } from "@/components/story-decorators";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  indicators,
  likelihoodDiagnostics,
  likelihoods,
} from "./__fixtures__/statistical-model-spec-fixtures";
import { MeasurementTable } from "./measurement-table";

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/MeasurementTable",
  component: MeasurementTable,
  decorators: [withContainer()],
} satisfies Meta<typeof MeasurementTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { indicators, likelihoods, diagnostics: likelihoodDiagnostics },
};

export const WithPriorPredictive: Story = {
  args: { indicators, likelihoods, diagnostics: likelihoodDiagnostics },
};
