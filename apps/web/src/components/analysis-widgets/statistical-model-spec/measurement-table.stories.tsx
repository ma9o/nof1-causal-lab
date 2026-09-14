import { withContainer } from "@/components/story-decorators";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { indicators, likelihoodDiagnostics } from "./__fixtures__/statistical-model-spec-fixtures";
import { MeasurementTable } from "./measurement-table";

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/MeasurementTable",
  component: MeasurementTable,
  decorators: [withContainer()],
} satisfies Meta<typeof MeasurementTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { indicators, diagnostics: likelihoodDiagnostics },
};

export const WithPriorPredictive: Story = {
  args: { indicators, diagnostics: likelihoodDiagnostics },
};
