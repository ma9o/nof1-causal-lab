import type { BaselineReportArtifact } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { demoBaselineReport } from "../../__fixtures__/demo-artifacts";
import { TreatmentRankingTable } from "./treatment-ranking-table";

const data = demoBaselineReport as BaselineReportArtifact;

const meta = {
  title: "Pipeline/Outputs/Posterior/TreatmentRankingTable",
  component: TreatmentRankingTable,
  decorators: [withContainer()],
} satisfies Meta<typeof TreatmentRankingTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { results: data.intervention_results },
};
