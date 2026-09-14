import { modelConstructs } from "@/lib/model-accessors";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { IndicatorTable } from "./indicator-table";
import { demoModelSnapshot } from "../../__fixtures__/demo-artifacts";

const data = demoModelSnapshot as ModelSnapshot;
const indicators = modelConstructs(data.model!.value).flatMap((construct) => construct.indicators);

const meta = {
  title: "Pipeline/Outputs/Measurement Structure/IndicatorTable",
  component: IndicatorTable,
  args: { constructs: modelConstructs(data.model!.value) },
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof IndicatorTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { indicators },
};

export const Empty: Story = {
  args: { indicators: [] },
};
