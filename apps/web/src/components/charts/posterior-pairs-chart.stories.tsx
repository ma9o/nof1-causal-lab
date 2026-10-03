import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { posterior } from "@/components/__fixtures__/inference-data";
import { PosteriorPairsChart } from "./posterior-pairs-chart";

const [xRef, yRef] = fixtureValue(posterior.detail.posterior_pairs[0]);
const x = fixtureValue(
  posterior.detail.trace_data.find((column) => column.subject.element_id === xRef.element_id),
);
const y = fixtureValue(
  posterior.detail.trace_data.find((column) => column.subject.element_id === yRef.element_id),
);

const meta = {
  title: "Charts/PosteriorPairsChart",
  component: PosteriorPairsChart,
  decorators: [withContainer("max-w-sm")],
} satisfies Meta<typeof PosteriorPairsChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    x: { label: x.parameter, values: x.chains.flat() },
    y: { label: y.parameter, values: y.chains.flat() },
    divergent: posterior.detail.divergent,
  },
};
