import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { PosteriorPairsChart } from "./posterior-pairs-chart";

const meta = {
  title: "Charts/PosteriorPairsChart",
  component: PosteriorPairsChart,
  decorators: [withContainer("max-w-sm")],
} satisfies Meta<typeof PosteriorPairsChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    x: { label: "No retained posterior", values: [] },
    y: { label: "No retained posterior", values: [] },
    divergent: null,
  },
};
