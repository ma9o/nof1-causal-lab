import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { StaleResultsBannerView } from "./stale-results-banner";

const meta = {
  title: "V1/Pipeline/StaleResultsBanner",
  component: StaleResultsBannerView,
  decorators: [withContainer("max-w-4xl")],
} satisfies Meta<typeof StaleResultsBannerView>;
export default meta;
type Story = StoryObj<typeof meta>;
export const Stale: Story = { args: { staleTransitionCount: 3 } };
export const SingleArtifact: Story = { args: { staleTransitionCount: 1 } };
