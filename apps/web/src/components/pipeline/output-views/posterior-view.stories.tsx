import { demoMeasurementStructure } from "@/components/__fixtures__/demo-artifacts";
const indicators = demoMeasurementStructure.causal_design.measurement.indicators;
import type { PosteriorArtifact } from "@nof1-causal-lab/api-types";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { Meta } from "@storybook/nextjs-vite";
import { demoPosterior } from "../../__fixtures__/demo-artifacts";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import PosteriorView from "./posterior-view";

const output = TRANSITIONS.find((s) => s.id === "posterior")!;
const data = demoPosterior as PosteriorArtifact;

const meta = {
  args: { indicators },
  title: "Pipeline/Outputs/Posterior/Panel",
  component: PosteriorView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof PosteriorView>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export const Running = createOutputStatusStory(output, "running");

export const CompletedParticleMCMC = createCompletedOutputStory({
  name: "Completed (marginal particle Gibbs)",
  output,
  args: { data, indicators, workspaceId: "demo-user" },
  elapsedMs: Math.round(data.inference_metadata.duration_seconds * 1000),
  renderContent: (args) => <PosteriorView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
