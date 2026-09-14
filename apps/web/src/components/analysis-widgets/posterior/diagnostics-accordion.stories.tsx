import { modelConstructs } from "@/lib/model-accessors";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
const indicators = modelConstructs(demoModelSnapshot.model!.value).flatMap(
  (construct) => construct.indicators,
);
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { posterior } from "@/components/__fixtures__/inference-data";
import { DiagnosticsAccordion } from "./diagnostics-accordion";

const meta = {
  args: { indicators },
  title: "Pipeline/Outputs/Posterior/DiagnosticsAccordion",
  component: DiagnosticsAccordion,
  decorators: [withContainer()],
} satisfies Meta<typeof DiagnosticsAccordion>;

export default meta;
type Story = StoryObj<typeof meta>;

export const InferenceOnly: Story = {
  args: {
    inferenceDiagnostics: posterior.inference_diagnostics,
  },
};

export const AllSections: Story = {
  args: {
    ppc: posterior.assessment.ppc,
    inferenceDiagnostics: posterior.inference_diagnostics,
    looDiagnostics: posterior.assessment.loo_diagnostics,
    posteriorMarginals: posterior.posterior_marginals,
    posteriorPairs: posterior.posterior_pairs,
  },
};

export const EngineDefinedFields: Story = {
  args: {
    inferenceDiagnostics: {
      warmup: { iterations: 120, step_sizes: [0.1, 0.05, 0.025] },
      particle_moves: { accepted: [true, false, true], unavailable_metric: null },
      message: "Engine telemetry",
    },
  },
};

export const Empty: Story = {};
