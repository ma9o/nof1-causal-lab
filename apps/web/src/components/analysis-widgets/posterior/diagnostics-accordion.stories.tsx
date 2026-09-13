import { demoMeasurementStructure } from "@/components/__fixtures__/demo-artifacts";
const indicators = demoMeasurementStructure.causal_design.measurement.indicators;
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { posterior, posteriorAuxKalmanMCMC } from "@/components/__fixtures__/inference-data";
import { DiagnosticsAccordion } from "./diagnostics-accordion";

const meta = {
  args: { indicators },
  title: "Pipeline/Outputs/Posterior/DiagnosticsAccordion",
  component: DiagnosticsAccordion,
  decorators: [withContainer()],
} satisfies Meta<typeof DiagnosticsAccordion>;

export default meta;
type Story = StoryObj<typeof meta>;

export const MCMCOnly: Story = {
  args: {
    mcmcDiagnostics: posteriorAuxKalmanMCMC.assessment.mcmc_diagnostics,
    posteriorMarginals: posteriorAuxKalmanMCMC.posterior_marginals,
    posteriorPairs: posteriorAuxKalmanMCMC.posterior_pairs,
  },
};

export const AllSections: Story = {
  args: {
    ppc: posteriorAuxKalmanMCMC.assessment.ppc,
    mcmcDiagnostics: posteriorAuxKalmanMCMC.assessment.mcmc_diagnostics,
    looDiagnostics: posteriorAuxKalmanMCMC.assessment.loo_diagnostics,
    posteriorMarginals: posteriorAuxKalmanMCMC.posterior_marginals,
    posteriorPairs: posteriorAuxKalmanMCMC.posterior_pairs,
  },
};

export const ParticleDiagnosticsWithLOO: Story = {
  args: {
    smcDiagnostics: posterior.assessment.smc_diagnostics,
    looDiagnostics: posterior.assessment.loo_diagnostics,
    ppc: posterior.assessment.ppc,
    posteriorMarginals: posterior.posterior_marginals,
    posteriorPairs: posterior.posterior_pairs,
  },
};

export const Empty: Story = {};
