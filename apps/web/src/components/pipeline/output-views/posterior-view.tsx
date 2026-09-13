"use client";

import type { Indicator, PosteriorArtifact } from "@nof1-causal-lab/api-types";
import { useState } from "react";
import { DiagnosticsAccordion } from "@/components/analysis-widgets/posterior/diagnostics-accordion";
import { MockMethodSwitcher } from "@/components/analysis-widgets/posterior/mock-method-switcher";
import { isMockMode } from "@/lib/api/mock-provider";

export default function PosteriorView({
  workspaceId,
  data,
  indicators,
}: {
  workspaceId: string;
  data: PosteriorArtifact;
  indicators: Indicator[];
}) {
  const [activeData, setActiveData] = useState(data);
  const mock = isMockMode();

  return (
    <div className="space-y-4">
      {mock && (
        <MockMethodSwitcher
          workspaceId={workspaceId}
          baseData={data}
          onDataChange={setActiveData}
        />
      )}
      <div className="rounded-lg border p-3 text-sm">
        <p className="font-medium">Joint posterior</p>
        <p className="text-muted-foreground">
          {activeData.draws.n_draws.toLocaleString()} aligned draws
          {activeData.draws.latent_shape &&
            ` · ${activeData.draws.latent_shape[1]} latent states at ${activeData.draws.latent_shape[0]} times`}
        </p>
        <p className="text-xs text-muted-foreground">
          Model v{activeData.provenance.compiled_ssm_version} · observations v
          {activeData.provenance.panel_version}
        </p>
      </div>
      <DiagnosticsAccordion
        indicators={indicators}
        ppc={activeData.assessment.ppc}
        mcmcDiagnostics={activeData.assessment.mcmc_diagnostics}
        smcDiagnostics={activeData.assessment.smc_diagnostics}
        looDiagnostics={activeData.assessment.loo_diagnostics}
        posteriorMarginals={activeData.posterior_marginals}
        posteriorPairs={activeData.posterior_pairs}
      />
    </div>
  );
}
