"use client";

import type { IndicatorSpec, InferenceReport } from "@nof1-causal-lab/api-types";
import { DiagnosticsAccordion } from "@/components/analysis-widgets/posterior/diagnostics-accordion";

export default function PosteriorView({
  data,
  indicators,
}: {
  data: InferenceReport;
  indicators: IndicatorSpec[];
}) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border p-3 text-sm">
        <p className="font-medium">Inference report</p>
        <p className="text-muted-foreground">
          {data.inference_metadata.n_samples.toLocaleString()} draws reported
        </p>
        <p className="text-xs text-muted-foreground">
          {data.inference_metadata.method.replaceAll("_", " ")}
        </p>
      </div>
      <DiagnosticsAccordion
        indicators={indicators}
        ppc={data.ppc}
        inferenceDiagnostics={data.inference_diagnostics}
        looDiagnostics={data.loo_diagnostics}
        posteriorMarginals={data.posterior_marginals}
        posteriorPairs={data.posterior_pairs}
      />
    </div>
  );
}
