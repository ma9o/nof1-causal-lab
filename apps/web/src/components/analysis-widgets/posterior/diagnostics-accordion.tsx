"use client";

import type {
  IndicatorSpec,
  JsonObject,
  LOODiagnostics,
  PosteriorMarginal,
  PosteriorPair,
  PosteriorPredictiveChecks,
} from "@nof1-causal-lab/api-types";
import { LOOPITChart } from "@/components/charts/loo-pit-chart";
import { ParetoKChart } from "@/components/charts/pareto-k-chart";
import { PosteriorDensityChart } from "@/components/charts/posterior-density-chart";
import { PosteriorPairsChart } from "@/components/charts/posterior-pairs-chart";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { JsonViewer } from "@/components/ui/json-viewer";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { VirtualizedChartGrid } from "@/components/ui/virtualized-chart-grid";
import { formatNumber } from "@/lib/utils/format";
import { PPCWarningsTable } from "./ppc-warnings-table";

interface DiagnosticsAccordionProps {
  indicators: IndicatorSpec[];
  ppc?: PosteriorPredictiveChecks | null;
  inferenceDiagnostics?: JsonObject;
  looDiagnostics?: LOODiagnostics | null;
  posteriorMarginals?: PosteriorMarginal[] | null;
  posteriorPairs?: PosteriorPair[] | null;
}

export function DiagnosticsAccordion({
  indicators,
  ppc,
  inferenceDiagnostics,
  looDiagnostics,
  posteriorMarginals,
  posteriorPairs,
}: DiagnosticsAccordionProps) {
  const marginals = posteriorMarginals ?? [];
  const pairs = posteriorPairs ?? [];
  const hasMarginals = marginals.length > 0;
  const hasPairs = pairs.length > 0;
  const hasPPC = ppc != null && ppc.per_variable_warnings.length > 0;

  const defaultOpen = ["inference", "ppc", "loo"];

  return (
    <Accordion defaultValue={defaultOpen} multiple>
      {inferenceDiagnostics && Object.keys(inferenceDiagnostics).length > 0 && (
        <AccordionItem value="inference">
          <AccordionTrigger className="text-sm">Inference diagnostics</AccordionTrigger>
          <AccordionContent>
            <JsonViewer data={inferenceDiagnostics} />
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── Posterior Predictive Checks (warnings + overlays + test stats) ── */}
      {hasPPC && ppc && (
        <AccordionItem value="ppc">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              Posterior Predictive Checks
              <StatTooltip explanation="Checks whether the fitted model can reproduce aspects of the observed data (distributional shape, variance, autocorrelation). Passing does not validate causal structure — only that the statistical model is not grossly misspecified." />
              <Badge
                variant={ppc.per_variable_warnings.every((w) => w.passed) ? "success" : "warning"}
              >
                {ppc.per_variable_warnings.every((w) => w.passed)
                  ? "Consistent"
                  : "Misfit detected"}
              </Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-6">
              <PPCWarningsTable
                indicators={indicators}
                warnings={ppc.per_variable_warnings}
                testStats={ppc.test_stats ?? []}
                overlays={ppc.overlays ?? []}
              />
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── LOO Cross-Validation (PIT + Pareto-K side by side) ── */}
      {looDiagnostics && (
        <AccordionItem value="loo">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              LOO Cross-Validation
              <StatTooltip explanation="PSIS estimates how well the model interpolates a held-out measurement row using all other rows, including future measurements. Each row contains all observed indicators at that time. Pareto-k values assess whether the approximation is reliable." />
              <Badge
                variant={
                  looDiagnostics.n_bad_k == null
                    ? "outline"
                    : looDiagnostics.n_bad_k === 0
                      ? "success"
                      : "warning"
                }
              >
                ELPD = {formatNumber(looDiagnostics.elpd_loo, 1)}
              </Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="outline">ELPD = {formatNumber(looDiagnostics.elpd_loo, 1)}</Badge>
                <Badge variant="outline">p_loo = {formatNumber(looDiagnostics.p_loo, 1)}</Badge>
                <Badge variant="outline">SE = {formatNumber(looDiagnostics.se, 1)}</Badge>
                {looDiagnostics.n_bad_k != null && (
                  <Badge variant={looDiagnostics.n_bad_k === 0 ? "success" : "destructive"}>
                    {looDiagnostics.n_bad_k === 0
                      ? "All Pareto k OK"
                      : `${looDiagnostics.n_bad_k} bad Pareto k`}
                  </Badge>
                )}
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {looDiagnostics.loo_pit && <LOOPITChart loo={looDiagnostics} />}
                {looDiagnostics.pareto_k && <ParetoKChart loo={looDiagnostics} />}
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── Posterior Exploration (marginals + pairs) ── */}
      {(hasMarginals || hasPairs) && (
        <AccordionItem value="posteriors">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              Posterior Exploration
              <StatTooltip explanation="Marginal posterior densities with 94% HDI, and pairwise scatter plots revealing parameter correlations and identifiability issues." />
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4">
              {hasMarginals && (
                <div>
                  <h4 className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Marginal distributions
                  </h4>
                  <VirtualizedChartGrid
                    items={marginals}
                    estimateRowHeight={152}
                    maxHeight={456}
                    renderItem={(m) => <PosteriorDensityChart marginal={m} />}
                    keyExtractor={(m) => m.parameter}
                  />
                </div>
              )}
              {hasPairs && (
                <div>
                  <h4 className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Pairwise correlations
                  </h4>
                  <VirtualizedChartGrid
                    items={pairs}
                    estimateRowHeight={180}
                    maxHeight={540}
                    renderItem={(p) => <PosteriorPairsChart pair={p} />}
                    keyExtractor={(p) => `${p.param_x}-${p.param_y}`}
                  />
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  );
}
