import { modelConstructs } from "@/lib/model-accessors";
import type { Indicator, ParameterSpec, StateEquation } from "@nof1-causal-lab/api-types";
import katex from "katex";
import { FunctionalSpecLink } from "@/components/analysis-widgets/statistical-model-spec/functional-spec-link";
import { ObsModelTable } from "@/components/analysis-widgets/statistical-model-spec/obs-model-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface SsmEquationDisplayProps {
  confounderEquations: StateEquation[];
  observationEquations: Record<string, string>;
  equations: StateEquation[];
  parameters: ParameterSpec[];
  indicators: Indicator[];
  model: import("@nof1-causal-lab/api-types").ModelSpec;
}

/** Render a LaTeX string to an HTML string via KaTeX. */
function tex(latex: string, displayMode = true): string {
  return katex.renderToString(latex, {
    displayMode,
    throwOnError: false,
    strict: false,
  });
}

/** Inline KaTeX span. */
function Katex({ latex }: { latex: string }) {
  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
  return <span dangerouslySetInnerHTML={{ __html: tex(latex, false) }} />;
}

export function SSMEquationDisplay({
  equations,
  confounderEquations,
  observationEquations,
  parameters,
  indicators,
  model,
}: SsmEquationDisplayProps) {
  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <h3 className="text-sm font-semibold">Model Equations</h3>
          <p className="max-w-3xl text-sm text-muted-foreground">
            Continuous-time dynamics from the declared mechanisms. Each θ label refers to the
            parameter in the prior table. Δ is that prior&apos;s reference interval in days, or the
            model interval when no reference interval is specified. L is the process-noise Cholesky
            factor and W is standard Brownian motion.
          </p>
        </div>
        <FunctionalSpecLink />
      </div>
      {equations.length > 0 && (
        <section className="overflow-x-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>State</TableHead>
                <TableHead>Continuous-time dynamics</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {equations.map((row) => (
                <TableRow key={row.construct_id}>
                  <TableCell className="whitespace-nowrap align-top">{row.label}</TableCell>
                  <TableCell className="align-top">
                    <Katex latex={row.latex} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </section>
      )}

      {/* ── Correlated errors (per marginalized confounder) ── */}
      {confounderEquations.length > 0 && (
        <section>
          <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Marginalized Confounders
            <StatTooltip explanation="Marginalized time-varying confounders induce shared process noise among these states." />
          </h4>
          <div className="space-y-3">
            {confounderEquations.map((group) => (
              <div
                key={group.construct_id}
                className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3"
              >
                <div
                  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
                  dangerouslySetInnerHTML={{
                    __html: tex(group.latex),
                  }}
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Observation model ── */}
      {indicators.some((indicator) => indicator.likelihood) && (
        <section>
          <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Observation Model
            <StatTooltip explanation="Maps latent states to observed indicators. Each conditional distribution is rendered from its declared expressions over scientific states and parameters." />
          </h4>
          <div className="mt-3">
            <ObsModelTable
              observationEquations={observationEquations}
              parameters={parameters}
              indicators={indicators}
              constructs={modelConstructs(model)}
            />
          </div>
        </section>
      )}
    </div>
  );
}
