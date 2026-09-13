import type {
  Indicator,
  LikelihoodSpec,
  ParameterSpec,
  StateEquation,
} from "@nof1-causal-lab/api-types";
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
import { confounderGroupLatex, confounderGroups } from "@/lib/utils/ssm-latex";

interface SsmEquationDisplayProps {
  likelihoods: LikelihoodSpec[];
  equations: StateEquation[];
  parameters: ParameterSpec[];
  indicators?: Indicator[];
  structuralPlan: import("@nof1-causal-lab/api-types").StructuralPlan;
}

/** Render a LaTeX string to an HTML string via KaTeX. */
function tex(latex: string, displayMode = true): string {
  return katex.renderToString(latex, {
    displayMode,
    throwOnError: false,
    strict: false,
  });
}

/** Render a confounder group's LaTeX to HTML via KaTeX. */
function confounderGroupHtml(group: Parameters<typeof confounderGroupLatex>[0]): string {
  return tex(confounderGroupLatex(group));
}

/** Inline KaTeX span. */
function Katex({ latex }: { latex: string }) {
  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
  return <span dangerouslySetInnerHTML={{ __html: tex(latex, false) }} />;
}

export function SSMEquationDisplay({
  equations,
  likelihoods,
  parameters,
  indicators,
  structuralPlan,
}: SsmEquationDisplayProps) {
  const corrGroups = confounderGroups(structuralPlan);
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
      {corrGroups && (
        <section>
          <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Marginalized Confounders
            <StatTooltip explanation="Marginalized time-varying confounders induce shared process noise among these states." />
          </h4>
          <div className="space-y-3">
            {corrGroups.map((group) => (
              <div
                key={group.confounder}
                className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3"
              >
                <div
                  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
                  dangerouslySetInnerHTML={{
                    __html: confounderGroupHtml(group),
                  }}
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Observation model ── */}
      {likelihoods.length > 0 && (
        <section>
          <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Observation Model
            <StatTooltip explanation="Maps latent states to observed indicators. Each variable has a distribution family (e.g. Gaussian, Poisson) and a link function (e.g. identity, log, logit) that transforms the linear predictor λᵀη(t) to the distribution's natural parameter." />
          </h4>
          <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Semantic model-spec Form
            </p>
            <div
              // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
              dangerouslySetInnerHTML={{
                __html: tex(
                  String.raw`\begin{aligned}
\mu_k(t) &= \boldsymbol{\lambda}_k^\top \boldsymbol{\eta}(t) \\[4pt]
\mathbb{E}[y_k(t)] &= g_k^{-1}\!\bigl(\mu_k(t)\bigr), \quad y_k(t) \sim \mathcal{F}_k
\end{aligned}`,
                ),
              }}
            />
          </div>
          <div className="mt-3">
            <ObsModelTable
              likelihoods={likelihoods}
              parameters={parameters}
              indicators={indicators}
              constructs={Object.values(structuralPlan.semantics.constructs)}
            />
          </div>
        </section>
      )}
    </div>
  );
}
