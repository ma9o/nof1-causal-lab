"use client";

import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { collectModelSpecObservationPriorTerms } from "@/lib/model-spec-data";
import {
  observationEquationLatex,
  observationParameterSymbol,
  observationPriorLatex,
} from "@/lib/utils/ssm-latex";
import type { Indicator, LikelihoodSpec, ParameterSpec } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import katex from "katex";

// ── helpers ──────────────────────────────────────────────

function inlineKatex(latex: string): string {
  return katex.renderToString(latex, { displayMode: false, throwOnError: false, strict: false });
}

// ── row type ─────────────────────────────────────────────

interface ObsModelRow {
  likelihood: import("@/lib/utils/ssm-latex").LabeledLikelihood;
  variable: string;
  construct: string | undefined;
  equationLatex: string;
  loadingFixed: boolean;
  priorTerms: ParameterSpec[];
}

// ── columns ──────────────────────────────────────────────

const col = createColumnHelper<ObsModelRow>();

const columns = [
  col.accessor("variable", {
    header: "Variable",
    cell: (info) => <span className="font-medium font-mono text-xs">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  col.accessor("construct", {
    header: "Latent",
    cell: (info) => {
      const v = info.getValue();
      return v ? (
        <span className="font-mono text-xs">{v}</span>
      ) : (
        <span className="text-muted-foreground">—</span>
      );
    },
    meta: { mono: true },
  }),
  col.display({
    id: "equation",
    header: "Equation",
    cell: ({ row }) => (
      // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
      <span dangerouslySetInnerHTML={{ __html: inlineKatex(row.original.equationLatex) }} />
    ),
    enableSorting: false,
  }),
  col.accessor("loadingFixed", {
    header: () => (
      <HeaderWithTooltip
        label="Loading"
        tooltip="Whether the factor loading λ is fixed to 1 (reference indicator for scale identification) or freely estimated with a prior."
      />
    ),
    cell: ({ row }) => {
      const { construct, loadingFixed } = row.original;
      if (!construct) return <span className="text-muted-foreground">—</span>;
      return loadingFixed ? (
        // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
        <span dangerouslySetInnerHTML={{ __html: inlineKatex("= 1") }} />
      ) : (
        <span className="text-muted-foreground">estimated</span>
      );
    },
  }),
  col.display({
    id: "priors",
    header: "Priors",
    cell: ({ row }) => (
      <ObsPriorList likelihood={row.original.likelihood} terms={row.original.priorTerms} />
    ),
    enableSorting: false,
  }),
] as ColumnDef<ObsModelRow, unknown>[];

export function ObsPriorList({
  likelihood,
  terms,
}: {
  likelihood: import("@/lib/utils/ssm-latex").LabeledLikelihood;
  terms: ParameterSpec[];
}) {
  if (terms.length === 0) {
    return <span className="text-xs text-muted-foreground">Not authored</span>;
  }
  return (
    <div className="space-y-1">
      {terms.map((term) => (
        <div
          key={term.name}
          className="text-muted-foreground"
          // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
          dangerouslySetInnerHTML={{
            __html: inlineKatex(
              term.prior
                ? observationPriorLatex({
                    prior: term.prior,
                    parameterName: term.name,
                    likelihood,
                  })
                : `${observationParameterSymbol({ parameterName: term.name, likelihood })}:\\ \\text{Not authored}`,
            ),
          }}
        />
      ))}
    </div>
  );
}

// ── component ────────────────────────────────────────────

export function ObsModelTable({
  likelihoods,
  parameters,
  indicators,
  constructs,
}: {
  likelihoods: LikelihoodSpec[];
  parameters: ParameterSpec[];
  indicators?: Indicator[];
  constructs: import("@nof1-causal-lab/api-types").Construct[];
}) {
  const rows: ObsModelRow[] = likelihoods.map((lik) => {
    const indicator = indicators?.find((item) => item.id === lik.indicator_id);
    const construct = constructs.find((item) => item.id === indicator?.construct_id)?.name;
    const v = indicator?.name ?? lik.indicator_id;
    const labeledLikelihood = { ...lik, label: v };
    const priorTerms = collectModelSpecObservationPriorTerms({
      likelihood: lik,
      parameters,
    });
    const hasLoadingParam = priorTerms.some((term) =>
      parameters.some((parameter) => parameter.name === term.name && parameter.role === "loading"),
    );
    const loadingFixed = indicator != null && !hasLoadingParam;
    return {
      likelihood: labeledLikelihood,
      variable: v,
      construct,
      equationLatex: observationEquationLatex({
        likelihood: labeledLikelihood,
        constructName: construct,
        parameterNames: priorTerms.map((term) => term.name),
      }).replace(/&/g, ""),
      loadingFixed,
      priorTerms,
    };
  });

  return <InfoTable columns={columns} data={rows} estimateRowHeight={48} />;
}
