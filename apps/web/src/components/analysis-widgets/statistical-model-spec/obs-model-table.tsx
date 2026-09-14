"use client";

import type { IndicatorSpec, ParameterSpec } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import katex from "katex";
import { InfoTable } from "@/components/ui/info-table";
import { collectModelSpecObservationPriorTerms } from "@/lib/model-spec-data";
import { observationParameterSymbol, observationPriorLatex } from "@/lib/utils/ssm-latex";

// ── helpers ──────────────────────────────────────────────

function inlineKatex(latex: string): string {
  return katex.renderToString(latex, { displayMode: false, throwOnError: false, strict: false });
}

// ── row type ─────────────────────────────────────────────

interface ObsModelRow {
  distributions: import("@nof1-causal-lab/api-types").ModelSpec["distributions"];
  variable: string;
  construct: string | undefined;
  equationLatex: string;
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
  col.display({
    id: "priors",
    header: "Priors",
    cell: ({ row }) => (
      <ObsPriorList terms={row.original.priorTerms} distributions={row.original.distributions} />
    ),
    enableSorting: false,
  }),
] as ColumnDef<ObsModelRow, unknown>[];

export function ObsPriorList({
  terms,
  distributions,
}: {
  terms: ParameterSpec[];
  distributions: import("@nof1-causal-lab/api-types").ModelSpec["distributions"];
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
              term.value != null
                ? `${observationParameterSymbol({ parameterName: term.name })} = ${term.value}`
                : term.distribution
                  ? observationPriorLatex({
                      prior: distributions[term.distribution],
                      parameterName: term.name,
                    })
                  : `${observationParameterSymbol({ parameterName: term.name })}:\\ \\text{Not authored}`,
            ),
          }}
        />
      ))}
    </div>
  );
}

// ── component ────────────────────────────────────────────

export function ObsModelTable({
  parameters,
  distributions,
  indicators,
  constructs,
  observationEquations,
}: {
  parameters: ParameterSpec[];
  distributions: import("@nof1-causal-lab/api-types").ModelSpec["distributions"];
  indicators: IndicatorSpec[];
  constructs: import("@nof1-causal-lab/api-types").ConstructSpec[];
  observationEquations: Record<string, string>;
}) {
  const rows: ObsModelRow[] = indicators.flatMap((indicator) => {
    const lik = indicator.likelihood;
    if (!lik) return [];
    const construct = constructs.find((item) =>
      item.indicators.some((owned) => owned.id === indicator.id),
    )?.name;
    const v = indicator.name;
    const priorTerms = collectModelSpecObservationPriorTerms({ indicator, parameters });
    return {
      variable: v,
      construct,
      equationLatex: observationEquations[indicator.id],
      priorTerms,
      distributions,
    };
  });

  return <InfoTable columns={columns} data={rows} estimateRowHeight={48} />;
}
