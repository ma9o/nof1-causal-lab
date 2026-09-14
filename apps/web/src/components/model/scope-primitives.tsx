import type { ArtifactId, FactSource, PosteriorEstimate } from "@nof1-causal-lab/api-types";
import type { ReactNode } from "react";
import { signColor } from "@/components/dag/core/palette";
import { cn } from "@/lib/utils";
import { formatPosteriorIntervalLabel } from "@/lib/utils/format";
import { formatPlain, formatSigned } from "./model-selection";

/** One section of a scope: a titled block that flows into the details pane's columns. */
export function Section({
  title,
  chips,
  wide = false,
  children,
}: {
  title: string;
  chips?: ReactNode;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      className={cn(
        "flex flex-none flex-col gap-1.5 border-t border-border pt-2",
        wide ? "w-[340px]" : "w-[232px]",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-1.5">
        <span className="text-xs font-semibold">{title}</span>
        {chips ? <span className="flex flex-wrap gap-1">{chips}</span> : null}
      </div>
      <div className="flex flex-col gap-1.5 text-[11.5px]">{children}</div>
    </section>
  );
}

function ReferenceChip({
  label,
  stale = false,
  retracted = false,
}: {
  label: string;
  stale?: boolean;
  retracted?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex h-[18px] items-center gap-1 whitespace-nowrap rounded-full border px-1.5 font-mono text-[9.5px]",
        retracted
          ? "border-dashed text-muted-foreground line-through"
          : stale
            ? "border-warning/70 bg-[repeating-linear-gradient(45deg,color-mix(in_oklab,var(--warning)_18%,transparent)_0_3px,transparent_3px_7px)]"
            : "border-foreground",
      )}
    >
      {label}
      {retracted ? " retracted" : stale ? " · stale" : ""}
    </span>
  );
}

/** The artifact revision that owns these facts. */
export function ArtifactChip({
  id,
  version,
  stale,
  retracted,
}: {
  id: ArtifactId;
  version: number | null;
  stale?: boolean;
  retracted?: boolean;
}) {
  return (
    <ReferenceChip
      label={`${id}${version != null ? ` v${version}` : ""}`}
      stale={stale}
      retracted={retracted}
    />
  );
}

/** Findings can be owned by a transition log or by an artifact. */
export function FactChip({ source }: { source: FactSource | undefined }) {
  if (!source) return null;
  const ref = source.ref;
  const label = "seq" in ref ? `move ${ref.seq}` : `${ref.artifact_id} v${ref.version}`;
  return <ReferenceChip label={label} stale={source.validity === "stale"} />;
}

export function KeyValue({ rows }: { rows: Array<[string, ReactNode]> }) {
  return (
    <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-2.5 gap-y-0.5 text-[11px]">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="text-muted-foreground">{key}</dt>
          <dd className="m-0 min-w-0">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Hint({ children, issue = false }: { children: ReactNode; issue?: boolean }) {
  return (
    <p
      className={cn(
        "m-0 text-[11px] leading-snug text-pretty",
        issue ? "text-warning-foreground" : "text-muted-foreground",
      )}
    >
      {children}
    </p>
  );
}

export function Prose({ children }: { children: ReactNode }) {
  return <p className="m-0 leading-snug text-pretty text-foreground">{children}</p>;
}

export function Callout({ tone, children }: { tone: "ok" | "bad" | "warn"; children: ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-lg border px-2.5 py-2 text-[11px] leading-snug",
        tone === "ok" && "border-success/25 bg-success/8",
        tone === "bad" && "border-destructive/30 bg-destructive/6",
        tone === "warn" && "border-warning/40 bg-warning/12 text-warning-foreground",
      )}
    >
      {children}
    </div>
  );
}

export function Tag({
  children,
  tone = "outline",
}: {
  children: ReactNode;
  tone?: "outline" | "secondary" | "success" | "warning" | "destructive";
}) {
  return (
    <span
      className={cn(
        "inline-flex h-4 items-center whitespace-nowrap rounded-full border border-transparent px-1.5 text-[10px] font-medium",
        tone === "outline" && "border-border",
        tone === "secondary" && "bg-secondary",
        tone === "success" && "bg-success/10 text-success",
        tone === "warning" && "bg-warning/15 text-warning-foreground",
        tone === "destructive" && "bg-destructive/10 text-destructive",
      )}
    >
      {children}
    </span>
  );
}

/** A link to another owner of the asset: selecting it swaps the pane to that owner. */
export function OwnerLink({
  onClick,
  children,
  mono = true,
}: {
  onClick: () => void;
  children: ReactNode;
  mono?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "cursor-pointer truncate text-left underline underline-offset-[3px] hover:text-muted-foreground",
        mono && "font-mono text-[10.5px]",
      )}
    >
      {children}
    </button>
  );
}

export interface PriorRow {
  parameter: string;
  role: string;
  prior: string | null;
}

export function PriorTable({ rows }: { rows: PriorRow[] }) {
  if (rows.length === 0) return null;
  return (
    <table className="w-full table-fixed border-collapse text-[10px]">
      <thead>
        <tr className="text-left text-muted-foreground">
          <th className="border-b pb-0.5 pr-1.5 font-medium">parameter</th>
          <th className="border-b pb-0.5 pr-1.5 font-medium">role</th>
          <th className="border-b pb-0.5 font-medium">distribution</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.parameter}>
            <td className="truncate border-b py-0.5 pr-1.5 font-mono">{row.parameter}</td>
            <td className="truncate border-b py-0.5 pr-1.5">{row.role}</td>
            <td className="truncate border-b py-0.5 font-mono">{row.prior ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export interface PosteriorRow extends PosteriorEstimate {
  parameter: string;
}

export function PosteriorTable({ rows }: { rows: PosteriorRow[] }) {
  if (rows.length === 0) return null;
  return (
    <table className="w-full table-fixed border-collapse text-[10px]">
      <thead>
        <tr className="text-left text-muted-foreground">
          <th className="border-b pb-0.5 pr-1.5 font-medium">parameter</th>
          <th className="border-b pb-0.5 pr-1.5 font-medium">mean</th>
          <th className="border-b pb-0.5 pr-1.5 font-medium">interval</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.parameter}>
            <td className="truncate border-b py-0.5 pr-1.5 font-mono">{row.parameter}</td>
            <td
              className="truncate border-b py-0.5 pr-1.5 font-mono"
              style={{ color: signColor(row.mean) }}
            >
              {formatSigned(row.mean)}
            </td>
            <td
              className="truncate border-b py-0.5 pr-1.5 font-mono"
              title={formatPosteriorIntervalLabel(row)}
            >
              [{formatPlain(row.lower)}, {formatPlain(row.upper)}] ·{" "}
              {formatPosteriorIntervalLabel(row)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
