import type { FactSource, ParameterSpec } from "@nof1-causal-lab/api-types";
import { Check, CircleDashed, ClockAlert, TriangleAlert, X } from "lucide-react";
import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { signColor } from "@/lib/dag/palette";
import type { PosteriorRow } from "@/lib/model-asset/inspector";
import {
  formatPlain,
  formatSigned,
  humanize,
  type ModelSelection,
} from "@/lib/model-asset/selection";
import { cn } from "@/lib/utils";
import { formatPosteriorIntervalLabel } from "@/lib/utils/format";

const STATUS_PRESENTATION = {
  passed: { icon: Check, className: "text-success", label: "Passed" },
  failed: { icon: X, className: "text-destructive", label: "Failed" },
  warning: { icon: TriangleAlert, className: "text-warning-foreground", label: "Warning" },
  not_evaluated: { icon: CircleDashed, className: "text-muted-foreground", label: "Not evaluated" },
  stale: {
    icon: ClockAlert,
    className: "text-warning-foreground",
    label: "Stale: these results do not match the selected model.",
  },
};

export function StatusIcon({
  status,
  label,
}: {
  status: keyof typeof STATUS_PRESENTATION;
  label?: string;
}) {
  const { icon: Icon, className, label: description } = STATUS_PRESENTATION[status];
  return (
    <Tooltip>
      <TooltipTrigger
        render={<span role="img" aria-label={label ?? description} tabIndex={0} />}
        className={cn(
          "inline-flex shrink-0 rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring",
          className,
        )}
      >
        <Icon className="size-3.5" aria-hidden="true" />
      </TooltipTrigger>
      <TooltipContent>{label ?? description}</TooltipContent>
    </Tooltip>
  );
}

/** A finding owns its freshness cue; source identities stay out of headings. */
export function Section({
  title,
  source,
  wide = false,
  children,
}: {
  title: string;
  source?: FactSource;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      aria-label={title}
      className={cn(
        "flex max-h-full w-[280px] min-w-0 flex-none flex-col gap-2 border-t border-border pt-2",
        wide && "w-[400px]",
      )}
    >
      <div className="flex flex-none items-center justify-between gap-2">
        <span className="text-xs font-semibold">{title}</span>
        {source?.validity === "stale" && <StatusIcon status="stale" />}
      </div>
      <div className="flex min-h-0 flex-col gap-2 overflow-auto text-[11.5px]">{children}</div>
    </section>
  );
}

export function KeyValue({ rows }: { rows: Array<[string, ReactNode]> }) {
  return (
    <dl className="m-0 grid grid-cols-[auto_minmax(0,1fr)] gap-x-2.5 gap-y-1 text-[11px]">
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
        "m-0 text-[11px] leading-relaxed text-pretty",
        issue ? "text-warning-foreground" : "text-muted-foreground",
      )}
    >
      {children}
    </p>
  );
}

export function Prose({ children }: { children: ReactNode }) {
  return <p className="m-0 leading-relaxed text-pretty text-foreground">{children}</p>;
}

export function Callout({ tone, children }: { tone: "ok" | "bad" | "warn"; children: ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-lg border px-2.5 py-2 text-[11px] leading-relaxed",
        tone === "ok" && "border-success/25 bg-success/8",
        tone === "bad" && "border-destructive/30 bg-destructive/6",
        tone === "warn" && "border-warning/40 bg-warning/12 text-warning-foreground",
      )}
    >
      {children}
    </div>
  );
}

/** References navigate to the canonical entity inspector. */
export function OwnerLink({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="cursor-pointer text-left text-pretty underline underline-offset-[3px] hover:text-muted-foreground"
    >
      {children}
    </button>
  );
}

export function ParameterLinks({
  parameters,
  onSelect,
}: {
  parameters: ParameterSpec[];
  onSelect: (selection: ModelSelection) => void;
}) {
  return (
    <ul className="m-0 flex list-none flex-col gap-2 p-0">
      {parameters.map((parameter) => (
        <li key={parameter.id} title={humanize(parameter.description)}>
          <OwnerLink onClick={() => onSelect({ kind: "parameter", id: parameter.id })}>
            {humanize(parameter.name)}
          </OwnerLink>
        </li>
      ))}
    </ul>
  );
}

export function PosteriorTable({ rows }: { rows: PosteriorRow[] }) {
  if (rows.length === 0) return null;
  const intervalLabel = formatPosteriorIntervalLabel(rows[0]);
  const sharedInterval = rows.every((row) => formatPosteriorIntervalLabel(row) === intervalLabel);
  const multiple = rows.length > 1;
  return (
    <table className="w-full border-collapse text-[11px]">
      <thead>
        <tr className="text-left text-muted-foreground">
          {multiple && <th className="border-b pb-1 pr-2 font-medium">Element</th>}
          <th className="border-b pb-1 pr-2 font-medium">Mean</th>
          <th className="border-b pb-1 font-medium">
            {sharedInterval ? intervalLabel : "Interval"}
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.parameter}>
            {multiple && (
              <td className="border-b py-1 pr-2" title={row.parameter}>
                {humanize(row.parameter)}
              </td>
            )}
            <td className="border-b py-1 pr-2 font-mono" style={{ color: signColor(row.mean) }}>
              {formatSigned(row.mean)}
            </td>
            <td className="border-b py-1 font-mono">
              [{formatPlain(row.lower)}, {formatPlain(row.upper)}]
              {!sharedInterval && ` · ${formatPosteriorIntervalLabel(row)}`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
