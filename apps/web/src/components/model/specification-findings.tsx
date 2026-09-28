import type { SpecificationReport } from "@nof1-causal-lab/api-types";
import { humanize } from "@/lib/model-asset/selection";
import { StatusIcon } from "./scope-primitives";

export function SpecificationFindings({ report }: { report: SpecificationReport }) {
  return (
    <ul className="space-y-3 text-xs">
      {report.findings.map((finding) => (
        <li key={finding.check} className="flex items-start gap-2">
          <StatusIcon status={finding.status} />
          <div className="min-w-0 space-y-1">
            <span className="font-medium">{humanize(finding.check)}</span>
            <p className="leading-relaxed text-muted-foreground">{finding.message}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}
