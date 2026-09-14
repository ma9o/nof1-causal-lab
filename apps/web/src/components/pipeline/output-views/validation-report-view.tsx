import type { IndicatorSpec, ValidationReportArtifact } from "@nof1-causal-lab/api-types";
import { IndicatorHealthTable } from "@/components/analysis-widgets/validation-report/indicator-health-table";

export default function ValidationReportView({
  data,
  indicators,
}: {
  data: ValidationReportArtifact;
  indicators: IndicatorSpec[];
}) {
  const audits = data.indicators;

  return (
    <div className="space-y-4">
      {Object.keys(audits).length > 0 && (
        <IndicatorHealthTable audits={audits} indicators={indicators} />
      )}
    </div>
  );
}
