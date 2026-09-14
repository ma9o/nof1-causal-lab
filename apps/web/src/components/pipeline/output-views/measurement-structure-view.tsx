"use client";

import { modelConstructs } from "@/lib/model-accessors";
import { constructStatuses } from "@/components/dag/construct-statuses";
import { StructureDag } from "@/components/dag/structure-dag";
import { IndicatorTable } from "@/components/analysis-widgets/measurement-structure/indicator-table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import { AlertTriangle } from "lucide-react";

export default function MeasurementStructureView({ data }: { data: ModelSnapshot }) {
  const spec = data.model?.value;
  if (!spec) return null;
  const indicators = modelConstructs(spec).flatMap((construct) => construct.indicators);
  const names = new Map<string, string>(
    modelConstructs(spec).map((construct) => [construct.id, construct.name]),
  );
  const nonId = data.findings.identification?.value.status.non_identifiable_treatments ?? {};
  const nonIdEntries = Object.entries(nonId);
  const nodeStatuses = constructStatuses(data);

  return (
    <div className="space-y-4">
      {nonIdEntries.length > 0 && (
        <Alert variant="warning" className="border-2">
          <AlertTriangle className="h-5 w-5 mt-0.5" />
          <AlertTitle className="text-base font-semibold">
            Some Treatment Effects Were Excluded
          </AlertTitle>
          <AlertDescription className="mt-2 space-y-2">
            <p>
              {nonIdEntries.length} treatment(s) remain non-identifiable and will be excluded from
              downstream intervention analysis. Identifiable treatments still remain.
            </p>
            <div className="space-y-1.5">
              {nonIdEntries.map(([id, status]) => (
                <div key={id} className="flex flex-wrap items-center gap-1.5 text-sm">
                  <span className="font-medium">{names.get(id)}</span>
                  <span className="text-warning/70">&larr;</span>
                  {status?.confounders.map((c) => (
                    <Badge key={c} variant="warning" className="text-xs">
                      {names.get(c)}
                    </Badge>
                  ))}
                  {status?.notes && (
                    <span className="text-muted-foreground text-xs">({status.notes})</span>
                  )}
                </div>
              ))}
            </div>
          </AlertDescription>
        </Alert>
      )}
      <StructureDag
        outcomeId={spec.default_outcome?.id}
        constructs={modelConstructs(spec)}
        edges={spec.edges}
        indicators={indicators}
        nodeStatuses={nodeStatuses}
      />
      <IndicatorTable constructs={modelConstructs(spec)} indicators={indicators} />
    </div>
  );
}
