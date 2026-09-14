import { indicatorOwners } from "@/lib/model-accessors";
import { Badge } from "@/components/ui/badge";
import { InfoTable } from "@/components/ui/info-table";
import type { Indicator, Construct } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";

const col = createColumnHelper<Indicator>();

const columns = [
  col.accessor("name", {
    header: "Name",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("measurement_dtype", {
    header: "Dtype",
    cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
  }),
  col.accessor("aggregation", {
    header: "Aggregation",
    cell: (info) => <Badge variant="secondary">{info.getValue()}</Badge>,
  }),
  col.accessor("observation_window", {
    header: "Window",
    cell: (info) => (
      <span className="text-sm text-muted-foreground">{info.getValue() ?? "model_clock"}</span>
    ),
  }),
  col.accessor("how_to_measure", {
    header: "How to Measure",
    cell: (info) => <p className="max-w-xs text-pretty text-muted-foreground">{info.getValue()}</p>,
  }),
];

export function IndicatorTable({
  indicators,
  constructs,
}: {
  indicators: Indicator[];
  constructs: Construct[];
}) {
  const owners = indicatorOwners(constructs);
  return (
    <InfoTable
      columns={columns as ColumnDef<Indicator, unknown>[]}
      data={indicators}
      groupBy={(row) => owners.get(row.id)!.id}
      renderGroupHeader={(construct, rows) => (
        <>
          <span className="text-sm font-semibold">
            {constructs.find((item) => item.id === construct)!.name}
          </span>
          <span className="ml-2 text-xs text-muted-foreground">
            {rows.length} indicator{rows.length !== 1 && "s"}
          </span>
        </>
      )}
    />
  );
}
