import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import type { CausalEdge, Construct } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { SourceBadges } from "../source-badges";

const col = createColumnHelper<CausalEdge>();

function columns(constructs: Construct[]) {
  const names = new Map(constructs.map((item) => [item.id, item.name]));
  return [
    col.accessor("cause_id", {
      header: "Cause",
      cell: (info) => <span className="font-medium">{names.get(info.getValue())}</span>,
    }),
    col.accessor("effect_id", {
      header: "Effect",
      cell: (info) => <span className="font-medium">{names.get(info.getValue())}</span>,
    }),
    col.accessor("lagged", {
      header: "Timing",
      cell: (info) => (
        <Badge variant={info.getValue() ? "default" : "secondary"}>
          {info.getValue() ? "Lagged" : "Contemporaneous"}
        </Badge>
      ),
    }),
    col.accessor("description", {
      header: "Description",
      cell: (info) => (
        <span className="max-w-xs whitespace-normal text-muted-foreground">{info.getValue()}</span>
      ),
    }),
    col.display({
      id: "sources",
      header: () => (
        <HeaderWithTooltip
          label="Sources"
          tooltip="Literature sources supporting this causal link. Click to open."
        />
      ),
      cell: ({ row }) => <SourceBadges sources={row.original.sources} />,
      meta: { align: "center" },
    }),
  ];
}

export function EdgeList({ edges, constructs }: { edges: CausalEdge[]; constructs: Construct[] }) {
  return (
    <InfoTable columns={columns(constructs) as ColumnDef<CausalEdge, unknown>[]} data={edges} />
  );
}
