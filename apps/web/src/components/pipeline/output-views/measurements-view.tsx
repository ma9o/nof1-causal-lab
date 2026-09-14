import { DataTable } from "@/components/ui/data-table";
import { ExploreDataframeButton } from "@/components/ui/explore-dataframe-button";
import type { Indicator, MeasurementsData } from "@nof1-causal-lab/api-types";

export default function MeasurementsView({
  data,
  workspaceId,
  indicators,
}: {
  data: MeasurementsData;
  indicators: Indicator[];
  workspaceId: string;
}) {
  const definitions = new Map(indicators.map((indicator) => [indicator.id, indicator]));
  const rows = data.combined_extractions_sample.map(({ indicator_id, ...row }) => ({
    indicator: definitions.get(indicator_id)!.name,
    ...row,
  }));
  const totalExtractions = data.n_observations;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">
          {totalExtractions.toLocaleString()} observations
        </span>
        <ExploreDataframeButton artifactId="panel" workspaceId={workspaceId} />
      </div>

      {data.combined_extractions_sample.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Showing a sample of {data.combined_extractions_sample.length} rows out of{" "}
          {totalExtractions.toLocaleString()}
        </p>
      )}
      <DataTable rows={rows} />
    </div>
  );
}
