import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, KeyValue, Section } from "../scope-primitives";

export function DataDetails({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const metadata = context.model.data.metadata;
  const raw = context.model.data.raw_data;
  if (!metadata)
    return (
      <Section title="Prepared data" source={raw?.source}>
        {raw && tick.produced.some((artifact) => artifact.artifact_id === "raw_data") ? (
          <KeyValue
            rows={[
              ["Imported records", raw.value.n_records.toLocaleString()],
              ["Columns", String(raw.value.n_columns)],
              ["First observation", raw.value.date_range.start],
              ["Last observation", raw.value.date_range.end],
            ]}
          />
        ) : (
          <Hint>No observation panel was produced.</Hint>
        )}
      </Section>
    );
  const { source, variables, time_origin } = metadata.value;
  return (
    <Section title="Prepared data" source={metadata.source}>
      <KeyValue
        rows={[
          ...("files" in source
            ? ([
                ["Files", source.files.join(", ")],
                ["Start (inclusive)", source.start ?? "Unbounded"],
                ["End (exclusive)", source.end ?? "Unbounded"],
              ] as Array<[string, string]>)
            : ([
                ["Simulation", source.revision],
                ["Replicate", String(source.replicate)],
              ] as Array<[string, string]>)),
          ["Variables", String(variables.length)],
          ["Model day zero", time_origin ?? "Calendar-free"],
        ]}
      />
    </Section>
  );
}
