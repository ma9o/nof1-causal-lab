import type { DataDiffReport, ModelSnapshot } from "@nof1-causal-lab/api-types";
import type { ModelEntities } from "./entities";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type { JournalTick } from "./journal";

/** The viewed version and the selection shared by the graph and details. */
export interface ScopeContext {
  model: ModelSnapshot;
  entities: ModelEntities;
  select: (selection: EntitySelection | null) => void;
  ticks: JournalTick[];
  dataDiff: { data: DataDiffReport | undefined; error: Error | null };
}
