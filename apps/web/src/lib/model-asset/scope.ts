import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ModelEntities } from "./entities";
import type { ModelSelection } from "@/lib/model-asset/selection";

/** The viewed version and the selection shared by the graph and details. */
export interface ScopeContext {
  model: ModelSnapshot;
  entities: ModelEntities;
  ticks: JournalTick[];
  select: (selection: ModelSelection) => void;
}
