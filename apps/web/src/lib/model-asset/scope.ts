import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import type { ModelEntities } from "./entities";
import type { EntitySelection } from "@/lib/model-asset/selection";

/** The viewed version and the selection shared by the graph and details. */
export interface ScopeContext {
  model: ModelSnapshot;
  entities: ModelEntities;
  select: (selection: EntitySelection) => void;
}
