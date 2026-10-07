import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import history from "../../../../../data/HEALTHDEMO/fixture/model_history.json";
import snapshot from "../../../../../data/HEALTHDEMO/fixture/model_snapshot.json";

/** Validated by the production reader when read fixtures are generated. */
export const demoModelSnapshot = snapshot;

export function demoSnapshotAt(seq: number): ModelSnapshot {
  if (seq === demoModelSnapshot.selected_seq) return demoModelSnapshot;
  const revision = history[seq];
  if (!revision) throw new Error(`No committed HEALTHDEMO revision ${seq}`);
  return revision;
}
