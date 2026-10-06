import type { RecordDependency, TimelineRevision } from "@nof1-causal-lab/api-types";

/** Input roles supplying the saved outputs composed into each action's view. */
const viewArguments = {
  edit_question: [],
  edit_model: ["parent"],
  prepare_data: ["model"],
  fit: ["model", "data"],
  simulate: ["model", "panel"],
  model_diff: ["after"],
  data_diff: ["right"],
} satisfies Record<TimelineRevision["record"]["attempt"]["action"], string[]>;

export function viewDependencies(
  entry: TimelineRevision,
  dependencies: readonly RecordDependency[],
): readonly RecordDependency[] {
  const roles: readonly string[] = viewArguments[entry.record.attempt.action];
  return dependencies.filter(
    (item) => item.seq === entry.record.seq && roles.includes(item.argument),
  );
}

/** Resolve a recorded commit or artifact to its first successful producing call. */
export function producingCall(
  attempts: readonly TimelineRevision[],
  identity: string | undefined,
): TimelineRevision | undefined {
  return attempts.find((entry) => {
    const { outcome } = entry.record.attempt;
    return (
      entry.call_id !== null &&
      outcome.status === "applied" &&
      (entry.commit_id === identity ||
        outcome.effects.produced.some((artifact) => artifact.revision === identity))
    );
  });
}

/** Follow parent references using timeline metadata, without loading intermediate model bodies. */
export function questionCall(
  entry: TimelineRevision,
  attempts: readonly TimelineRevision[],
  dependencies: readonly RecordDependency[],
): TimelineRevision {
  if (entry.record.attempt.action === "edit_question") return entry;
  const argument = entry.record.attempt.action === "edit_model" ? "parent" : "model";
  const dependency = dependencies.find(
    (item) => item.seq === entry.record.seq && item.argument === argument,
  );
  const parent = attempts.find((item) => item.record.seq === dependency?.source_seq);
  if (!parent) throw new Error(`Missing ${argument} dependency for call ${entry.record.seq}`);
  return questionCall(parent, attempts, dependencies);
}

/** Only declared input dependencies belong to a call's view; journal neighbors do not. */
export function callDependencies(
  selected: TimelineRevision,
  attempts: readonly TimelineRevision[],
  dependencies: readonly RecordDependency[],
): readonly TimelineRevision[] {
  const bySeq = new Map(attempts.map((entry) => [entry.record.seq, entry]));
  const visited = new Set<number>();
  const calls: TimelineRevision[] = [];
  function visit(entry: TimelineRevision) {
    const seq = entry.record.seq;
    if (visited.has(seq)) return;
    visited.add(seq);
    if (entry.record.attempt.action === "edit_model") {
      visit(questionCall(entry, attempts, dependencies));
    } else {
      for (const dependency of viewDependencies(entry, dependencies)) {
        const parent = bySeq.get(dependency.source_seq);
        if (parent === undefined)
          throw new Error(`Missing recorded dependency ${dependency.source_seq} for call ${seq}`);
        visit(parent);
      }
    }
    calls.push(entry);
  }
  visit(selected);
  return calls;
}
