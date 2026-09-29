import type { JournalTick } from "./journal";

export interface RevisionTimelineNode {
  tick: JournalTick;
  column: number;
  lane: number;
  modelRevision: string | null;
  inputModels: string[];
}

export interface RevisionTimelineLink {
  from: RevisionTimelineNode;
  to: RevisionTimelineNode;
  kind: "ancestry";
}

/** Display the ancestry reachable from a native Git branch head. */
export function revisionBranch(timeline: ReturnType<typeof revisionTimeline>, head: string) {
  const byId = new Map(timeline.nodes.map((node) => [node.tick.commitId, node]));
  const selected = new Set<string>();
  const pending = [head];
  while (pending.length) {
    const id = pending.pop()!;
    if (selected.has(id)) continue;
    selected.add(id);
    pending.push(...(byId.get(id)?.tick.parentIds ?? []));
  }
  const branchName = byId.get(head)?.tick.branch;
  const nodes = timeline.nodes
    .filter(
      (node) =>
        selected.has(node.tick.commitId) ||
        (node.tick.status !== "applied" &&
          node.tick.parentIds.some((id) => selected.has(id)) &&
          (node.tick.branch === branchName ||
            timeline.nodes.some(
              (child) =>
                selected.has(child.tick.commitId) &&
                child.tick.branch === node.tick.branch &&
                child.tick.seq > node.tick.seq &&
                child.tick.parentIds.some((id) => node.tick.parentIds.includes(id)),
            ))),
    )
    .map((node, column) => ({ ...node, column }));
  const visible = new Map(nodes.map((node) => [node.tick.commitId, node]));
  const links = timeline.links.flatMap((link) => {
    const from = visible.get(link.from.tick.commitId);
    const to = visible.get(link.to.tick.commitId);
    return from && to ? [{ ...link, from, to }] : [];
  });
  return { nodes, links };
}

/** Presentation only: commits supply ancestry; refs supply the branch lanes. */
export function revisionTimeline(ticks: readonly JournalTick[], branches: Record<string, string>) {
  const lanes = [...new Set([...Object.keys(branches), ...ticks.map((tick) => tick.branch)])].map(
    (name) => ({ name }),
  );
  const nodes: RevisionTimelineNode[] = ticks.map((tick, column) => ({
    tick,
    column,
    inputModels: typeof tick.inputs.model_revision === "string" ? [tick.inputs.model_revision] : [],
    lane: lanes.findIndex((lane) => lane.name === tick.branch),
    modelRevision: tick.produced.find((info) => info.artifact_id === "model")?.revision ?? null,
  }));
  const commits = new Map(nodes.map((node) => [node.tick.commitId, node]));
  const links: RevisionTimelineLink[] = nodes.flatMap((to) =>
    to.tick.parentIds.flatMap((id) => {
      const from = commits.get(id);
      return from ? [{ from, to, kind: "ancestry" as const }] : [];
    }),
  );
  return { nodes, links, lanes };
}
