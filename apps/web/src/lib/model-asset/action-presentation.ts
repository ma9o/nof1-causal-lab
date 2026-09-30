import type { ModelDiffReport, ModelSnapshot } from "@nof1-causal-lab/api-types";
import { humanize } from "./selection";

export function recordValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

const countLabel = (n: number, noun: string) => `${n} ${noun}${n === 1 ? "" : "s"}`;
const symbol = { added: "+", removed: "−", revised: "Revised " };

/** Count identity-aligned changes already classified by the backend diff. */
export function modelChangeSummary(diff: ModelDiffReport): string {
  const groups = new Map<string, Set<string>>();
  const names = new Map<string, string>();
  for (const item of diff.graph.constructs) {
    const construct = item.after ?? item.before;
    if (construct) names.set(item.construct_id, humanize(construct.name));
    for (const indicator of [
      ...(item.before?.indicators ?? []),
      ...(item.after?.indicators ?? []),
    ]) {
      names.set(indicator.id, humanize(indicator.name));
    }
  }
  for (const item of diff.parameters) {
    const parameter = item.after ?? item.before;
    if (parameter) names.set(item.parameter_id, humanize(parameter.name));
  }
  for (const item of diff.graph.edges) {
    const edge = item.after ?? item.before;
    if (edge) names.set(item.edge_id, `${names.get(edge.cause.id)} → ${names.get(edge.effect.id)}`);
  }
  const add = (change: keyof typeof symbol, noun: string, id: string) => {
    const key = `${symbol[change]}|${noun}`;
    if (!groups.has(key)) groups.set(key, new Set());
    groups.get(key)!.add(id);
  };
  for (const item of diff.graph.constructs) {
    if (item.change === "added" || item.change === "removed")
      add(item.change, "construct", item.construct_id);
  }
  for (const item of diff.graph.edges) {
    if (item.change === "added" || item.change === "removed")
      add(item.change, "edge", item.edge_id);
  }
  for (const item of diff.definition_changes) {
    const path = item.path.split("/").slice(1);
    if (path[0] === "constructs" && path[2] === "indicators") {
      add(path.length === 4 ? item.change : "revised", "indicator", path[3]);
    }
    if (path[0] === "distributions")
      add(path.length === 2 ? item.change : "revised", "law", path[1]);
  }
  if (!groups.size) {
    for (const item of diff.graph.constructs) {
      if (item.change === "revised") add("revised", "construct", item.construct_id);
    }
    for (const item of diff.graph.edges) {
      if (item.change === "revised") add("revised", "edge", item.edge_id);
    }
    for (const item of diff.parameters)
      add(
        item.change === "added" || item.change === "removed" ? item.change : "revised",
        "parameter",
        item.parameter_id,
      );
  }
  const parts = [...groups].map(([key, ids]) => {
    const [prefix, noun] = key.split("|");
    const name = ids.size === 1 ? names.get([...ids][0]) : null;
    if (name) return `${prefix}${name}`;
    return `${prefix}${countLabel(ids.size, noun)}`;
  });
  if (diff.definition_changes.some((item) => item.path === "/question")) parts.push("Question");
  if (!parts.length && diff.definition_changes.length) {
    parts.push(
      ...new Set(diff.definition_changes.map((item) => humanize(item.path.split("/")[1]))),
    );
  }
  return parts.join(" · ") || "Definitions unchanged";
}

export function initialModelSummary(snapshot: ModelSnapshot): string {
  const graph = snapshot.findings.graph;
  return graph.construct_ids.length || graph.edge_ids.length
    ? `${countLabel(graph.construct_ids.length, "construct")} · ${countLabel(graph.edge_ids.length, "edge")}`
    : "Question";
}
