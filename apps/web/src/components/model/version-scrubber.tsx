"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useEffect, useId, useRef } from "react";
import type { JournalTick } from "@/lib/model-asset/journal";
import { useRevisionTimeline } from "@/lib/model-asset/use-revision-timeline";
import {
  COLUMN,
  ROW,
  TOP,
  colorFor,
  timelineLinkPath,
  timelineNodeLabel,
} from "@/lib/model-asset/timeline-presentation";
import { cn } from "@/lib/utils";
import { actionLabel } from "@/lib/model-asset/selection";

/** Chronological versions with explicit model ancestry, including fits of earlier revisions. */
export function VersionScrubber({
  ticks,
  playhead,
  latest,
  branches,
  branch,
  comparedSeq,
  onPlayhead,
  onPreviewComparison,
  onEndPreview,
  onKeepComparison,
}: {
  ticks: JournalTick[];
  playhead: number;
  latest: number;
  branches: Record<string, string>;
  branch: string;
  comparedSeq: number | null;
  onPlayhead: (seq: number) => void;
  onPreviewComparison: (seq: number) => void;
  onEndPreview: () => void;
  onKeepComparison: (seq: number) => void;
}) {
  const lineageId = useId();
  const viewport = useRef<HTMLDivElement>(null);
  const selected = useRef<HTMLDivElement>(null);
  const {
    timeline,
    expanded,
    setExpanded,
    visible,
    width,
    height,
    selectedNode,
    position,
    hasComparisons,
  } = useRevisionTimeline(ticks, branches, branch, playhead);

  useEffect(() => {
    const frame = viewport.current;
    const node = selected.current;
    if (!frame || !node) return;
    const left = node.offsetLeft;
    if (left < frame.scrollLeft || left + node.offsetWidth > frame.scrollLeft + frame.clientWidth) {
      frame.scrollLeft = Math.max(0, left - frame.clientWidth / 2 + node.offsetWidth / 2);
    }
    if (node.offsetTop < frame.scrollTop) frame.scrollTop = node.offsetTop;
    else if (node.offsetTop + node.offsetHeight > frame.scrollTop + frame.clientHeight) {
      frame.scrollTop = node.offsetTop + node.offsetHeight - frame.clientHeight;
    }
  }, [playhead, selectedNode?.lane, selectedNode?.column, expanded]);

  return (
    <nav aria-label="Committed history" className="flex-none border-b bg-card">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-5 pt-3 text-[10px] text-muted-foreground">
        <span className="text-xs font-semibold text-foreground">Timeline</span>
        <button
          type="button"
          aria-expanded={expanded}
          aria-controls={lineageId}
          onClick={() => {
            onEndPreview();
            setExpanded((value) => !value);
          }}
          className="inline-flex cursor-pointer items-center gap-1 rounded px-1 py-1 hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          {expanded ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
          {expanded ? "Collapse lineage" : "Expand lineage"}
        </button>
        {playhead !== latest && (
          <button
            type="button"
            onClick={() => onPlayhead(latest)}
            className="ml-auto cursor-pointer rounded-md border px-2 py-1 text-foreground hover:bg-muted"
          >
            Return to latest
          </button>
        )}
        <span aria-live="polite" className="sr-only">
          Viewing version {playhead} on {branch}.
        </span>
      </div>
      {hasComparisons && (
        <span id="model-version-comparison-instructions" className="sr-only">
          Preview differences with selected version {playhead}. Use Compare to keep the comparison
          open.
        </span>
      )}
      {visible.nodes.length === 0 ? (
        <p id={lineageId} className="px-5 py-5 text-xs text-muted-foreground">
          No model history yet.
        </p>
      ) : (
        <div id={lineageId} ref={viewport} className="max-h-80 overflow-auto px-3 pb-1">
          <div className="relative" style={{ width, minWidth: "100%", height }}>
            {expanded ? (
              timeline.lanes.map((lane, index) => (
                <span
                  key={index}
                  className="absolute left-2 text-[10px]"
                  style={{ top: TOP + index * ROW - 6, color: colorFor(index) }}
                >
                  {lane.name}
                </span>
              ))
            ) : (
              <span
                className="absolute left-2 text-[10px] text-muted-foreground"
                style={{ top: TOP - 6 }}
              >
                {branch}
              </span>
            )}
            <svg
              aria-hidden="true"
              width={width}
              height={height}
              className="pointer-events-none absolute inset-0 overflow-visible"
            >
              {selectedNode && (
                <line
                  x1={position(selectedNode).x}
                  x2={position(selectedNode).x}
                  y1={2}
                  y2={height - 4}
                  stroke="currentColor"
                  strokeOpacity={0.12}
                />
              )}
              {visible.links.map((link) => {
                const path = timelineLinkPath(position(link.from), position(link.to));
                return (
                  <path
                    key={`${link.from.tick.seq}:${link.to.tick.seq}:${link.kind}`}
                    data-lineage={link.kind}
                    data-from={link.from.modelRevision}
                    data-to={link.to.modelRevision ?? `version:${link.to.tick.seq}`}
                    d={path}
                    fill="none"
                    stroke={colorFor(link.to.lane)}
                    strokeWidth={2}
                    strokeOpacity={0.75}
                  />
                );
              })}
            </svg>
            {visible.nodes.map((node) => {
              const point = position(node);
              const current = node === selectedNode;
              const canCompare = node.tick.seq !== playhead;
              const compared = node.tick.seq === comparedSeq;
              const label = timelineNodeLabel(node);
              return (
                <div
                  key={node.tick.seq}
                  ref={current ? selected : undefined}
                  className="group absolute"
                  style={{ left: point.x - COLUMN / 2 + 16, top: point.y - 8, width: COLUMN - 32 }}
                  onPointerEnter={() =>
                    canCompare ? onPreviewComparison(node.tick.seq) : onEndPreview()
                  }
                  onPointerLeave={onEndPreview}
                  onFocus={() => canCompare && onPreviewComparison(node.tick.seq)}
                  onBlur={onEndPreview}
                >
                  <button
                    type="button"
                    aria-label={`${label}${node.tick.seq === latest ? " · latest" : ""}`}
                    aria-current={current ? "step" : undefined}
                    aria-describedby={
                      canCompare ? "model-version-comparison-instructions" : undefined
                    }
                    aria-expanded={canCompare ? compared : undefined}
                    aria-controls={compared ? "model-comparison-preview" : undefined}
                    title={`${label}\n${new Date(node.tick.ts).toLocaleString()}`}
                    onClick={() => onPlayhead(node.tick.seq)}
                    className="flex w-full cursor-pointer flex-col items-center rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    data-compared={compared || undefined}
                  >
                    <span
                      className={cn(
                        "relative z-10 block size-4 shrink-0 border-2 border-card",
                        node.modelRevision != null ? "rounded-full" : "rotate-45 rounded-[3px]",
                        (current || compared) && "outline-2 outline-offset-2",
                      )}
                      style={{
                        backgroundColor: colorFor(node.lane),
                        outlineColor: compared ? "#b45309" : colorFor(node.lane),
                      }}
                    />
                    <span
                      className={cn(
                        "mt-1 block w-full rounded-md bg-card/95 px-1.5 py-1 text-center leading-tight group-hover:bg-muted",
                        current && "bg-muted",
                        compared && "bg-amber-50 text-amber-950 ring-1 ring-amber-300",
                      )}
                    >
                      <span className="block truncate text-[11px] font-medium">
                        {actionLabel(node.tick.action)}
                      </span>
                      <span className="flex items-center justify-center gap-1.5 font-mono text-[9px] text-muted-foreground">
                        v{node.tick.seq}
                        {node.tick.seq === latest && (
                          <span
                            title="Latest version"
                            aria-hidden="true"
                            className="size-1.5 rounded-full bg-current"
                          />
                        )}
                      </span>
                    </span>
                  </button>
                  {canCompare && (
                    <button
                      type="button"
                      aria-label={`Compare version ${playhead} with version ${node.tick.seq}`}
                      onClick={() => onKeepComparison(node.tick.seq)}
                      className="absolute -right-4 -top-1 cursor-pointer rounded border bg-card px-1.5 py-0.5 text-[9px] opacity-0 shadow-sm transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:outline-2 focus-visible:outline-ring [@media(hover:none)]:opacity-100"
                    >
                      Compare
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </nav>
  );
}
