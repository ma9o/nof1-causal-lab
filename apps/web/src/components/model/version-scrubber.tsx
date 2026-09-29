"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useEffect, useId, useRef } from "react";
import type { JournalTick } from "@/lib/model-asset/journal";
import {
  ACTION_STYLE,
  type ActionGlyph,
  COLUMN,
  FAILED_COLOR,
  ROW,
  TOP,
  timelineLinkPath,
  timelineTickLabel,
} from "@/lib/model-asset/timeline-presentation";
import { useRevisionTimeline } from "@/lib/model-asset/use-revision-timeline";
import { cn } from "@/lib/utils";

const MARK = 16;
const HOVER_INTENT_MS = 300;

/** One action's mark on the rail; a failed attempt takes the same slot as a cross. */
function ActionMark({
  glyph,
  color,
  failed,
}: {
  glyph: ActionGlyph;
  color: string;
  failed: boolean;
}) {
  return (
    <svg viewBox="0 0 16 16" width={MARK} height={MARK} aria-hidden="true">
      <circle cx={8} cy={8} r={8} className="fill-card" />
      {failed ? (
        <path
          d="M4.75 4.75 11.25 11.25M11.25 4.75 4.75 11.25"
          stroke={FAILED_COLOR}
          strokeWidth={2}
          strokeLinecap="round"
        />
      ) : glyph === "dot" ? (
        <circle cx={8} cy={8} r={5} fill={color} />
      ) : glyph === "diamond" ? (
        <rect x={4} y={4} width={8} height={8} rx={1.5} transform="rotate(45 8 8)" fill={color} />
      ) : glyph === "ring" ? (
        <circle cx={8} cy={8} r={4.25} fill="none" stroke={color} strokeWidth={2.5} />
      ) : (
        <path d="M5.5 3.75 12.25 8 5.5 12.25Z" fill={color} stroke={color} strokeLinejoin="round" />
      )}
    </svg>
  );
}

/** The study's actions in order: every state change is one of the four scientific actions. */
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
  const branched = timeline.lanes.length > 1;
  // A comparison costs a backend diff: start it only once the pointer rests on a tick.
  const hoverIntent = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelHoverIntent = () => {
    if (hoverIntent.current) clearTimeout(hoverIntent.current);
    hoverIntent.current = null;
  };
  useEffect(() => cancelHoverIntent, []);

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
    <nav aria-label="Action history" className="flex-none border-b bg-card">
      <div className="flex items-center gap-4 px-5 pt-2.5 text-xs">
        <span className="font-semibold">Timeline</span>
        {branched && (
          <button
            type="button"
            aria-expanded={expanded}
            aria-controls={lineageId}
            onClick={() => {
              onEndPreview();
              setExpanded((value) => !value);
            }}
            className="inline-flex cursor-pointer items-center gap-1 rounded px-1 py-0.5 text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
          >
            {expanded ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
            {expanded ? "Collapse lineage" : "Expand lineage"}
          </button>
        )}
        {playhead !== latest && (
          <button
            type="button"
            onClick={() => onPlayhead(latest)}
            className="ml-auto cursor-pointer rounded-md border px-2 py-0.5 hover:bg-muted"
          >
            Return to latest
          </button>
        )}
        {selectedNode && (
          <span aria-live="polite" className="sr-only">
            Viewing {timelineTickLabel(selectedNode.tick)} on {branch}.
          </span>
        )}
      </div>
      {hasComparisons && selectedNode && (
        <span id="model-version-comparison-instructions" className="sr-only">
          Preview differences with {timelineTickLabel(selectedNode.tick)}. Use Compare to keep the
          comparison open.
        </span>
      )}
      {visible.nodes.length === 0 ? (
        <p id={lineageId} className="px-5 py-4 text-xs text-muted-foreground">
          No actions yet.
        </p>
      ) : (
        <div id={lineageId} ref={viewport} className="max-h-72 overflow-auto px-3 pb-1.5">
          <div className="relative" style={{ width, minWidth: "100%", height }}>
            {(expanded ? timeline.lanes.map((lane) => lane.name) : [branch]).map((name, index) => (
              <span
                key={name}
                className="absolute left-2 max-w-14 truncate font-mono text-[10px] text-muted-foreground"
                style={{ top: TOP + index * ROW - 7 }}
                title={name}
              >
                {name}
              </span>
            ))}
            <svg
              aria-hidden="true"
              width={width}
              height={height}
              className="pointer-events-none absolute inset-0 overflow-visible"
            >
              {visible.links.map((link) => (
                <path
                  key={`${link.from.tick.seq}:${link.to.tick.seq}:${link.kind}`}
                  data-lineage={link.kind}
                  data-from={link.from.modelRevision}
                  data-to={link.to.modelRevision ?? `version:${link.to.tick.seq}`}
                  d={timelineLinkPath(position(link.from), position(link.to))}
                  fill="none"
                  className="stroke-border"
                  strokeWidth={2}
                  strokeDasharray={link.to.tick.status !== "applied" ? "3 4" : undefined}
                />
              ))}
            </svg>
            {visible.nodes.map((node) => {
              const point = position(node);
              const current = node === selectedNode;
              const failed = node.tick.status !== "applied";
              const canCompare = !failed && node.tick.seq !== playhead;
              const compared = node.tick.seq === comparedSeq;
              const isLatest = node.tick.seq === latest;
              const style = ACTION_STYLE[node.tick.action];
              const label = timelineTickLabel(node.tick);
              const accessibleLabel = `${label}${failed ? " · failed" : ""}${isLatest ? " · latest" : ""}`;
              return (
                <div
                  key={node.tick.seq}
                  ref={current ? selected : undefined}
                  className="group absolute"
                  style={{ left: point.x - COLUMN / 2, top: point.y - MARK / 2, width: COLUMN }}
                  onPointerEnter={() => {
                    cancelHoverIntent();
                    if (!canCompare) return onEndPreview();
                    hoverIntent.current = setTimeout(
                      () => onPreviewComparison(node.tick.seq),
                      HOVER_INTENT_MS,
                    );
                  }}
                  onPointerLeave={() => {
                    cancelHoverIntent();
                    onEndPreview();
                  }}
                  onFocus={() => canCompare && onPreviewComparison(node.tick.seq)}
                  onBlur={onEndPreview}
                >
                  <button
                    type="button"
                    aria-label={accessibleLabel}
                    aria-current={current ? "step" : undefined}
                    aria-describedby={
                      canCompare ? "model-version-comparison-instructions" : undefined
                    }
                    aria-expanded={canCompare ? compared : undefined}
                    aria-controls={compared ? "model-comparison-preview" : undefined}
                    title={`${accessibleLabel}\n${new Date(node.tick.ts).toLocaleString()}${failed && node.tick.error ? `\n${node.tick.error}` : ""}`}
                    onClick={() => {
                      cancelHoverIntent();
                      onPlayhead(node.tick.seq);
                    }}
                    className="flex w-full cursor-pointer flex-col items-center rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    data-compared={compared || undefined}
                  >
                    <span
                      className={cn(
                        "grid size-4 place-items-center rounded-full",
                        (current || compared) && "outline-2 outline-offset-2",
                      )}
                      style={{
                        outlineColor: compared ? "#b45309" : failed ? FAILED_COLOR : style.color,
                      }}
                    >
                      <ActionMark glyph={style.glyph} color={style.color} failed={failed} />
                    </span>
                    <span
                      className={cn(
                        "mt-1.5 block max-w-full truncate px-1 text-[11px] leading-4",
                        current
                          ? "font-semibold text-foreground"
                          : failed
                            ? "text-muted-foreground/70 group-hover:text-foreground"
                            : "text-muted-foreground group-hover:text-foreground",
                        compared && "text-amber-900",
                      )}
                    >
                      {node.tick.action}
                    </span>
                    <span
                      className={cn(
                        "flex items-center gap-1 font-mono text-[10px] leading-3.5",
                        current ? "text-foreground/70" : "text-muted-foreground/70",
                      )}
                    >
                      {node.tick.commitId.slice(0, 7)}
                      {isLatest && (
                        <span
                          title="Latest action"
                          aria-hidden="true"
                          className="size-1.5 rounded-full bg-current"
                        />
                      )}
                    </span>
                  </button>
                  {canCompare && selectedNode && (
                    <button
                      type="button"
                      aria-label={`Compare ${timelineTickLabel(selectedNode.tick)} with ${label}`}
                      onClick={() => onKeepComparison(node.tick.seq)}
                      className="absolute top-0 left-[calc(50%+14px)] cursor-pointer rounded border bg-card px-1 text-[9px] leading-[14px] text-muted-foreground opacity-0 shadow-xs transition-opacity hover:text-foreground group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:outline-2 focus-visible:outline-ring [@media(hover:none)]:opacity-100"
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
