"use client";

import type { RecordDependency } from "@nof1-causal-lab/api-types";
import { useEffect, useMemo, useRef } from "react";
import type { JournalTick } from "@/lib/model-asset/journal";
import { revisionTimeline, TIMELINE_LANES } from "@/lib/model-asset/revision-timeline";
import {
  ACTION_STYLE,
  type ActionGlyph,
  COLUMN,
  FAILED_COLOR,
  GUTTER,
  LABEL_TOP,
  laneY,
  timelineLinkPath,
  timelinePosition,
  timelineSize,
  timelineTickLabel,
} from "@/lib/model-asset/timeline-presentation";
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

/**
 * The study's actions in execution order, one column each, in lanes by what they produce. Links
 * follow the served dependencies: each action connects to the earlier actions whose outputs its
 * request named, and dotted links mark outputs only its checks read.
 */
export function VersionScrubber({
  ticks,
  dependencies,
  playhead,
  latest,
  branch,
  comparedSeq,
  onPlayhead,
  onPreviewComparison,
  onEndPreview,
  onKeepComparison,
}: {
  ticks: JournalTick[];
  dependencies: RecordDependency[];
  playhead: number;
  latest: number;
  branch: string;
  comparedSeq: number | null;
  onPlayhead: (seq: number) => void;
  onPreviewComparison: (seq: number) => void;
  onEndPreview: () => void;
  onKeepComparison: (seq: number) => void;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const selected = useRef<HTMLDivElement>(null);
  const timeline = useMemo(() => revisionTimeline(ticks, dependencies), [ticks, dependencies]);
  const { width, height } = timelineSize(timeline.nodes.length);
  const selectedNode = timeline.nodes.find((node) => node.tick.seq === playhead);
  const hasComparisons = timeline.nodes.some(
    (node) =>
      node.tick.status === "applied" &&
      node.tick.action !== "data_diff" &&
      node.tick.seq !== playhead,
  );
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
  }, [playhead, selectedNode?.column]);

  return (
    <nav aria-label="Action history" className="flex-none border-b bg-card">
      <div className="flex items-center gap-4 px-5 pt-2.5 text-xs">
        <span className="font-semibold">Timeline</span>
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
          Preview topology differences with {timelineTickLabel(selectedNode.tick)}. Use Compare to
          keep the comparison open.
        </span>
      )}
      {timeline.nodes.length === 0 ? (
        <p className="px-5 py-4 text-xs text-muted-foreground">No actions yet.</p>
      ) : (
        <div ref={viewport} className="overflow-x-auto px-3 pb-1.5">
          <div className="relative" style={{ width, minWidth: "100%", height }}>
            {TIMELINE_LANES.map((lane, index) => (
              <span
                key={lane.name}
                className="absolute left-2 font-mono text-[10px] text-muted-foreground"
                style={{ top: laneY(index) - 7 }}
              >
                {lane.name}
              </span>
            ))}
            <svg
              aria-hidden="true"
              width={width}
              height={height}
              className="pointer-events-none absolute inset-0 overflow-visible"
            >
              {TIMELINE_LANES.map((lane, index) => (
                <line
                  key={lane.name}
                  x1={GUTTER - 8}
                  x2={width - 12}
                  y1={laneY(index)}
                  y2={laneY(index)}
                  className="stroke-border"
                  strokeDasharray="1 5"
                />
              ))}
              {timeline.links.map((link) => {
                const from = timelinePosition(link.from);
                const to = timelinePosition(link.to);
                const touches = link.from === selectedNode || link.to === selectedNode;
                return (
                  <path
                    key={`${link.from.tick.seq}:${link.to.tick.seq}`}
                    data-argument={link.argument}
                    d={timelineLinkPath(from, to)}
                    fill="none"
                    className={touches ? "stroke-muted-foreground" : "stroke-border"}
                    strokeWidth={touches ? 2 : 1.5}
                    strokeDasharray={
                      link.check ? "2 4" : link.to.tick.status !== "applied" ? "3 4" : undefined
                    }
                  />
                );
              })}
            </svg>
            {timeline.nodes.map((node) => {
              const point = timelinePosition(node);
              const current = node === selectedNode;
              const failed = node.tick.status !== "applied";
              const canCompare =
                !failed && node.tick.action !== "data_diff" && node.tick.seq !== playhead;
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
                  style={{
                    left: point.x - COLUMN / 2,
                    top: point.y - MARK / 2,
                    width: COLUMN,
                    height: LABEL_TOP + 30 - (point.y - MARK / 2),
                  }}
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
                    className="flex h-full w-full cursor-pointer flex-col items-center rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
                    {/* Every label sits in the shared row; a hairline ties it to its mark. */}
                    <span aria-hidden="true" className="my-0.5 w-px flex-1 bg-border/70" />
                    <span
                      className={cn(
                        "block max-w-full truncate px-1 text-[11px] leading-4",
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
