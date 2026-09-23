import type { Point } from "@/lib/utils/dag-graph-layout";
import type { ConstructStatus } from "./construct-statuses";

// ---- node geometry — shared between the layout sizing and the SVG card ----
export const NODE_W = 220;
export const NODE_W_WITH_INDICATORS = 248;
export const HEADER_H = 56; // name + role·temporal subtitle, no indicators
export const SEPARATOR_Y = 46; // dashed rule between header and the indicator rows
export const INDICATOR_ROW_H = 17;

export function nodeHeight(indicatorCount: number): number {
  if (indicatorCount === 0) return HEADER_H;
  return SEPARATOR_Y + indicatorCount * INDICATOR_ROW_H + 8;
}

// Cross-construct lagged edges and self-dynamics originate from t−1 ghosts;
// contemporaneous edges stay within the present-time slice.
const EDGE_DEFAULT = "var(--edge-contemporary)";
const EDGE_BLOCKING = "var(--destructive)";
const EDGE_MARGINALIZED = "var(--warning)";
export const SELF_EDGE_OPACITY = 0.45;

/** Edge tint from its endpoints' identifiability status (blocking wins over marginalized). */
export function edgeStatusColor(
  a: ConstructStatus | undefined,
  b: ConstructStatus | undefined,
): { color: string; flagged: boolean } {
  if (a === "blocking" || b === "blocking") return { color: EDGE_BLOCKING, flagged: true };
  if (a === "marginalized" || b === "marginalized")
    return { color: EDGE_MARGINALIZED, flagged: true };
  return { color: EDGE_DEFAULT, flagged: false };
}

/**
 * Stitch the two glyph-split halves into one polyline and remove the spacer spur.
 * Splitting a → [spacer] → b makes ELK route into the spacer's ports; on a back-edge
 * it enters and exits the same side, leaving a short out-and-back "spur" by the
 * spacer (an empty slot would otherwise expose it as a squiggle). A spur is a
 * H-V-H (or V-H-V) run whose outer two segments
 * reverse direction with only a short perpendicular hop between — that overshoot is
 * dropped. Genuine routing detours (large perpendicular travel to clear a node) are
 * left untouched.
 */
export function cleanGlyphPath(raw: Point[]): Point[] {
  const SPUR_PERP = 32;
  const dedupe = (pts: Point[]): Point[] =>
    pts.filter((p, i, a) => i === 0 || p.x !== a[i - 1].x || p.y !== a[i - 1].y);
  const pts = dedupe(raw.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y) })));
  for (let i = 0; i + 3 < pts.length; ) {
    const a = pts[i];
    const b = pts[i + 1];
    const c = pts[i + 2];
    const d = pts[i + 3];
    const s1x = b.x - a.x;
    const s1y = b.y - a.y;
    const s2x = c.x - b.x;
    const s2y = c.y - b.y;
    const s3x = d.x - c.x;
    const s3y = d.y - c.y;
    if (s1y === 0 && s3y === 0 && s2x === 0 && s1x * s3x < 0 && Math.abs(s2y) <= SPUR_PERP) {
      pts.splice(i + 1, 2, { x: d.x, y: a.y }); // horizontal overshoot → clamp back
      i = Math.max(0, i - 1);
    } else if (s1x === 0 && s3x === 0 && s2y === 0 && s1y * s3y < 0 && Math.abs(s2x) <= SPUR_PERP) {
      pts.splice(i + 1, 2, { x: a.x, y: d.y }); // vertical overshoot → clamp back
      i = Math.max(0, i - 1);
    } else {
      i++;
    }
  }
  return dedupe(pts);
}
