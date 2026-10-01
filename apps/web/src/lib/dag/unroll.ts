/** Temporal layout helpers. The backend selects evolving states; ghosts are their previous slice. */

/** Suffix marking a node id as the t−1 (previous-timestep) ghost of its base. */
export const GHOST_SUFFIX = "__p";

/** The t−1 ghost id for a present-time node. */
export const ghostId = (base: string): string => `${base}${GHOST_SUFFIX}`;

/** Whether an id refers to a t−1 ghost rather than a present-time node. */
export const isGhost = (id: string): boolean => id.endsWith(GHOST_SUFFIX);

/** Glyph/spacer node size. */
export const GLYPH_W = 86;
export const GLYPH_H = 36;

/** A causal pair to lay out as `a → [glyph] → b`. */
export interface GlyphPair {
  a: string;
  b: string;
  /** Self-dynamics edge (ghost → its own present node) rather than a cross-edge. */
  isSelf: boolean;
  crossSlice: boolean;
}

export interface GlyphSplit {
  glyphNodes: { id: string; width: number; height: number }[];
  edges: { id: string; source: string; target: string }[];
  /** Glyph layout-node id (`G__<i>`) → the pair it sits on. */
  glyphs: Map<string, GlyphPair>;
}

/**
 * Split each pair `a → b` into `a → [glyph] → b`, inserting a glyph node on the
 * edge — the same construction the analysis DAG uses. Putting a real node on every
 * edge is what gives the layered layout its column rhythm (each edge spans an
 * extra layer), so structural and intervention graphs share it.
 *
 * `glyphSize` controls that node's footprint. analysis uses the full GLYPH_W×GLYPH_H
 * because it draws a drift glyph there, and the box hides ELK's port-entry jog. The
 * structural DAG leaves the slot empty and draws the edge straight through (it
 * concatenates the `e<i>s` + `e<i>t` routed segments), so it passes a near-point
 * size: layer insertion is size-independent, so the columns still match analysis,
 * but a point has no port-entry jog to expose — no squiggle.
 */
export function splitEdgesWithGlyphs(
  pairs: GlyphPair[],
  glyphSize: { width: number; height: number } = { width: GLYPH_W, height: GLYPH_H },
): GlyphSplit {
  const glyphNodes: GlyphSplit["glyphNodes"] = [];
  const edges: GlyphSplit["edges"] = [];
  const glyphs = new Map<string, GlyphPair>();
  pairs.forEach((pair, i) => {
    const gid = `G__${i}`;
    glyphNodes.push({ id: gid, width: glyphSize.width, height: glyphSize.height });
    glyphs.set(gid, pair);
    edges.push({ id: `e${i}s`, source: pair.a, target: gid });
    edges.push({ id: `e${i}t`, source: gid, target: pair.b });
  });
  return { glyphNodes, edges, glyphs };
}
