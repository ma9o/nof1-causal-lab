import { recordValue } from "@/lib/model-asset/action-presentation";

type ChainValues = { chain: number; values: number[] };
type RankHistogram = { expected: number; chains: ChainValues[] };
type LatentSteps = { initial: number[][]; final: number[][] };

/** Chains use the theme's categorical chart colors, cycling after five. */
export const chainColor = (chain: number) => `var(--chart-${(chain % 5) + 1})`;

const finite = (value: unknown): number[] =>
  Array.isArray(value)
    ? value.filter((item): item is number => typeof item === "number" && Number.isFinite(item))
    : [];

const records = (value: unknown): Record<string, unknown>[] =>
  Array.isArray(value)
    ? value.map(recordValue).filter((item): item is Record<string, unknown> => item !== null)
    : [];

function chainValues(entry: Record<string, unknown>, key: "values" | "counts"): ChainValues[] {
  return records(entry.chains).flatMap((chain) =>
    typeof chain.chain === "number" ? [{ chain: chain.chain, values: finite(chain[key]) }] : [],
  );
}

/** Diagnostics label parameters differently but share the engine coordinate. */
export function coordinateKey(value: unknown): string | null {
  const coordinate = recordValue(value);
  return coordinate && typeof coordinate.site_name === "string" && Array.isArray(coordinate.indices)
    ? `${coordinate.site_name}[${coordinate.indices.join(",")}]`
    : null;
}

function byCoordinate<T>(
  value: unknown,
  read: (entry: Record<string, unknown>) => T,
): Map<string, T> {
  return new Map(
    records(value).flatMap((entry) => {
      const key = coordinateKey(entry.coordinate);
      return key ? [[key, read(entry)] as const] : [];
    }),
  );
}

/** Traces and rank histograms recorded by the engine, keyed by parameter coordinate. */
export function readChainDiagnostics(diagnostics: Record<string, unknown>) {
  const mcmc = recordValue(diagnostics.mcmc);
  return {
    traces: byCoordinate(mcmc?.trace_data, (entry) => chainValues(entry, "values")),
    ranks: byCoordinate(
      mcmc?.rank_histograms,
      (entry): RankHistogram => ({
        expected: typeof entry.expected_per_bin === "number" ? entry.expected_per_bin : 0,
        chains: chainValues(entry, "counts"),
      }),
    ),
  };
}

/** Per-time-point latent proposal step sizes before and after warmup adaptation. */
export function readLatentSteps(diagnostics: Record<string, unknown>): LatentSteps | null {
  const gibbs = recordValue(diagnostics.marginal_particle_gibbs);
  const rows = (value: unknown) => (Array.isArray(value) ? value.map(finite) : []);
  const final = rows(gibbs?.final_latent_delta);
  return final.length ? { initial: rows(gibbs?.initial_latent_delta), final } : null;
}

const TRACE_WIDTH = 120;
const TRACE_HEIGHT = 28;

export function TraceSparkline({ chains }: { chains: ChainValues[] }) {
  const values = chains.flatMap((chain) => chain.values);
  if (!values.length) return <span className="text-muted-foreground">—</span>;
  const low = Math.min(...values);
  const span = Math.max(...values) - low || 1;
  const length = Math.max(...chains.map((chain) => chain.values.length));
  return (
    <svg
      viewBox={`0 0 ${TRACE_WIDTH} ${TRACE_HEIGHT}`}
      className="h-7 w-[120px]"
      role="img"
      aria-label="Draws by chain"
    >
      {chains.map(({ chain, values: draws }) => (
        <polyline
          key={chain}
          fill="none"
          stroke={chainColor(chain)}
          strokeWidth={0.75}
          vectorEffect="non-scaling-stroke"
          points={draws
            .map(
              (value, index) =>
                `${(index / Math.max(1, length - 1)) * TRACE_WIDTH},${TRACE_HEIGHT - ((value - low) / span) * TRACE_HEIGHT}`,
            )
            .join(" ")}
        />
      ))}
    </svg>
  );
}

const RANK_ROW = 7;
const RANK_GAP = 2;

export function RankBars({ histogram }: { histogram: RankHistogram }) {
  const rows = histogram.chains;
  const bins = Math.max(0, ...rows.map((row) => row.values.length));
  if (!bins) return <span className="text-muted-foreground">—</span>;
  // Scale so evenly mixed chains fill about half a row and pile-ups stay visible.
  const peak = Math.max(histogram.expected * 2, ...rows.flatMap((row) => row.values));
  const height = rows.length * (RANK_ROW + RANK_GAP) - RANK_GAP;
  const expected = RANK_ROW - (histogram.expected / peak) * RANK_ROW;
  return (
    <svg
      viewBox={`0 0 ${TRACE_WIDTH} ${height}`}
      className="w-[120px]"
      style={{ height }}
      role="img"
      aria-label="Rank histogram by chain"
    >
      {rows.map(({ chain, values }, row) => {
        const top = row * (RANK_ROW + RANK_GAP);
        return (
          <g key={chain}>
            {values.map((count, bin) => {
              const barHeight = (count / peak) * RANK_ROW;
              return (
                <rect
                  // biome-ignore lint/suspicious/noArrayIndexKey: bins are positional
                  key={bin}
                  x={(bin / bins) * TRACE_WIDTH}
                  y={top + RANK_ROW - barHeight}
                  width={Math.max(0.5, TRACE_WIDTH / bins - 0.75)}
                  height={barHeight}
                  fill={chainColor(chain)}
                />
              );
            })}
            <line
              x1={0}
              x2={TRACE_WIDTH}
              y1={top + expected}
              y2={top + expected}
              stroke="var(--muted-foreground)"
              strokeDasharray="2 2"
              strokeWidth={0.5}
              vectorEffect="non-scaling-stroke"
            />
          </g>
        );
      })}
    </svg>
  );
}

const STEP_WIDTH = 360;
const STEP_HEIGHT = 72;

export function LatentStepChart({ steps }: { steps: LatentSteps }) {
  const positive = [...steps.final.flat(), ...steps.initial.flat()].filter((value) => value > 0);
  if (!positive.length) return null;
  const low = Math.log10(Math.min(...positive));
  const span = Math.log10(Math.max(...positive)) - low || 1;
  const y = (value: number) => STEP_HEIGHT - ((Math.log10(value) - low) / span) * STEP_HEIGHT;
  const length = Math.max(...steps.final.map((row) => row.length));
  const starts = steps.initial.flat();
  const start = starts.length && starts.every((value) => value === starts[0]) ? starts[0] : null;
  return (
    <svg
      viewBox={`0 0 ${STEP_WIDTH} ${STEP_HEIGHT}`}
      preserveAspectRatio="none"
      className="h-[72px] w-full"
      role="img"
      aria-label="Latent proposal step size by time point and chain"
    >
      {start !== null && start > 0 && (
        <line
          x1={0}
          x2={STEP_WIDTH}
          y1={y(start)}
          y2={y(start)}
          stroke="var(--muted-foreground)"
          strokeDasharray="3 3"
          strokeWidth={0.75}
          vectorEffect="non-scaling-stroke"
        />
      )}
      {steps.final.map((row, chain) => (
        <polyline
          // biome-ignore lint/suspicious/noArrayIndexKey: rows are chains in engine order
          key={chain}
          fill="none"
          stroke={chainColor(chain)}
          strokeWidth={0.75}
          vectorEffect="non-scaling-stroke"
          points={row
            .flatMap((value, index) =>
              value > 0 ? [`${(index / Math.max(1, length - 1)) * STEP_WIDTH},${y(value)}`] : [],
            )
            .join(" ")}
        />
      ))}
    </svg>
  );
}

export function ChainLegend({ chains }: { chains: number }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
      {Array.from({ length: chains }, (_, chain) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: chains are numbered by position
        <span key={chain} className="inline-flex items-center gap-1">
          <span className="size-2 rounded-full" style={{ background: chainColor(chain) }} />
          chain {chain + 1}
        </span>
      ))}
    </div>
  );
}
