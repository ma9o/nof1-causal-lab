import type { ConstructId, MechanismViewRequest } from "@nof1-causal-lab/api-types";
import { useState } from "react";
import { HistoryPlot, PATH_COLORS } from "@/components/charts/history-plot";
import { PlotNumberInput } from "@/components/charts/plot-number-input";
import { useMechanismCurves } from "@/lib/hooks/use-visuals";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, Section } from "../scope-primitives";
import { DrawPager } from "./recorded-history";

export function MechanismResponse({ context, owner }: { context: ScopeContext; owner: string }) {
  const [request, setRequest] = useState<MechanismViewRequest>({
    owner_id: owner,
    lower: -3,
    upper: 3,
    held: {},
    levels: [-1, 0, 1],
    start: 0,
    count: 24,
    points: 201,
  });
  const query = useMechanismCurves(context.model, request);
  const curve = query.data;
  return (
    <Section title="Mechanism response" wide>
      <Hint>
        Conditional contribution to the rate of change. Each curve evaluates the declared nonlinear
        equations for one parameter draw; it is not a total intervention effect.
      </Hint>
      <form
        className="flex flex-wrap items-end gap-2 text-[10px]"
        onSubmit={(event) => {
          event.preventDefault();
          const fields = new FormData(event.currentTarget);
          const lower = Number(fields.get("lower")),
            upper = Number(fields.get("upper"));
          if (Number.isFinite(lower) && Number.isFinite(upper) && lower < upper)
            setRequest({ ...request, lower, upper });
        }}
      >
        <label>
          State from
          <input
            name="lower"
            aria-label="Response state minimum"
            type="number"
            step="any"
            required
            defaultValue={request.lower}
            className="block w-20 rounded border bg-background p-1"
          />
        </label>
        <label>
          to
          <input
            name="upper"
            aria-label="Response state maximum"
            type="number"
            step="any"
            required
            defaultValue={request.upper}
            className="block w-20 rounded border bg-background p-1"
          />
        </label>
        <button type="submit" className="rounded border px-2 py-1">
          Apply range
        </button>
        <label>
          Grid
          <select
            aria-label="Response grid points"
            className="block rounded border bg-background p-1"
            value={request.points}
            onChange={(event) => setRequest({ ...request, points: Number(event.target.value) })}
          >
            {[201, 501, 1001].map((n) => (
              <option key={n} value={n}>
                {n} points
              </option>
            ))}
          </select>
        </label>
      </form>
      {curve && (
        <>
          <div className="flex flex-wrap gap-2 text-[10px]">
            <label>
              Horizontal state
              <select
                aria-label="Response axis"
                className="block max-w-44 rounded border bg-background p-1"
                value={curve.axis}
                onChange={(event) => {
                  const axis = event.target.value as ConstructId;
                  const held = { ...request.held };
                  delete held[axis];
                  setRequest({
                    ...request,
                    axis,
                    held,
                    moderator: request.moderator === axis ? null : request.moderator,
                  });
                }}
              >
                {Object.entries(curve.states).map(([id, name]) => (
                  <option key={id} value={id}>
                    {name.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Compare another state
              <select
                aria-label="Response moderator"
                className="block max-w-44 rounded border bg-background p-1"
                value={request.moderator ?? ""}
                onChange={(event) => {
                  const moderator = (event.target.value || null) as ConstructId | null;
                  const held = { ...request.held };
                  if (moderator) delete held[moderator];
                  setRequest({ ...request, moderator, held });
                }}
              >
                <option value="">None</option>
                {Object.entries(curve.states)
                  .filter(([id]) => id !== curve.axis)
                  .map(([id, name]) => (
                    <option key={id} value={id}>
                      {name.replaceAll("_", " ")}
                    </option>
                  ))}
              </select>
            </label>
          </div>
          {request.moderator && (
            <div className="flex gap-2">
              {request.levels.map((value, index) => (
                <label key={index} className="text-[10px]" style={{ color: PATH_COLORS[index] }}>
                  Level {index + 1}
                  <PlotNumberInput
                    aria-label={`Moderator level ${index + 1}`}
                    step="any"
                    className="block w-20 rounded border bg-background p-1"
                    value={value}
                    onValue={(value) => {
                      const levels: MechanismViewRequest["levels"] = [...request.levels];
                      levels[index] = value;
                      setRequest({ ...request, levels });
                    }}
                  />
                </label>
              ))}
            </div>
          )}
          {Object.entries(curve.held).map(([id, value]) => (
            <label key={id} className="flex items-center justify-between gap-2 text-[10px]">
              Hold {curve.states[id].replaceAll("_", " ")} at
              <PlotNumberInput
                aria-label={`Hold ${curve.states[id]}`}
                step="any"
                className="w-20 rounded border bg-background p-1"
                value={value}
                onValue={(value) =>
                  setRequest({
                    ...request,
                    held: { ...request.held, [id]: value },
                  })
                }
              />
            </label>
          ))}
          {curve.total_draws > 1 && (
            <DrawPager
              start={request.start}
              count={request.count}
              total={curve.total_draws}
              source={curve.law === "sampled" ? "current-law" : "saved"}
              onStart={(start) => setRequest({ ...request, start })}
              onCount={(count) => setRequest({ ...request, count })}
            />
          )}
          <HistoryPlot
            times={curve.x}
            xLabel={curve.axis_label}
            yLabel={`${curve.target_label}: drift contribution / day`}
            label={`${curve.axis_label} → ${curve.target_label}: mechanism response`}
            description={[
              "Conditional drift contribution per day; this is not a total intervention effect.",
              curve.law === "fixed"
                ? "Fixed coefficients."
                : curve.law === "retained"
                  ? "Original joint parameter draws; correlations are preserved."
                  : "Reproducible draws from the current authored laws (plot seed 0).",
              ...Object.entries(curve.held).map(
                ([id, value]) => `${curve.states[id].replaceAll("_", " ")} held at ${value}.`,
              ),
              "Curves join evaluated grid points without smoothing.",
            ].join(" ")}
            series={curve.curves.map((path) => ({
              id: `${path.draw}:${path.level}`,
              label: `Draw ${path.draw + 1}${path.level != null ? ` · ${curve.states[curve.moderator!]} = ${path.level}` : ""}`,
              values: path.values,
              color:
                PATH_COLORS[
                  path.level != null
                    ? request.levels.indexOf(path.level) % PATH_COLORS.length
                    : path.draw % PATH_COLORS.length
                ],
            }))}
          />
          <Hint>The range is user selected on the latent state scale.</Hint>
          {curve.nonfinite > 0 && (
            <Hint issue>
              {curve.nonfinite} undefined evaluations remain gaps. Narrow the range to inspect the
              valid domain.
            </Hint>
          )}
        </>
      )}
      {query.isLoading && <Hint>Evaluating model equations…</Hint>}
      {query.error && <Hint issue>{query.error.message}</Hint>}
    </Section>
  );
}
