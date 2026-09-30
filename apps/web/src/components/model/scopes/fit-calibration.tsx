import { LOOPITChart } from "@/components/charts/loo-pit-chart";
import { ParetoKChart } from "@/components/charts/pareto-k-chart";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatNumber } from "@/lib/utils/format";
import { KeyValue, Section } from "../scope-primitives";

/** The fit's leave-one-out predictive checks, as the engine recorded them. */
export function FitCalibration({ context }: { context: ScopeContext }) {
  const fit = context.model.findings.fit;
  const loo = fit?.value.report.loo_diagnostics;
  if (!fit || !loo) return null;
  return (
    <Section title="Leave-one-out calibration" source={fit.source} wide>
      <KeyValue
        rows={[
          ["ELPD", `${formatNumber(loo.elpd_loo, 1)} ± ${formatNumber(loo.se, 1)}`],
          ["Effective parameters", formatNumber(loo.p_loo, 1)],
          ["Held-out rows", loo.n_data_points.toLocaleString()],
          ...(loo.n_bad_k != null
            ? [["Unreliable rows", loo.n_bad_k.toLocaleString()] as [string, string]]
            : []),
        ]}
      />
      {loo.loo_pit && loo.loo_pit.length > 0 && <LOOPITChart loo={loo} />}
      {loo.pareto_k && loo.pareto_k.length > 0 && <ParetoKChart loo={loo} />}
    </Section>
  );
}
