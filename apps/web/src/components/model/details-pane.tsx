import { DefinitionContext } from "./definition-context";
import { Button } from "@/components/ui/button";
import { resolveEntity } from "@/lib/model-asset/entities";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { Hint } from "./scope-primitives";
import { ConstructScope } from "./scopes/construct-scope";
import { EdgeScope } from "./scopes/edge-scope";
import { IndicatorScope } from "./scopes/indicator-scope";
import type { ScopeContext } from "@/lib/model-asset/scope";

/**
 * The state the selected action left, in depth. With a graph part selected it shows that part.
 * With nothing selected it shows the model-wide state that action produced, and only that:
 * - edit_model: the model as specified, meaning how the question is identified and the equations.
 * - prepare_data: the panel as a whole, meaning each variable's coverage over time and the
 *   dataset-level issues.
 * - fit: the fitted model as a whole, meaning predictive calibration, latent mixing and the joint
 *   posterior.
 * - simulate: the simulation as a whole, meaning its design, the effect or why it is withheld,
 *   and its checks.
 */
export function DetailsPane({
  selection,
  context,
  loading,
  onClose,
}: {
  selection: EntitySelection;
  context: ScopeContext;
  loading: boolean;
  onClose: () => void;
}) {
  const entity = resolveEntity(context.entities, selection);
  return (
    <section
      aria-label="Model details"
      className="flex h-[460px] min-h-0 min-w-0 flex-none flex-col gap-3 overflow-hidden rounded-2xl border bg-card px-3 py-3 md:max-h-[52%]"
    >
      <div className="flex flex-none items-center gap-2 text-xs">
        <h2 className="min-w-0 flex-1 font-semibold">{entity?.label ?? "Absent entity"}</h2>
        <Button
          type="button"
          size="icon-sm"
          variant="ghost"
          aria-label="Close details"
          onClick={onClose}
        >
          ×
        </Button>
      </div>
      {entity && <DefinitionContext entity={entity} onSelect={context.select} />}
      <div
        key={selection.id}
        className="flex min-h-0 flex-1 flex-col flex-wrap content-start items-start gap-x-[18px] gap-y-3 overflow-x-auto pb-2 [&>section]:max-h-full [&>section]:w-[280px] [&>section[data-wide]]:w-[400px] [&>section>div:last-child]:overflow-auto"
      >
        {loading ? (
          <p role="status" className="text-xs">
            Loading version…
          </p>
        ) : !entity ? (
          <Hint>This {selection.kind} is absent from this revision.</Hint>
        ) : selection.kind === "construct" ? (
          <ConstructScope context={context} id={selection.id} />
        ) : selection.kind === "edge" ? (
          <EdgeScope context={context} id={selection.id} />
        ) : (
          <IndicatorScope context={context} id={selection.id} />
        )}
      </div>
    </section>
  );
}
