import { MACHINE_DESCRIPTION, type PipelineSectionId } from "@nof1-causal-lab/api-types";

/** Include model authoring recipes, fits and simulation, including failed attempts. */
export function isModelOperation(id: PipelineSectionId): boolean {
  return (
    id === "simulate" ||
    MACHINE_DESCRIPTION.transitions.some(
      (operation) =>
        operation.transition_id === id &&
        [...operation.produces, ...operation.produces_optional].includes("model"),
    )
  );
}
