"use client";

import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import { useActionResult } from "./use-model-snapshot";

/** The complete report is already part of the saved action result. */
export function useInferenceReport(model: ModelSnapshot) {
  const query = useActionResult(
    model.workspace_id,
    model.state.current.model?.revision,
    model.fit != null,
  );
  return { ...query, data: query.data?.action === "fit" ? query.data.body.inference_report : null };
}
