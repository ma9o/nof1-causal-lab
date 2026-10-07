"use client";

import { inferenceView } from "@/lib/model-asset/result-values";

import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import { useActionResult } from "./use-model-snapshot";

/** The complete report is already part of the saved action result. */
export function useInferenceReport(modelSnapshot: ModelSnapshot) {
  const query = useActionResult(
    modelSnapshot.workspace_id,
    modelSnapshot.state.current.model?.revision,
    modelSnapshot.fit != null,
  );
  return { ...query, data: query.data?.action === "fit" ? inferenceView(query.data.body) : null };
}
