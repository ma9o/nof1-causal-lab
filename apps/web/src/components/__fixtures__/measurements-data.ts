import type { ObservationRecord, MeasurementsData } from "@nof1-causal-lab/api-types";
import { demoMeasurements } from "./demo-artifacts";
import extractionsSample from "./extraction-sample.json";

// Stories use a larger panel sample than the standard backend preview.
export const combinedExtractionsSample =
  extractionsSample.combined_extractions_sample as unknown as ObservationRecord[];

export const perIndicatorCounts = extractionsSample.per_indicator_counts as Record<string, number>;

export const measurementsData = {
  ...(demoMeasurements as object),
  combined_extractions_sample: combinedExtractionsSample,
  per_indicator_counts: perIndicatorCounts,
} as MeasurementsData;
