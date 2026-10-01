import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { parseSimulationReport, type SimulationWithEffects } from "@/lib/simulation-report";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import simulationTrace from "./simulation-trace.json";

/** The newest `simulate` result recorded in the DEMO trace. */
export const demoSimulationResult: SimulationWithEffects = traceToUIMessages(
  simulationTrace as LLMTrace,
)
  .flatMap((message) => (message.role === "assistant" ? message.parts : []))
  .flatMap((part) =>
    part.type === "dynamic-tool" &&
    part.state === "output-available" &&
    part.toolName === "simulate"
      ? [parseSimulationReport(part.output)]
      : [],
  )
  .filter((result): result is SimulationWithEffects => result !== null)
  .at(-1)!;
