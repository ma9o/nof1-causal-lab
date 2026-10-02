import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { hasCausalEffects } from "@/lib/simulation-report";
import reports from "./simulation-reports.json";

/** The newest certified result in the owner-validated illustrative reports. */
export const demoSimulationResult = fixtureValue(reports.filter(hasCausalEffects).at(-1));
