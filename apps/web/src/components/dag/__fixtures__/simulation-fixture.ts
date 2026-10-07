import { decodeFixture, fixtureValue } from "@/components/__fixtures__/fixture-value";
import { hasCausalEffects } from "@/lib/simulation-report";
import rawReports from "./simulation-reports.json";

const reports = decodeFixture(rawReports);

/** The newest certified result in the owner-validated illustrative reports. */
export const demoSimulationResult = fixtureValue(reports.filter(hasCausalEffects).at(-1));
