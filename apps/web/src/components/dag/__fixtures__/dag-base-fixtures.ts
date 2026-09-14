import { modelConstructs } from "@/lib/model-accessors";
import { demoModel } from "../../__fixtures__/demo-artifacts";
export const constructs = modelConstructs(demoModel);
export const edges = demoModel.edges;
export const indicators = constructs.flatMap((construct) => construct.indicators);
