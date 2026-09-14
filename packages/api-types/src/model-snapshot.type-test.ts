import type { MethodResponse } from "openapi-fetch";
import type { createModelClient } from "./client";
import type { paths } from "./generated/model-api";
import type {
  Construct,
  FactSource,
  Indicator,
  InferenceReport,
  ModelSnapshot,
  ModelSpec,
  ParameterSpec,
} from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;
type Equal<A, B> = [A, B] extends [B, A] ? true : false;

export type CanonicalDefinition = Expect<
  Equal<NonNullable<ModelSnapshot["model"]>["value"], ModelSpec>
>;
export type CanonicalInferenceReport = Expect<
  Equal<NonNullable<ModelSnapshot["findings"]["fit"]>["value"]["report"], InferenceReport>
>;
export type CanonicalParameter = Expect<Equal<ModelSpec["parameters"][number], ParameterSpec>>;
export type CanonicalConstruct = Expect<Equal<ModelSpec["constructs"][number], Construct>>;
export type OwnedIndicator = Expect<Equal<Construct["indicators"][number], Indicator>>;
// @ts-expect-error Indicator ownership is declared by containment.
export type NoIndependentIndicatorOwner = Indicator["construct_id"];
export type SourceValidityIsScalar = Expect<Extends<FactSource["validity"], "fresh" | "stale">>;

type ModelRead = paths["/api/episodes/{workspace_id}/model"]["get"];
type DefinitionRead = paths["/api/episodes/{workspace_id}/model/definition"]["get"];
type ConstructsRead = paths["/api/episodes/{workspace_id}/model/constructs"]["get"];
export type GeneratedReadReusesBatch = Expect<
  Equal<ModelRead["responses"][200]["content"]["application/json"], ModelSnapshot>
>;
export type GeneratedReadReusesDefinition = Expect<
  Equal<
    DefinitionRead["responses"][200]["content"]["application/json"],
    Exclude<ModelSnapshot["model"], undefined>
  >
>;
export type GeneratedReadReusesConstructs = Expect<
  Equal<ConstructsRead["responses"][200]["content"]["application/json"], Construct[]>
>;
export type InvalidRevisionQuery = Expect<
  // @ts-expect-error Revision queries require numeric journal positions.
  Extends<string, NonNullable<ModelRead["parameters"]["query"]>["at_seq"]>
>;

type FetchedModel = MethodResponse<
  ReturnType<typeof createModelClient>,
  "get",
  "/api/episodes/{workspace_id}/model"
>;
export type FetchedModelRetainsCanonicalTuples = Expect<Equal<FetchedModel, ModelSnapshot>>;
