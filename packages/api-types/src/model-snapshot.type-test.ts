import type { MethodResponse } from "openapi-fetch";
import type { createModelClient } from "./client";
import type { paths } from "./generated/model-api";
import type {
  Construct,
  ConstructId,
  EdgeId,
  FactSource,
  Indicator,
  LatentStructure,
  MeasurementStructureArtifact,
  PosteriorArtifact,
  StatisticalModelSpecArtifact,
  ModelSnapshot,
  ParameterSpec,
} from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;
type Equal<A, B> = [A, B] extends [B, A] ? true : false;

export type CanonicalLatentStructure = Expect<
  Equal<NonNullable<ModelSnapshot["latent_structure"]>["value"], LatentStructure>
>;
export type CanonicalMeasurementStructure = Expect<
  Equal<NonNullable<ModelSnapshot["measurement_structure"]>["value"], MeasurementStructureArtifact>
>;
export type CanonicalSpecification = Expect<
  Equal<NonNullable<ModelSnapshot["specification"]>["value"], StatisticalModelSpecArtifact>
>;
export type CanonicalPosterior = Expect<
  Equal<NonNullable<ModelSnapshot["fit"]>["value"]["posterior"], PosteriorArtifact>
>;
export type CanonicalParameter = Expect<
  Equal<NonNullable<ModelSnapshot["compiled_parameters"]>["value"][number], ParameterSpec>
>;
export type IndicatorHasConstructOwner = Expect<Extends<Indicator["construct_id"], ConstructId>>;
// @ts-expect-error Indicator owners cannot reference edges.
export type InvalidIndicatorOwner = Expect<Extends<Indicator["construct_id"], EdgeId>>;
export type SourceValidityIsScalar = Expect<Extends<FactSource["validity"], "fresh" | "stale">>;

type ModelRead = paths["/api/episodes/{workspace_id}/model"]["get"];
type LatentRead = paths["/api/episodes/{workspace_id}/model/latent-structure"]["get"];
type ConstructsRead = paths["/api/episodes/{workspace_id}/model/constructs"]["get"];
export type GeneratedReadReusesBatch = Expect<
  Equal<ModelRead["responses"][200]["content"]["application/json"], ModelSnapshot>
>;
export type GeneratedReadReusesLatentAggregate = Expect<
  Equal<LatentRead["responses"][200]["content"]["application/json"], Exclude<ModelSnapshot["latent_structure"], undefined>>
>;
export type GeneratedReadReusesConstructs = Expect<
  Equal<ConstructsRead["responses"][200]["content"]["application/json"], Construct[]>
>;
export type InvalidRevisionQuery = Expect<
  // @ts-expect-error Revision queries are numbers, not prose.
  Extends<string, NonNullable<ModelRead["parameters"]["query"]>["at_seq"]>
>;

type FetchedModel = MethodResponse<
  ReturnType<typeof createModelClient>,
  "get",
  "/api/episodes/{workspace_id}/model"
>;
export type FetchedModelRetainsCanonicalTuples = Expect<Equal<FetchedModel, ModelSnapshot>>;
