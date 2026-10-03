import type { MethodResponse } from "openapi-fetch";
import type openapi from "../schemas/openapi.json";
import type { createModelClient } from "./client";
import type { paths } from "./generated/model-api";
import type {
  ActionPoll,
  Assessment,
  Attempt,
  Change,
  ConstructId,
  ConstructRef,
  ConstructSpec,
  DistributionId,
  EditModelRequest,
  FactSource,
  IndicatorId,
  IndicatorSpec,
  InferenceReport,
  InferenceReportCore,
  ModelSnapshot,
  ModelSpec,
  NumPyroDistribution,
  ParameterRef,
  ParameterSpec,
  Rejected,
  ScientificActionId,
  Sourced,
  TimelineResponse,
} from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;
type Equal<A, B> = [A, B] extends [B, A] ? true : false;

export type GenericAttemptCorrelatesRequest = Expect<
  Equal<Attempt<"edit_model", EditModelRequest, null>["request"], EditModelRequest | null>
>;
export type GenericSourceRetainsValue = Expect<Equal<Sourced<ModelSpec>["value"], ModelSpec>>;
export type GenericAssessmentRetainsSubject = Expect<
  Equal<Assessment<string, number>["subject"], string>
>;
export type GenericChangeRetainsPayload = Expect<
  Equal<Extract<Change<ModelSpec>, { kind: "added" }>["after"], ModelSpec>
>;
// @ts-expect-error Rejection reasons are the closed domain reason type.
export type RejectionHasNoUnrelatedReason = Rejected<number>;

export type CanonicalDefinition = Expect<
  Equal<NonNullable<ModelSnapshot["model"]>["value"], ModelSpec>
>;
export type CanonicalInferenceCore = Expect<
  Equal<NonNullable<ModelSnapshot["fit"]>["value"]["report"], InferenceReportCore>
>;
export type CanonicalPairAxes = Expect<
  Equal<InferenceReport["detail"]["posterior_pairs"][number], readonly [ParameterRef, ParameterRef]>
>;
export type CanonicalParameter = Expect<Equal<ModelSpec["parameters"][number], ParameterSpec>>;
export type CanonicalConstruct = Expect<
  Equal<ModelSpec["edges"][number]["cause"], ConstructSpec | ConstructRef>
>;
export type OwnedIndicator = Expect<Equal<ConstructSpec["indicators"][number], IndicatorSpec>>;
// @ts-expect-error Indicator ownership is declared by containment.
export type NoIndependentIndicatorOwner = IndicatorSpec["construct_id"];
export type SourceValidityIsScalar = Expect<Extends<FactSource["validity"], "fresh" | "stale">>;

type ModelRead = paths["/api/studies/{workspace_id}/model"]["get"];
type DefinitionRead = paths["/api/studies/{workspace_id}/model/definition"]["get"];
type ConstructsRead = paths["/api/studies/{workspace_id}/model/constructs"]["get"];
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
  Equal<ConstructsRead["responses"][200]["content"]["application/json"], readonly ConstructSpec[]>
>;
export type InvalidRevisionQuery = Expect<
  // @ts-expect-error Revision queries require numeric journal positions.
  Extends<string, NonNullable<ModelRead["parameters"]["query"]>["at_seq"]>
>;

type FetchedModel = MethodResponse<
  ReturnType<typeof createModelClient>,
  "get",
  "/api/studies/{workspace_id}/model"
>;
export type FetchedModelRetainsCanonicalTuples = Expect<Equal<FetchedModel, ModelSnapshot>>;

export type SparseLawLookup = Expect<
  Equal<ModelSpec["distributions"][DistributionId], NumPyroDistribution | undefined>
>;
export type DistinctScientificIds = Expect<Equal<Extends<IndicatorId, ConstructId>, false>>;
export type RequiredNullableResponse = Expect<
  Equal<Record<string, never> extends Pick<ModelSnapshot, "model"> ? true : false, false>
>;
type MechanismInput =
  paths["/api/studies/{workspace_id}/model/visuals/mechanism"]["post"]["requestBody"]["content"]["application/json"];
export type RequestDefaultsMayBeOmitted = Expect<Extends<{ owner_id: string }, MechanismInput>>;

// Coverage follows the exported schema; adding an endpoint cannot silently leave it out of the client.
export type EveryExportedPath = Expect<Equal<keyof paths, keyof typeof openapi.paths>>;
type HttpMethod = "get" | "post" | "put" | "patch" | "delete" | "head" | "options" | "trace";
type ExportedOperations = {
  [Path in keyof typeof openapi.paths]: `${Extract<keyof (typeof openapi.paths)[Path], HttpMethod>} ${Path}`;
}[keyof typeof openapi.paths];
type GeneratedOperations = {
  [Path in keyof paths]: {
    [Method in Extract<keyof paths[Path], HttpMethod>]: paths[Path][Method] extends undefined
      ? never
      : `${Method} ${Path}`;
  }[Extract<keyof paths[Path], HttpMethod>];
}[keyof paths];
export type EveryExportedOperation = Expect<Equal<GeneratedOperations, ExportedOperations>>;
type Dispatch = paths["/api/studies/{workspace_id}/actions"]["post"];
type DispatchInput = Dispatch["requestBody"]["content"]["application/json"];
export type CanonicalDispatchAlternatives = Expect<
  Equal<NonNullable<DispatchInput["action"]>, ScientificActionId>
>;
export type CanonicalPoll = Expect<
  Equal<
    MethodResponse<
      ReturnType<typeof createModelClient>,
      "get",
      "/api/studies/{workspace_id}/actions/{attempt_id}"
    >,
    ActionPoll
  >
>;
export type CanonicalTimeline = Expect<
  Equal<
    MethodResponse<
      ReturnType<typeof createModelClient>,
      "get",
      "/api/studies/{workspace_id}/timeline"
    >,
    TimelineResponse
  >
>;
export type CanonicalCapabilities = Expect<
  Equal<
    MethodResponse<ReturnType<typeof createModelClient>, "get", "/api/actions-enabled">,
    boolean
  >
>;
export type CanonicalWorkspaces = Expect<
  Equal<
    MethodResponse<ReturnType<typeof createModelClient>, "get", "/api/workspaces">,
    Readonly<Record<string, string | null>>
  >
>;
type Upload = paths["/api/upload"]["post"]["requestBody"]["content"]["multipart/form-data"];
export type NativeUploadFile = Expect<Equal<Upload["file"], Blob>>;
export type DefaultsAreAbsent = Expect<
  Equal<Extends<{ owner_id: string; points: undefined }, MechanismInput>, false>
>;

declare const definition: ModelSpec;
declare const edge: ModelSpec["edges"][number];
// @ts-expect-error Published arrays are read-only.
definition.parameters.push({});
// @ts-expect-error Read-only guarantees extend to nested values.
edge.description = "changed";
// @ts-expect-error Published maps are read-only.
definition.distributions["distribution:changed"] = {};
// @ts-expect-error A map retains its scientific key kind.
export type WrongMapKey = ModelSpec["distributions"][IndicatorId];
