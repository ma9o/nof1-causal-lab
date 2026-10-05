import type { MethodResponse } from "openapi-fetch";
import type openapi from "../schemas/openapi.json";
import type { createModelClient } from "./client";
import type { paths } from "./generated/model-api";
import type {
  ActionId,
  ActionPoll,
  Assessment,
  Attempt,
  Change,
  ConstructId,
  ConstructRef,
  ConstructSpec,
  DistributionId,
  EditModelRequest,
  Evaluation,
  FactSource,
  IndicatorId,
  IndicatorSpec,
  InferenceReportCore,
  ModelSnapshot,
  ModelSpec,
  NumPyroDistribution,
  ObservationSpec,
  ParameterSpec,
  Rejected,
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
export type GenericAvailabilityRetainsPayload = Expect<
  Equal<Extract<Evaluation<ModelSpec>, { kind: "available" }>["value"], ModelSpec>
>;
export type GenericObservationRetainsWindow = Expect<
  Equal<ObservationSpec<"1d">["observation_window"], "1d">
>;
// @ts-expect-error Rejection reasons are the closed domain reason type.
export type RejectionHasNoUnrelatedReason = Rejected<number>;

export type CanonicalDefinition = Expect<
  Equal<NonNullable<ModelSnapshot["model"]>["value"], ModelSpec>
>;
export type CanonicalInferenceCore = Expect<
  Equal<NonNullable<ModelSnapshot["fit"]>["value"]["report"], InferenceReportCore>
>;
export type CanonicalParameter = Expect<Equal<ModelSpec["parameters"][number], ParameterSpec>>;
export type CanonicalConstruct = Expect<
  Equal<ModelSpec["edges"][number]["cause"], ConstructSpec | ConstructRef>
>;
export type OwnedIndicator = Expect<Equal<ConstructSpec["indicators"][number], IndicatorSpec>>;
// @ts-expect-error Indicator ownership is declared by containment.
export type NoIndependentIndicatorOwner = IndicatorSpec["construct_id"];
export type SourceValidityIsScalar = Expect<Extends<FactSource["validity"], "fresh" | "stale">>;

type FitCall = paths["/api/studies/{workspace_id}/fit"]["post"];
type FetchedCall = MethodResponse<
  ReturnType<typeof createModelClient>,
  "post",
  "/api/studies/{workspace_id}/fit"
>;
export type GeneratedCallRetainsCanonicalSnapshot = Expect<
  Equal<Extract<FetchedCall, { kind: "completed" }>["snapshot"], ModelSnapshot | null>
>;
export type GeneratedCallRetainsCanonicalOutcome = Expect<Equal<FetchedCall, ActionPoll>>;

export type SparseLawLookup = Expect<
  Equal<ModelSpec["distributions"][DistributionId], NumPyroDistribution | undefined>
>;
export type DistinctScientificIds = Expect<Equal<Extends<IndicatorId, ConstructId>, false>>;
export type RequiredNullableResponse = Expect<
  Equal<Record<string, never> extends Pick<ModelSnapshot, "model"> ? true : false, false>
>;

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
type CallRoute = `/api/studies/{workspace_id}/${ActionId}`;
type CallInput = paths[CallRoute]["post"]["requestBody"]["content"]["application/json"];
export type CanonicalCallAlternatives = Expect<Equal<NonNullable<CallInput["action"]>, ActionId>>;
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
export type CanonicalWorkspaces = Expect<
  Equal<
    MethodResponse<ReturnType<typeof createModelClient>, "get", "/api/workspaces">,
    Readonly<Record<string, string | null>>
  >
>;
type Upload = paths["/api/upload"]["post"]["requestBody"]["content"]["multipart/form-data"];
export type NativeUploadFile = Expect<Equal<Upload["file"], Blob>>;
type FitInput = FitCall["requestBody"]["content"]["application/json"];
export type RequestDefaultsMayBeOmitted = Expect<
  Extends<{ model_revision: string; panel_revision: string }, FitInput>
>;
export type DefaultsAreAbsent = Expect<
  Equal<
    Extends<{ model_revision: string; panel_revision: string; settings: undefined }, FitInput>,
    false
  >
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
