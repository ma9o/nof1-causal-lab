import type {
  DistributionId,
  ModelSpec,
  NumPyroDistribution,
  ParameterId,
  ParameterSpec,
} from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;

// @ts-expect-error Authoring evidence belongs in transition logs.
export type NoParameterProvenance = ParameterSpec["prior_reasoning"];
export type ParameterOwnsIdentity = Expect<Extends<ParameterSpec["id"], ParameterId>>;
export type ParameterCarriesNativeLaw = Expect<
  Extends<NonNullable<ParameterSpec["distribution"]>, NumPyroDistribution | DistributionId>
>;
export type ModelOwnsParameters = Expect<Extends<ModelSpec["parameters"][number], ParameterSpec>>;
