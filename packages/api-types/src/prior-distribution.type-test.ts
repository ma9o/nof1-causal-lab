import type {
  CompiledDistribution,
  ParameterId,
  ParameterSpec,
  NumPyroDistribution,
} from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;
type GammaRecipe = Extract<CompiledDistribution, { distribution: "Gamma" }>;

export type ParameterRetainsEvidence = Expect<Extends<ParameterSpec["prior_reasoning"], string>>;
export type ParameterOwnsIdentity = Expect<Extends<ParameterSpec["id"], ParameterId>>;
export type ParameterCarriesNativeLaw = Expect<
  Extends<NonNullable<ParameterSpec["prior"]>, NumPyroDistribution>
>;
export type CompiledRetainsTransforms = Expect<Extends<GammaRecipe["transforms"], unknown[]>>;
