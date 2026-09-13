import type { CompiledDistribution, ParameterId, PriorProposal } from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;
type NormalPrior = Extract<PriorProposal, { distribution: "Normal" }>;
type GammaRecipe = Extract<CompiledDistribution, { distribution: "Gamma" }>;
type NormalArgs = NormalPrior["params"];
type GammaArgs = GammaRecipe["params"];

export type PriorRetainsEvidence = Expect<Extends<NormalPrior["reasoning"], string>>;
export type PriorRetainsIdentity = Expect<Extends<NormalPrior["parameter_id"], ParameterId>>;
export type NormalRequiresLocation = Expect<Extends<NormalPrior["params"]["mu"], number>>;
export type CompiledRetainsTransforms = Expect<Extends<GammaRecipe["transforms"], unknown[]>>;

// @ts-expect-error A Normal distribution requires both location and scale.
export type InvalidNormalArgs = Expect<Extends<{ sigma: number }, NormalArgs>>;
// @ts-expect-error Gamma arguments cannot be replaced with Normal arguments.
export type InvalidGammaArgs = Expect<Extends<NormalArgs, GammaArgs>>;
