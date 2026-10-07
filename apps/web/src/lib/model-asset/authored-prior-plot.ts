import type { DensityCurve, NumPyroValue, NumPyroDistribution } from "@nof1-causal-lab/api-types";
import { readNumericalArray } from "@nof1-causal-lab/api-types";
import { jStat } from "jstat";

interface ScalarDensity {
  pdf(value: number): number;
  quantile(probability: number): number;
}

const isArray: (value: NumPyroValue) => value is readonly NumPyroValue[] = Array.isArray;

function object(value: NumPyroValue | undefined): NumPyroDistribution["params"] | null {
  return value !== undefined &&
    value !== null &&
    typeof value === "object" &&
    !isArray(value) &&
    !("npy" in value)
    ? value
    : null;
}

/** Native scalars may be JSON numbers or zero-dimensional array constructors. */
function scalar(value: NumPyroValue | undefined): number {
  if (typeof value === "number") return value;
  if (
    value !== null &&
    typeof value === "object" &&
    "npy" in value &&
    value.npy instanceof Uint8Array
  ) {
    const array = readNumericalArray({ npy: value.npy });
    if (array.shape.length !== 0)
      throw new Error("An authored prior plot requires scalar constructor arguments.");
    return Number(array.values[0]);
  }
  const encoded = object(value);
  if (encoded && "array" in encoded) return scalar(encoded.array);
  if (encoded?.float === "inf") return Infinity;
  if (encoded?.float === "-inf") return -Infinity;
  throw new Error("An authored prior plot requires scalar constructor arguments.");
}

function twoParameters(
  family: Pick<typeof jStat.normal, "pdf" | "inv">,
  first: number,
  second: number,
): ScalarDensity {
  return {
    pdf: (value) => family.pdf(value, first, second),
    quantile: (probability) => family.inv(probability, first, second),
  };
}

/** Normalize the declared truncation, reflecting right tails before subtracting CDFs. */
function truncatedNormal(loc: number, scale: number, low: number, high: number): ScalarDensity {
  const reflected = low > loc;
  const mean = reflected ? -loc : loc;
  const lower = reflected ? -high : low;
  const upper = reflected ? -low : high;
  const left = lower === -Infinity ? 0 : jStat.normal.cdf(lower, mean, scale);
  const right = upper === Infinity ? 1 : jStat.normal.cdf(upper, mean, scale);
  const mass = right - left;
  return {
    pdf: (value) => (value < low || value > high ? 0 : jStat.normal.pdf(value, loc, scale) / mass),
    quantile: (probability) => {
      const value = jStat.normal.inv(
        left + (reflected ? 1 - probability : probability) * mass,
        mean,
        scale,
      );
      return reflected ? -value : value;
    },
  };
}

/** Interpret the supported scalar constructor shapes without changing the saved law. */
function scalarDensity(law: NumPyroDistribution): ScalarDensity {
  const p = law.params;
  switch (law.distribution) {
    case "Normal":
      return twoParameters(jStat.normal, scalar(p.loc), scalar(p.scale));
    case "HalfNormal":
      return truncatedNormal(0, scalar(p.scale), 0, Infinity);
    case "Beta":
      return twoParameters(jStat.beta, scalar(p.concentration1), scalar(p.concentration0));
    case "Uniform":
      return twoParameters(jStat.uniform, scalar(p.low), scalar(p.high));
    case "Gamma": {
      const rate = scalar(p.rate);
      return twoParameters(jStat.gamma, scalar(p.concentration), 1 / rate);
    }
    case "LogNormal":
      return twoParameters(jStat.lognormal, scalar(p.loc), scalar(p.scale));
    case "Exponential": {
      const rate = scalar(p.rate);
      return {
        pdf: (value) => jStat.exponential.pdf(value, rate),
        quantile: (probability) => jStat.exponential.inv(probability, rate),
      };
    }
    case "LeftTruncatedDistribution":
    case "RightTruncatedDistribution":
    case "TwoSidedTruncatedDistribution": {
      const base = object(p.base_dist);
      const parameters = object(base?.params);
      if (base?.distribution !== "Normal" || parameters === null) {
        throw new Error("An authored truncated prior plot requires a Normal base law.");
      }
      return truncatedNormal(
        scalar(parameters.loc),
        scalar(parameters.scale),
        law.distribution === "RightTruncatedDistribution" ? -Infinity : scalar(p.low),
        law.distribution === "LeftTruncatedDistribution" ? Infinity : scalar(p.high),
      );
    }
    default:
      throw new Error(`Authored prior plots do not support the ${law.distribution} constructor.`);
  }
}

const plots = new WeakMap<NumPyroDistribution, DensityCurve>();

/** Plot the law's central 98%; every authoring family must produce a finite scalar PDF. */
export function authoredPriorPlot(law: NumPyroDistribution): DensityCurve {
  const cached = plots.get(law);
  if (cached) return cached;
  const distribution = scalarDensity(law);
  const low = distribution.quantile(0.01);
  const high = distribution.quantile(0.99);
  const x = Array.from({ length: 129 }, (_, index) => low + ((high - low) * index) / 128);
  const density = x.map((value) => distribution.pdf(value));
  if (
    !(low < high) ||
    !x.every(Number.isFinite) ||
    !density.every((value) => Number.isFinite(value) && value >= 0)
  ) {
    throw new Error(`Cannot plot a finite scalar density for the ${law.distribution} prior.`);
  }
  const result = { x, density };
  plots.set(law, result);
  return result;
}
