/** Signatures for the jStat distribution functions used by authored-law plots. */
declare module "jstat" {
  interface DistributionFunctions {
    pdf(value: number, first: number, second: number): number;
    cdf(value: number, first: number, second: number): number;
    inv(probability: number, first: number, second: number): number;
  }

  export const jStat: {
    normal: DistributionFunctions;
    beta: DistributionFunctions;
    gamma: DistributionFunctions;
    lognormal: DistributionFunctions;
    uniform: DistributionFunctions;
    exponential: {
      pdf(value: number, rate: number): number;
      inv(probability: number, rate: number): number;
    };
  };
}
