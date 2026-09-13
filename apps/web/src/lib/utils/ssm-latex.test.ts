import type { NumPyroDistribution } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import {
  confounderGroupLatex,
  type LabeledLikelihood,
  distName,
  likelihoodLine,
  linkInverse,
  observationEquationLatex,
  observationParameterDefinitionLatex,
  observationParameterSymbol,
  observationPriorLatex,
  paramSymbol,
  priorLatex,
  priorLine,
  textify,
} from "./ssm-latex";
import type { ConfounderGroup } from "./ssm-latex";

describe("textify", () => {
  it("replaces underscores with spaces", () => {
    expect(textify("hello_world")).toBe("hello world");
  });

  it("handles no underscores", () => {
    expect(textify("hello")).toBe("hello");
  });

  it("handles multiple underscores", () => {
    expect(textify("a_b_c_d")).toBe("a b c d");
  });
});

describe("linkInverse", () => {
  it("identity returns predictor unchanged", () => {
    expect(linkInverse("identity", "\\mu")).toBe("\\mu");
  });

  it("log returns exp wrapper", () => {
    expect(linkInverse("log", "x")).toBe("\\exp(x)");
  });

  it("logit returns sigma wrapper", () => {
    expect(linkInverse("logit", "x")).toBe("\\sigma(x)");
  });

  it("probit returns Phi wrapper", () => {
    expect(linkInverse("probit", "x")).toBe("\\Phi(x)");
  });

  it("unknown link returns generic inverse", () => {
    expect(linkInverse("unknown", "x")).toBe("g^{-1}(x)");
  });
});

describe("distName", () => {
  it("maps gaussian to mathcal N", () => {
    expect(distName("gaussian")).toBe("\\mathcal{N}");
  });

  it("maps poisson", () => {
    expect(distName("poisson")).toBe("\\text{Poisson}");
  });

  it("maps bernoulli", () => {
    expect(distName("bernoulli")).toBe("\\text{Bernoulli}");
  });

  it("unknown dist uses text wrapper", () => {
    expect(distName("exotic")).toBe("\\text{exotic}");
  });
});

describe("likelihoodLine", () => {
  it("renders gaussian likelihood", () => {
    const lik = {
      standardized: false,
      reasoning: "",
      sources: [],
      indicator_id: "indicator:mood",
      label: "mood",
      distribution: "gaussian",
      link: "identity",
    } as LabeledLikelihood;
    const result = likelihoodLine(lik);
    expect(result).toContain("\\mathcal{N}");
    expect(result).toContain("\\sigma");
  });

  it("renders poisson likelihood", () => {
    const lik = {
      standardized: false,
      reasoning: "",
      sources: [],
      indicator_id: "indicator:steps",
      label: "steps",
      distribution: "poisson",
      link: "log",
    } as LabeledLikelihood;
    const result = likelihoodLine(lik);
    expect(result).toContain("\\text{Poisson}");
    expect(result).toContain("\\exp");
  });

  it("renders beta likelihood with phi", () => {
    const lik = {
      standardized: false,
      reasoning: "",
      sources: [],
      indicator_id: "indicator:ratio",
      label: "ratio",
      distribution: "beta",
      link: "logit",
    } as LabeledLikelihood;
    const result = likelihoodLine(lik);
    expect(result).toContain("\\text{Beta}");
    expect(result).toContain("\\phi");
  });

  it("renders gamma likelihood with explicit shape-scale notation", () => {
    const lik = {
      standardized: false,
      reasoning: "",
      sources: [],
      indicator_id: "indicator:last_activity_clock_time",
      label: "last_activity_clock_time",
      distribution: "gamma",
      link: "log",
    } as LabeledLikelihood;
    const result = likelihoodLine(lik);
    expect(result).toContain("\\text{Gamma}");
    expect(result).toContain("\\kappa");
    expect(result).toContain("/\\kappa");
  });

  it("renders non-gaussian measurement error inside the likelihood when requested", () => {
    const lik = {
      standardized: false,
      reasoning: "",
      sources: [],
      indicator_id: "indicator:sleep_problem_search_count",
      label: "sleep_problem_search_count",
      distribution: "negative_binomial",
      link: "log",
    } as LabeledLikelihood;
    const result = likelihoodLine(lik, "sleep_quality", { includeMeasurementError: true });
    expect(result).toContain("\\text{NegBin}");
    expect(result).toContain("\\sigma_{\\text{sleep problem search count}}^{2}");
    expect(result).not.toContain("\\text{measurement-error SD}");
  });

  it("includes construct name when provided", () => {
    const lik = {
      standardized: false,
      reasoning: "",
      sources: [],
      indicator_id: "indicator:mood",
      label: "mood",
      distribution: "gaussian",
      link: "identity",
    } as LabeledLikelihood;
    const result = likelihoodLine(lik, "affect");
    expect(result).toContain("\\lambda");
    expect(result).toContain("affect");
  });
});

describe("paramSymbol", () => {
  it("converts beta prefix to greek", () => {
    expect(paramSymbol("beta_X_Y")).toContain("\\beta");
    expect(paramSymbol("beta_X_Y")).toContain("X Y");
  });

  it("converts sigma prefix to greek", () => {
    expect(paramSymbol("sigma_mood")).toContain("\\sigma");
    expect(paramSymbol("sigma_mood")).toContain("mood");
  });

  it("handles t0_mean prefix", () => {
    const result = paramSymbol("t0_mean_stress");
    expect(result).toContain("\\mu_{0");
    expect(result).toContain("stress");
  });

  it("handles t0_sd prefix", () => {
    const result = paramSymbol("t0_sd_sleep");
    expect(result).toContain("\\sigma_{0");
    expect(result).toContain("sleep");
  });

  it("unknown prefix uses text wrapper", () => {
    expect(paramSymbol("custom_param")).toContain("\\text{custom param}");
  });

  it("cor maps to psi", () => {
    expect(paramSymbol("cor_X_Y")).toContain("\\psi");
  });
});

describe("observationParameterSymbol", () => {
  it("maps measurement error to the row-relative sigma symbol", () => {
    expect(
      observationParameterSymbol({
        parameterName: "obs_sd_sleep",
        likelihood: {
          standardized: false,
          reasoning: "",
          sources: [],
          indicator_id: "indicator:sleep",
          label: "sleep",
          distribution: "gaussian",
          link: "identity",
        } as LabeledLikelihood,
      }),
    ).toBe("\\sigma_{\\text{sleep}}");
  });

  it("maps beta concentration to phi", () => {
    expect(
      observationParameterSymbol({
        parameterName: "obs_concentration",
        likelihood: {
          standardized: false,
          reasoning: "",
          sources: [],
          indicator_id: "indicator:appointment_attendance",
          label: "appointment_attendance",
          distribution: "beta",
          link: "logit",
        } as LabeledLikelihood,
      }),
    ).toBe("\\phi");
  });
});

describe("priorLine", () => {
  it("renders normal prior", () => {
    const prior = {
      distribution: "Normal",
      params: { loc: 0, scale: 1 },
      sources: [],
      reasoning: "",
    } as NumPyroDistribution;
    const result = priorLine(prior, "beta_X_Y");
    expect(result).toContain("\\beta");
    expect(result).toContain("\\mathcal{N}");
    expect(result).toContain("0");
    expect(result).toContain("1");
  });

  it("renders half-normal prior", () => {
    const prior = {
      distribution: "HalfNormal",
      params: { scale: 2 },
      sources: [],
      reasoning: "",
    } as NumPyroDistribution;
    const result = priorLine(prior, "sigma_mood");
    expect(result).toContain("\\text{HalfNormal}");
  });

  it("renders log-normal prior", () => {
    const prior = {
      distribution: "LogNormal",
      params: { loc: 0, scale: 0.5 },
      sources: [],
      reasoning: "",
    } as NumPyroDistribution;
    const result = priorLine(prior, "sigma_mood");
    expect(result).toContain("\\text{LogNormal}");
  });
});

describe("priorLatex", () => {
  it("strips alignment markers from priorLine output", () => {
    const prior = {
      distribution: "Normal",
      params: { loc: 0, scale: 1 },
      sources: [],
      reasoning: "",
    } as NumPyroDistribution;
    const result = priorLatex(prior, "beta_X_Y");
    expect(result).not.toContain("&");
    expect(result).toContain("\\sim");
    expect(result).toContain("\\mathcal{N}");
  });
});

describe("observationPriorLatex", () => {
  it("renders observation-family priors with row-relative symbols", () => {
    const result = observationPriorLatex({
      parameterName: "obs_concentration",
      prior: {
        distribution: "Gamma",
        params: { concentration: 5, rate: 0.5 },
        sources: [],
        reasoning: "",
      } as NumPyroDistribution,
      likelihood: {
        standardized: false,
        reasoning: "",
        sources: [],
        indicator_id: "indicator:appointment_attendance",
        label: "appointment_attendance",
        distribution: "beta",
        link: "logit",
      } as LabeledLikelihood,
    });

    expect(result).toContain("\\phi");
    expect(result).toContain("\\text{Gamma}");
    expect(result).not.toContain("obs concentration");
  });
});

describe("observationParameterDefinitionLatex", () => {
  it("describes non-gaussian measurement error with the row-relative sigma symbol", () => {
    expect(
      observationParameterDefinitionLatex({
        parameterName: "obs_sd_sleep_problem_search_count",
        likelihood: {
          standardized: false,
          reasoning: "",
          sources: [],
          indicator_id: "indicator:sleep_problem_search_count",
          label: "sleep_problem_search_count",
          distribution: "negative_binomial",
          link: "log",
        } as LabeledLikelihood,
      }),
    ).toContain("\\sigma_{\\text{sleep problem search count}}");
  });
});

describe("observationEquationLatex", () => {
  it("renders gamma shape directly in the main likelihood line", () => {
    const result = observationEquationLatex({
      likelihood: {
        standardized: false,
        reasoning: "",
        sources: [],
        indicator_id: "indicator:last_activity_clock_time",
        label: "last_activity_clock_time",
        distribution: "gamma",
        link: "log",
      } as LabeledLikelihood,
      parameterNames: ["obs_shape"],
    });

    expect(result).toContain("\\text{Gamma}");
    expect(result).toContain("\\kappa");
    expect(result).toContain("/\\kappa");
  });

  it("moves non-gaussian measurement error into the main likelihood line when obs_sd is present", () => {
    const result = observationEquationLatex({
      likelihood: {
        standardized: false,
        reasoning: "",
        sources: [],
        indicator_id: "indicator:sleep_problem_search_count",
        label: "sleep_problem_search_count",
        distribution: "negative_binomial",
        link: "log",
      } as LabeledLikelihood,
      constructName: "sleep_quality",
      parameterNames: [
        "lambda_sleep_problem_search_count_sleep_quality",
        "obs_sd_sleep_problem_search_count",
        "obs_r",
      ],
    });

    expect(result).toContain("\\text{NegBin}");
    expect(result).toContain("\\lambda");
    expect(result).toContain("r,\\;");
    expect(result).toContain("\\sigma_{\\text{sleep problem search count}}^{2}");
    expect(result).not.toContain("\\text{measurement-error SD}");
  });
});

describe("confounderGroupLatex", () => {
  it("renders aligned LaTeX block", () => {
    const group: ConfounderGroup = {
      confounder: "genetics",
      states: ["stress", "sleep"],
      pairs: [{ s1: "stress", s2: "sleep" }],
    };
    const result = confounderGroupLatex(group);
    expect(result).toContain("\\begin{aligned}");
    expect(result).toContain("\\end{aligned}");
    expect(result).toContain("genetics");
    expect(result).toContain("\\varepsilon");
    expect(result).toContain("\\psi");
  });
});
