import type { NumPyroDistribution } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { paramSymbol, priorLatex, priorLine, textify } from "./ssm-latex";

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
