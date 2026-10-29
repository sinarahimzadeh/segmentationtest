export const AI_MODEL_IDS = [
  "gpt-6.1-sol",
  "gpt-6.1-terra",
  "gpt-6-luna",
  "gemini-3.8-flash",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
] as const;

export type AiModelId = (typeof AI_MODEL_IDS)[number];
export type AiProvider = "openai" | "google";
export type ReasoningEffort = "minimal" | "low" | "medium" | "high";

export type AiModelOption = {
  id: AiModelId;
  provider: AiProvider;
  label: string;
  comparisonEfforts: readonly ReasoningEffort[];
  inputPricePerMillionTokens: number;
  cachedInputPricePerMillionTokens: number;
  outputPricePerMillionTokens: number;
  freeTierEligible: boolean;
};

/** One shared whitelist drives the dropdown, API validation, and cost estimate. */
export const AI_MODEL_OPTIONS: readonly AiModelOption[] = [
  {
    id: "gpt-6.1-sol",
    provider: "openai",
    label: "GPT-6.1 Sol",
    // Sol Medium is the fixed baseline, so only Low is a comparison choice.
    comparisonEfforts: ["low"],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 10,
    freeTierEligible: false,
  },
  {
    id: "gpt-6.1-terra",
    provider: "openai",
    label: "GPT-6.1 Terra",
    comparisonEfforts: ["low", "medium"],
    inputPricePerMillionTokens: 1,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 5,
    freeTierEligible: false,
  },
  {
    id: "gpt-6-luna",
    provider: "openai",
    label: "GPT-6 Luna",
    comparisonEfforts: ["low", "medium"],
    inputPricePerMillionTokens: 0.1,
    cachedInputPricePerMillionTokens: 0.01,
    outputPricePerMillionTokens: 0.5,
    freeTierEligible: false,
  },
  {
    id: "gemini-3.8-flash",
    provider: "google",
    label: "Gemini 3.8 Flash",
    comparisonEfforts: ["low", "medium", "high"],
    // Current Standard paid prices through 31 December 2026. Free-tier calls
    // can cost $0; we retain this paid equivalent for later cost comparisons.
    inputPricePerMillionTokens: 0.75,
    cachedInputPricePerMillionTokens: 0.075,
    outputPricePerMillionTokens: 3.75,
    freeTierEligible: true,
  },
  {
    id: "gemini-3.5-flash-lite",
    provider: "google",
    label: "Gemini 3.5 Flash-Lite",
    comparisonEfforts: ["minimal", "low", "medium", "high"],
    // Current Google Standard paid prices. Free-tier calls can cost $0;
    // retain the paid equivalent so benchmark costs remain comparable.
    inputPricePerMillionTokens: 0.3,
    cachedInputPricePerMillionTokens: 0.03,
    outputPricePerMillionTokens: 2.5,
    freeTierEligible: true,
  },
  {
    id: "gemini-3.1-flash-lite",
    provider: "google",
    label: "Gemini 3.1 Flash-Lite",
    comparisonEfforts: ["minimal", "low", "medium", "high"],
    inputPricePerMillionTokens: 0.25,
    cachedInputPricePerMillionTokens: 0.025,
    outputPricePerMillionTokens: 1.5,
    freeTierEligible: true,
  },
];

export const BASELINE_SELECTION = {
  model: "gpt-6.1-sol" as const,
  reasoningEffort: "medium" as const,
};

export function getAiModelOption(modelId: AiModelId) {
  const option = AI_MODEL_OPTIONS.find((item) => item.id === modelId);
  if (!option) throw new Error(`Unsupported AI model: ${modelId}`);
  return option;
}

export function isAllowedComparisonSelection(
  modelId: AiModelId,
  reasoningEffort: ReasoningEffort,
) {
  return getAiModelOption(modelId).comparisonEfforts.includes(reasoningEffort);
}

export function getModelRunLabel(
  modelId: AiModelId,
  reasoningEffort: ReasoningEffort,
) {
  return `${getAiModelOption(modelId).label} · ${
    reasoningEffort[0].toUpperCase() + reasoningEffort.slice(1)
  }`;
}

export function estimatePaidCostUsd({
  modelId,
  inputTokens,
  cachedInputTokens,
  outputTokens,
}: {
  modelId: AiModelId;
  inputTokens: number | null;
  cachedInputTokens: number | null;
  outputTokens: number | null;
}) {
  if (inputTokens === null || outputTokens === null) return null;
  const prices = getAiModelOption(modelId);
  const safeCachedTokens = Math.min(
    Math.max(cachedInputTokens ?? 0, 0),
    inputTokens,
  );
  const uncachedInputTokens = inputTokens - safeCachedTokens;

  return (
    (uncachedInputTokens / 1_000_000) *
      prices.inputPricePerMillionTokens +
    (safeCachedTokens / 1_000_000) *
      prices.cachedInputPricePerMillionTokens +
    (outputTokens / 1_000_000) * prices.outputPricePerMillionTokens
  );
}
