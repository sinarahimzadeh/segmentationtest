export const AI_MODEL_IDS = [
  "gpt-6.1-sol",
  "gpt-6.1-terra",
  "gpt-6-luna",
  "gemini-4-argon",
  "gemini-3.8-flash",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
  "claude-fable-5-1",
  "claude-opus-5-5",
  "claude-sonnet-5-5",
  "claude-haiku-5-5",
  // Keep the old IDs so benchmark JSON created before the 5.5 upgrade loads.
  "claude-opus-5",
  "claude-sonnet-5",
  "claude-haiku-4-5-20251001",
] as const;

export type AiModelId = (typeof AI_MODEL_IDS)[number];
export type AiProvider = "openai" | "google" | "anthropic";
export type ReasoningEffort = "minimal" | "low" | "medium" | "high";

export type AiModelOption = {
  id: AiModelId;
  provider: AiProvider;
  label: string;
  comparisonEfforts: readonly ReasoningEffort[];
  inputPricePerMillionTokens: number;
  cachedInputPricePerMillionTokens: number;
  outputPricePerMillionTokens: number;
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
  },
  {
    id: "gpt-6.1-terra",
    provider: "openai",
    label: "GPT-6.1 Terra",
    comparisonEfforts: ["low", "medium"],
    inputPricePerMillionTokens: 1,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 5,
  },
  {
    id: "gpt-6-luna",
    provider: "openai",
    label: "GPT-6 Luna",
    comparisonEfforts: ["low", "medium"],
    inputPricePerMillionTokens: 0.1,
    cachedInputPricePerMillionTokens: 0.01,
    outputPricePerMillionTokens: 0.5,
  },
  {
    id: "gemini-4-argon",
    provider: "google",
    label: "Gemini 4 Argon",
    comparisonEfforts: ["low", "medium", "high"],
    // Google Standard paid prices through November 30, 2026. Argon is not
    // available on Google's free tier, so every processed token is billable.
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "gemini-3.8-flash",
    provider: "google",
    label: "Gemini 3.8 Flash",
    comparisonEfforts: ["low", "medium", "high"],
    // Standard paid prices used for benchmark cost comparisons.
    inputPricePerMillionTokens: 0.75,
    cachedInputPricePerMillionTokens: 0.075,
    outputPricePerMillionTokens: 3.75,
  },
  {
    id: "gemini-3.5-flash-lite",
    provider: "google",
    label: "Gemini 3.5 Flash-Lite",
    comparisonEfforts: ["minimal", "low", "medium", "high"],
    // Standard paid prices used for benchmark cost comparisons.
    inputPricePerMillionTokens: 0.3,
    cachedInputPricePerMillionTokens: 0.03,
    outputPricePerMillionTokens: 2.5,
  },
  {
    id: "gemini-3.1-flash-lite",
    provider: "google",
    label: "Gemini 3.1 Flash-Lite",
    comparisonEfforts: ["minimal", "low", "medium", "high"],
    inputPricePerMillionTokens: 0.25,
    cachedInputPricePerMillionTokens: 0.025,
    outputPricePerMillionTokens: 1.5,
  },
  {
    id: "claude-fable-5-1",
    provider: "anthropic",
    label: "Claude Fable 5.1",
    // Anthropic's output_config.effort supports these three benchmark levels.
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 10,
    cachedInputPricePerMillionTokens: 0.25,
    outputPricePerMillionTokens: 50,
  },
  {
    id: "claude-opus-5-5",
    provider: "anthropic",
    label: "Claude Opus 5.5",
    // Match the requested benchmark list: Opus Low and Medium only.
    comparisonEfforts: ["low", "medium"],
    inputPricePerMillionTokens: 4,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 20,
  },
  {
    id: "claude-sonnet-5-5",
    provider: "anthropic",
    label: "Claude Sonnet 5.5",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "claude-haiku-5-5",
    provider: "anthropic",
    label: "Claude Haiku 5.5",
    comparisonEfforts: ["low", "medium", "high"],
    // This benchmark is far below Haiku's 100K-input long-context threshold.
    inputPricePerMillionTokens: 0.1,
    cachedInputPricePerMillionTokens: 0.01,
    outputPricePerMillionTokens: 0.5,
  },
  {
    id: "claude-opus-5",
    provider: "anthropic",
    label: "Claude Opus 5 (old saved runs)",
    comparisonEfforts: [],
    inputPricePerMillionTokens: 5,
    cachedInputPricePerMillionTokens: 0.5,
    outputPricePerMillionTokens: 25,
  },
  {
    id: "claude-sonnet-5",
    provider: "anthropic",
    label: "Claude Sonnet 5 (old saved runs)",
    comparisonEfforts: [],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "claude-haiku-4-5-20251001",
    provider: "anthropic",
    label: "Claude Haiku 4.5 (old saved runs)",
    comparisonEfforts: [],
    inputPricePerMillionTokens: 1,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 5,
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
