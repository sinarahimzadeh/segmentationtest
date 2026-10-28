export const OPENAI_MODEL_IDS = [
  "gpt-6.1-sol",
  "gpt-6.1-terra",
  "gpt-6-luna",
] as const;

export type OpenAIModelId = (typeof OPENAI_MODEL_IDS)[number];
export type ReasoningEffort = "low" | "medium";

export type OpenAIModelOption = {
  id: OpenAIModelId;
  label: string;
  comparisonEfforts: readonly ReasoningEffort[];
  inputPricePerMillionTokens: number;
  cachedInputPricePerMillionTokens: number;
  outputPricePerMillionTokens: number;
};

/**
 * The model list used by both the browser dropdown and the API route.
 * Keeping one shared whitelist prevents the UI and server from disagreeing.
 */
export const OPENAI_MODEL_OPTIONS: readonly OpenAIModelOption[] = [
  {
    id: "gpt-6.1-sol",
    label: "GPT-6.1 Sol",
    // Sol Medium is deliberately absent here because it has its own fixed
    // baseline button. The dropdown is for candidates compared to that baseline.
    comparisonEfforts: ["low"],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "gpt-6.1-terra",
    label: "GPT-6.1 Terra",
    comparisonEfforts: ["low", "medium"],
    inputPricePerMillionTokens: 1,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 5,
  },
  {
    id: "gpt-6-luna",
    label: "GPT-6 Luna",
    comparisonEfforts: ["low", "medium"],
    inputPricePerMillionTokens: 0.1,
    cachedInputPricePerMillionTokens: 0.01,
    outputPricePerMillionTokens: 0.5,
  },
];

export const BASELINE_SELECTION = {
  model: "gpt-6.1-sol" as const,
  reasoningEffort: "medium" as const,
};

export function getOpenAIModelOption(modelId: OpenAIModelId) {
  const option = OPENAI_MODEL_OPTIONS.find((item) => item.id === modelId);
  if (!option) throw new Error(`Unsupported OpenAI model: ${modelId}`);
  return option;
}

export function isAllowedComparisonSelection(
  modelId: OpenAIModelId,
  reasoningEffort: ReasoningEffort,
) {
  return getOpenAIModelOption(modelId).comparisonEfforts.includes(
    reasoningEffort,
  );
}

export function getModelRunLabel(
  modelId: OpenAIModelId,
  reasoningEffort: ReasoningEffort,
) {
  return `${getOpenAIModelOption(modelId).label} · ${
    reasoningEffort[0].toUpperCase() + reasoningEffort.slice(1)
  }`;
}

export function estimateOpenAICostUsd({
  modelId,
  inputTokens,
  cachedInputTokens,
  outputTokens,
}: {
  modelId: OpenAIModelId;
  inputTokens: number | null;
  cachedInputTokens: number | null;
  outputTokens: number | null;
}) {
  if (inputTokens === null || outputTokens === null) return null;
  const prices = getOpenAIModelOption(modelId);
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
