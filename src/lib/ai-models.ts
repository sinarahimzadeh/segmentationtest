export const AI_MODEL_IDS = [
  "gpt-6.1-sol",
  "gpt-6.1-terra",
  "gpt-6-sol",
  "gpt-6-luna",
  "gpt-5.5",
  "grok-4.7",
  "grok-4.6",
  "grok-4.3",
  "gemini-4-argon",
  "gemini-3.8-flash",
  "gemini-3.1-pro-preview",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
  "claude-fable-5-1",
  "claude-opus-5-5",
  "claude-sonnet-5-5",
  "claude-haiku-5-5",
  "mistral-large-4",
  "mistral-medium-3-5",
  "mistral-large-2512",
  "openrouter-mistral-large",
  "openrouter-mistral-small",
  "openrouter-ministral-8b",
  "openrouter-qwen-3-7-flash",
  "openrouter-qwen-3-8-flash",
  "openrouter-qwen-3-vl-30b",
  "openrouter-deepseek-v4-flash",
  "openrouter-gpt-6.1-terra",
  "openrouter-gpt-6.1-sol",
  "openrouter-gpt-6-sol",
  "openrouter-gpt-6-luna",
  "openrouter-gpt-5.5",
  // Keep the old IDs so benchmark JSON created before the 5.5 upgrade loads.
  "claude-opus-5",
  "claude-sonnet-5",
  "claude-haiku-4-5-20251001",
] as const;

export type AiModelId = (typeof AI_MODEL_IDS)[number];
export type AiProvider = "openai" | "xai" | "google" | "anthropic" | "mistral" | "openrouter";
export type ReasoningEffort = "default" | "none" | "minimal" | "low" | "medium" | "high";
export const PROCESSING_SPEEDS = ["standard", "fast", "ultrafast"] as const;
export type ProcessingSpeed = (typeof PROCESSING_SPEEDS)[number];

export type AiModelOption = {
  id: AiModelId;
  provider: AiProvider;
  label: string;
  comparisonEfforts: readonly ReasoningEffort[];
  inputPricePerMillionTokens: number;
  cachedInputPricePerMillionTokens: number;
  outputPricePerMillionTokens: number;
  apiModelName?: string;
  // Only OpenAI models selected through OpenRouter offer this speed selector.
  speedPriceMultipliers?: Partial<Record<ProcessingSpeed, number>>;
};

/** One shared whitelist drives the dropdown, API validation, and cost estimate. */
export const AI_MODEL_OPTIONS: readonly AiModelOption[] = [
  {
    id: "gpt-6.1-sol",
    provider: "openai",
    label: "GPT-6.1 Sol",
    // Sol Medium is the fixed baseline; High can be compared with it.
    comparisonEfforts: ["low", "high"],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "gpt-6.1-terra",
    provider: "openai",
    label: "GPT-6.1 Terra",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 1,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 5,
  },
  {
    id: "gpt-6-sol",
    provider: "openai",
    label: "GPT-6 Sol",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "gpt-6-luna",
    provider: "openai",
    label: "GPT-6 Luna",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 0.1,
    cachedInputPricePerMillionTokens: 0.01,
    outputPricePerMillionTokens: 0.5,
  },
  {
    id: "gpt-5.5",
    provider: "openai",
    label: "GPT-5.5",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 5,
    cachedInputPricePerMillionTokens: 0.5,
    outputPricePerMillionTokens: 30,
  },
  {
    id: "grok-4.7",
    provider: "xai",
    label: "Grok 4.7",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.5,
    outputPricePerMillionTokens: 6,
  },
  {
    id: "grok-4.6",
    provider: "xai",
    label: "Grok 4.6",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.5,
    outputPricePerMillionTokens: 6,
  },
  {
    id: "grok-4.3",
    provider: "xai",
    label: "Grok 4.3",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 1.25,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 2.5,
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
    id: "gemini-3.1-pro-preview",
    provider: "google",
    label: "Gemini 3.1 Pro Preview",
    comparisonEfforts: ["low", "medium", "high"],
    // Standard paid prices for prompts under 200K tokens.
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 12,
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
    id: "mistral-large-4",
    provider: "mistral",
    label: "Mistral Large 4 (deprecated direct)",
    comparisonEfforts: [],
    inputPricePerMillionTokens: 0.7,
    cachedInputPricePerMillionTokens: 0.07,
    outputPricePerMillionTokens: 2.1,
  },
  {
    id: "mistral-medium-3-5",
    provider: "mistral",
    label: "Mistral Medium 3.5 (deprecated direct)",
    comparisonEfforts: [],
    inputPricePerMillionTokens: 1.5,
    cachedInputPricePerMillionTokens: 0.15,
    outputPricePerMillionTokens: 7.5,
  },
  {
    id: "mistral-large-2512",
    provider: "mistral",
    label: "Mistral Large 3 (deprecated direct)",
    comparisonEfforts: [],
    inputPricePerMillionTokens: 0.5,
    cachedInputPricePerMillionTokens: 0.05,
    outputPricePerMillionTokens: 1.5,
  },
  {
    id: "openrouter-mistral-large",
    provider: "openrouter",
    label: "Mistral Large 4 (OpenRouter)",
    apiModelName: "mistralai/mistral-large-4-0",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 0.7,
    cachedInputPricePerMillionTokens: 0.07,
    outputPricePerMillionTokens: 2.1,
  },
  {
    id: "openrouter-mistral-small",
    provider: "openrouter",
    label: "Mistral Small (OpenRouter)",
    apiModelName: "mistralai/mistral-small-2603",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 0.15,
    cachedInputPricePerMillionTokens: 0.015,
    outputPricePerMillionTokens: 0.6,
  },
  {
    id: "openrouter-ministral-8b",
    provider: "openrouter",
    label: "Ministral 8B (OpenRouter)",
    apiModelName: "mistralai/ministral-8b-2512",
    comparisonEfforts: ["default"],
    inputPricePerMillionTokens: 0.15,
    cachedInputPricePerMillionTokens: 0.015,
    outputPricePerMillionTokens: 0.15,
  },
  {
    id: "openrouter-qwen-3-7-flash",
    provider: "openrouter",
    label: "Qwen 3.7 Flash (OpenRouter)",
    apiModelName: "qwen/qwen3.7-flash",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 0.03,
    cachedInputPricePerMillionTokens: 0.003,
    outputPricePerMillionTokens: 0.13,
  },
  {
    id: "openrouter-qwen-3-8-flash",
    provider: "openrouter",
    label: "Qwen 3.8 Flash (OpenRouter)",
    apiModelName: "qwen/qwen3.8-flash",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 0.15,
    cachedInputPricePerMillionTokens: 0.015,
    outputPricePerMillionTokens: 0.47,
  },
  {
    id: "openrouter-qwen-3-vl-30b",
    provider: "openrouter",
    label: "Qwen 3 VL 30B (OpenRouter)",
    apiModelName: "qwen/qwen3-vl-30b-a3b-instruct",
    comparisonEfforts: ["default"],
    inputPricePerMillionTokens: 0.15,
    cachedInputPricePerMillionTokens: 0.015,
    outputPricePerMillionTokens: 0.6,
  },
  {
    id: "openrouter-deepseek-v4-flash",
    provider: "openrouter",
    label: "DeepSeek V4.1 Flash (OpenRouter)",
    apiModelName: "deepseek/deepseek-v4.1-flash",
    comparisonEfforts: ["low", "medium", "high"],
    inputPricePerMillionTokens: 0.045,
    cachedInputPricePerMillionTokens: 0.005,
    outputPricePerMillionTokens: 0.5,
  },
  {
    id: "openrouter-gpt-6.1-terra",
    provider: "openrouter",
    label: "GPT-6.1 Terra (OpenRouter)",
    apiModelName: "openai/gpt-6.1-terra",
    comparisonEfforts: ["low", "medium", "high"],
    speedPriceMultipliers: { fast: 2 },
    inputPricePerMillionTokens: 1,
    cachedInputPricePerMillionTokens: 0.05,
    outputPricePerMillionTokens: 5,
  },
  {
    id: "openrouter-gpt-6.1-sol",
    provider: "openrouter",
    label: "GPT-6.1 Sol (OpenRouter)",
    apiModelName: "openai/gpt-6.1-sol",
    comparisonEfforts: ["low", "medium", "high"],
    speedPriceMultipliers: { fast: 2, ultrafast: 6 },
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.1,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "openrouter-gpt-6-sol",
    provider: "openrouter",
    label: "GPT-6 Sol (OpenRouter)",
    apiModelName: "openai/gpt-6-sol",
    comparisonEfforts: ["low", "medium", "high"],
    speedPriceMultipliers: { fast: 2 },
    inputPricePerMillionTokens: 2,
    cachedInputPricePerMillionTokens: 0.2,
    outputPricePerMillionTokens: 10,
  },
  {
    id: "openrouter-gpt-6-luna",
    provider: "openrouter",
    label: "GPT-6 Luna (OpenRouter)",
    apiModelName: "openai/gpt-6-luna",
    comparisonEfforts: ["low", "medium", "high"],
    speedPriceMultipliers: { fast: 2 },
    inputPricePerMillionTokens: 0.1,
    cachedInputPricePerMillionTokens: 0.01,
    outputPricePerMillionTokens: 0.5,
  },
  {
    id: "openrouter-gpt-5.5",
    provider: "openrouter",
    label: "GPT-5.5 (OpenRouter)",
    apiModelName: "openai/gpt-5.5",
    comparisonEfforts: ["low", "medium", "high"],
    speedPriceMultipliers: { fast: 2.5 },
    inputPricePerMillionTokens: 5,
    cachedInputPricePerMillionTokens: 0.5,
    outputPricePerMillionTokens: 30,
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
  processingSpeed: ProcessingSpeed = "standard",
) {
  return `${getAiModelOption(modelId).label} · ${
    reasoningEffort[0].toUpperCase() + reasoningEffort.slice(1)
  }${processingSpeed === "standard" ? "" : ` · ${getProcessingSpeedLabel(processingSpeed)}`}`;
}

export function getProcessingSpeedLabel(speed: ProcessingSpeed) {
  return speed === "ultrafast" ? "Ultrafast" : speed === "fast" ? "Fast" : "Standard";
}

export function getProcessingSpeeds(modelId: AiModelId) {
  const prices = getAiModelOption(modelId).speedPriceMultipliers;
  return PROCESSING_SPEEDS.filter((speed) => speed === "standard" || prices?.[speed] !== undefined);
}

export function getSpeedPriceMultiplier(modelId: AiModelId, speed: ProcessingSpeed) {
  if (speed === "standard") return 1;
  const multiplier = getAiModelOption(modelId).speedPriceMultipliers?.[speed];
  if (multiplier === undefined) throw new Error(`${getAiModelOption(modelId).label} does not offer ${getProcessingSpeedLabel(speed)}.`);
  return multiplier;
}

export function estimatePaidCostUsd({
  modelId,
  inputTokens,
  cachedInputTokens,
  outputTokens,
  processingSpeed = "standard",
}: {
  modelId: AiModelId;
  inputTokens: number | null;
  cachedInputTokens: number | null;
  outputTokens: number | null;
  processingSpeed?: ProcessingSpeed;
}) {
  if (inputTokens === null || outputTokens === null) return null;
  const prices = getAiModelOption(modelId);
  const safeCachedTokens = Math.min(
    Math.max(cachedInputTokens ?? 0, 0),
    inputTokens,
  );
  const uncachedInputTokens = inputTokens - safeCachedTokens;

  return getSpeedPriceMultiplier(modelId, processingSpeed) * (
    (uncachedInputTokens / 1_000_000) *
      prices.inputPricePerMillionTokens +
    (safeCachedTokens / 1_000_000) *
      prices.cachedInputPricePerMillionTokens +
    (outputTokens / 1_000_000) * prices.outputPricePerMillionTokens
  );
}
