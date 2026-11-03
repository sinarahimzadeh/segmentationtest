import type { AiProvider, ReasoningEffort } from "./ai-models";

export const CAPTURE_SIZE = 1024;

export type ImagePoint = [number, number];

export type Detection = {
  label: "scanbody";
  outline: ImagePoint[];
  confidence: number;
};

export type ViewDetections = {
  view: string;
  detections: Detection[];
};

export type AnalysisResult = {
  views: ViewDetections[];
  meta: {
    model: string;
    modelLabel: string;
    provider: AiProvider;
    // Present only on new Claude runs so direct and LangChain results are distinct.
    integration?: "langchain" | "anthropic-sdk";
    reasoningEffort: ReasoningEffort;
    elapsedMs: number;
    inputTokens: number | null;
    cachedInputTokens: number | null;
    outputTokens: number | null;
    // Calculated from token usage using the configured paid API prices.
    estimatedCostUsd: number | null;
    // Legacy compatibility for older benchmark JSON. New runs always set false,
    // and the UI always displays estimatedCostUsd instead of guessing billing.
    freeTierEligible: boolean;
  };
};

export type CapturedView = {
  name: string;
  label: string;
  image: string;
};

export type TriangleIdMap = {
  width: number;
  height: number;
  // Zero means background. A visible triangle is stored as triangle index + 1.
  encodedTriangleIds: Uint32Array;
};

