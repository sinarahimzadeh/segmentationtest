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
    reasoningEffort: ReasoningEffort;
    elapsedMs: number;
    inputTokens: number | null;
    cachedInputTokens: number | null;
    outputTokens: number | null;
    // For Gemini free-tier calls the actual charge can be $0. This value is
    // the current paid Standard equivalent so models remain comparable.
    estimatedCostUsd: number | null;
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

