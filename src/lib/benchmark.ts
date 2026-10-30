import type { AnalysisResult } from "./scanbody-types";
import type { AiModelId, ReasoningEffort } from "./ai-models";

export const BENCHMARK_STORAGE_VERSION = 1;

export type ComparisonMetrics = {
  sameTrianglePercent: number;
  baselineFoundPercent: number;
  missedBaselinePercent: number;
  extraCandidatePercent: number;
  sharedTriangleCount: number;
  missedTriangleCount: number;
  extraTriangleCount: number;
  objectCountDifference: number;
};

export type SavedBenchmarkRun = {
  id: string;
  storageVersion: number;
  meshKey: string;
  meshName: string;
  // Human-readable label taken from the STL's immediate parent folder.
  meshFolderName?: string;
  kind: "baseline" | "comparison";
  model: AiModelId;
  reasoningEffort: ReasoningEffort;
  createdAt: string;
  analysis: AnalysisResult;
  selectedTriangleIds: number[];
  objectCount: number;
  metrics: ComparisonMetrics | null;
};

export function compareTriangleSelections(
  baselineTriangleIds: readonly number[],
  candidateTriangleIds: ReadonlySet<number>,
  baselineObjectCount: number,
  candidateObjectCount: number,
): ComparisonMetrics {
  const baseline = new Set(baselineTriangleIds);
  let sharedTriangleCount = 0;
  let extraTriangleCount = 0;

  for (const triangleId of candidateTriangleIds) {
    if (baseline.has(triangleId)) sharedTriangleCount += 1;
    else extraTriangleCount += 1;
  }

  const missedTriangleCount = baseline.size - sharedTriangleCount;
  const unionCount =
    sharedTriangleCount + missedTriangleCount + extraTriangleCount;
  const percent = (numerator: number, denominator: number) =>
    denominator === 0 ? (numerator === 0 ? 100 : 0) : (numerator / denominator) * 100;

  return {
    // Intersection over union: one strict number for overall similarity.
    sameTrianglePercent: percent(sharedTriangleCount, unionCount),
    baselineFoundPercent: percent(sharedTriangleCount, baseline.size),
    missedBaselinePercent: percent(missedTriangleCount, baseline.size),
    extraCandidatePercent: percent(extraTriangleCount, candidateTriangleIds.size),
    sharedTriangleCount,
    missedTriangleCount,
    extraTriangleCount,
    objectCountDifference: candidateObjectCount - baselineObjectCount,
  };
}
