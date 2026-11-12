/** One provider response, before compact outlines are converted to 3D triangles. */
export type ViewBatchResult<Detection> = {
  parsed: { views: { view: string; detections: Detection[] }[] };
  inputTokens: number | null;
  cachedInputTokens: number | null;
  outputTokens: number | null;
  serviceTier: string | null;
  reportedCostUsd: number | null;
  estimatedCostUsd: number | null;
};

function sumKnown(values: (number | null)[]) {
  // Missing usage is unknown, not zero. Avoid displaying a misleading total.
  let total = 0;
  for (const value of values) {
    if (value === null) return null;
    total += value;
  }
  return total;
}

/**
 * Parallel mode sends every view once, at the same time. Grouped mode keeps the
 * original single request containing all views. Provider clients and billing
 * receipts must be created inside analyzeBatch so concurrent calls stay separate.
 */
export async function analyzeViewBatches<View extends { name: string }, Detection>(
  views: View[],
  parallel: boolean,
  analyzeBatch: (views: View[]) => Promise<ViewBatchResult<Detection>>,
) {
  if (views.length === 0 || new Set(views.map((view) => view.name)).size !== views.length) {
    throw new Error("Camera views must have unique names and cannot be empty.");
  }
  const batches = parallel ? views.map((view) => [view]) : [views];
  // Wait for every in-flight call even if one fails. Do not silently save a
  // partial segmentation or automatically send another paid all-images request.
  const settled = await Promise.allSettled(batches.map(async (batch) => {
    // Independent clocks: each measures its own request, including client setup,
    // provider/network waiting, retries if configured, and response parsing.
    const requestStartedAt = performance.now();
    const result = await analyzeBatch(batch);
    const requestElapsedMs = performance.now() - requestStartedAt;
    if (parallel && (
      result.parsed.views.length !== 1 ||
      result.parsed.views[0].view !== batch[0].name
    )) {
      throw new Error(`The AI did not return exactly the requested VIEW_ID (${batch[0].name}).`);
    }
    return { ...result, requestElapsedMs };
  }));
  const failures = settled.flatMap((result, index) => result.status === "rejected"
    ? [`${batches[index].map((view) => view.name).join(", ")}: ${
        result.reason instanceof Error ? result.reason.message : String(result.reason)
      }`]
    : []);
  if (failures.length > 0) {
    if (!parallel) {
      const failure = settled[0] as PromiseRejectedResult;
      throw failure.reason;
    }
    throw new Error(
      `${failures.length}/${batches.length} view requests failed. No partial result was saved. ` +
      `Completed requests may still be charged. ${failures.join("; ")}`,
    );
  }
  // allSettled preserves input order even when the provider finishes out of order.
  const results = settled.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
  const tiers = new Set(results.map((result) => result.serviceTier));
  return {
    parsed: { views: results.flatMap((result) => result.parsed.views) },
    inputTokens: sumKnown(results.map((result) => result.inputTokens)),
    cachedInputTokens: sumKnown(results.map((result) => result.cachedInputTokens)),
    outputTokens: sumKnown(results.map((result) => result.outputTokens)),
    // Each batch already uses its reported charge, or its own price estimate.
    estimatedCostUsd: sumKnown(results.map((result) => result.estimatedCostUsd)),
    reportedCostUsd: sumKnown(results.map((result) => result.reportedCostUsd)),
    serviceTier: tiers.size === 1 ? results[0].serviceTier : "mixed",
    viewRequestCount: batches.length,
    // A grouped request cannot reveal how long the model spent on each image.
    // Never divide its elapsed time by the number of images or invent timings.
    viewTimings: batches.every((batch) => batch.length === 1)
      ? results.map((result, index) => ({ view: batches[index][0].name, elapsedMs: result.requestElapsedMs }))
      : null,
  };
}
