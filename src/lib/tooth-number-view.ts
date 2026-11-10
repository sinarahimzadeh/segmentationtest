import type { ViewDetections } from "./scanbody-types";

export type ToothViewChoice = "auto" | "top" | "bottom";

/** PCA does not guarantee which arch-facing camera shows the teeth. Use the
 * existing detections as a heuristic, with an explicit override in the UI. */
export function chooseToothNumberView(
  availableNames: readonly string[],
  detectedViews: readonly ViewDetections[],
  choice: ToothViewChoice,
): "top" | "bottom" | null {
  if (choice !== "auto") return availableNames.includes(choice) ? choice : null;
  const candidates = (["top", "bottom"] as const).filter((name) => availableNames.includes(name));
  const detections = (name: string) => detectedViews.find((view) => view.view === name)?.detections ?? [];
  candidates.sort((first, second) => {
    const a = detections(first);
    const b = detections(second);
    // Prefer more visible posts. Confidence breaks a tie; a full tie keeps Top.
    return b.length - a.length ||
      b.reduce((sum, detection) => sum + detection.confidence, 0) -
      a.reduce((sum, detection) => sum + detection.confidence, 0);
  });
  return candidates[0] ?? null;
}
