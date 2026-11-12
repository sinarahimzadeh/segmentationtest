import { z } from "zod";

export const JAW_HINTS = ["auto", "upper", "lower"] as const;
export const RIGHT_SIDE_HINTS = ["unknown", "image-left", "image-right"] as const;
export type JawHint = (typeof JAW_HINTS)[number];
export type RightSideHint = (typeof RIGHT_SIDE_HINTS)[number];

export type ToothNumberAssignment = {
  detectionId: string;
  toothFdi: number | null;
  confidence: number;
  reason: string;
};

export type ToothNumberResult = {
  jaw: "upper" | "lower" | "unknown";
  assignments: ToothNumberAssignment[];
  notes: string;
  meta: {
    modelLabel: string;
    imageCount?: number;
    imageSize?: number;
    jpegQuality?: number;
    viewNames?: string[];
    elapsedMs: number;
    inputTokens: number | null;
    outputTokens: number | null;
    estimatedCostUsd: number | null;
  };
  savedPath: string | null;
  saveError?: string;
};

// Kept completely separate from the scanbody segmentation response schema.
export const toothNumberResponseSchema = z.object({
  jaw: z.enum(["upper", "lower", "unknown"]),
  assignments: z.array(z.object({
    detectionId: z.string().describe("An exact detectionId supplied with a scanbody outline."),
    toothFdi: z.number().int().nullable().describe("Permanent-tooth FDI number, or null when uncertain."),
    confidence: z.number().describe("Confidence from 0 to 1 in this tooth-position estimate."),
    reason: z.string().describe("Brief explanation, particularly when the number is uncertain."),
  })),
  notes: z.string().describe("Brief explanation of anatomical or orientation uncertainty."),
});

export function isPermanentFdi(value: number) {
  return Number.isInteger(value) && /^[1-4][1-8]$/.test(String(value));
}

/** Preserve input IDs, including any the model could not number. Never guess. */
export function normalizeToothAssignments(
  result: z.infer<typeof toothNumberResponseSchema>,
  targets: { detectionId: string; view: string }[],
  jawHint: JawHint,
) {
  const knownIds = new Set(targets.map((target) => target.detectionId));
  const byId = new Map<string, ToothNumberAssignment>();
  const jaw = jawHint === "auto" ? result.jaw : jawHint;
  for (const assignment of result.assignments) {
    if (!knownIds.has(assignment.detectionId)) {
      throw new Error(`Tooth numbering returned an unknown detection ID: ${assignment.detectionId}`);
    }
    if (byId.has(assignment.detectionId)) {
      throw new Error(`Tooth numbering repeated detection ID: ${assignment.detectionId}`);
    }
    const number = assignment.toothFdi;
    const valid = number === null || (isPermanentFdi(number) &&
      (jaw !== "upper" || number < 30) && (jaw !== "lower" || number >= 30));
    byId.set(assignment.detectionId, valid ? {
      ...assignment,
      confidence: Math.max(0, Math.min(1, assignment.confidence)),
    } : {
      ...assignment, toothFdi: null, confidence: 0,
      reason: "The returned tooth number was invalid or did not match the jaw.",
    });
  }
  const assignments = targets.map((target) => byId.get(target.detectionId) ?? {
    detectionId: target.detectionId, toothFdi: null, confidence: 0,
    reason: "The AI did not identify a tooth position for this outline.",
  });
  // Repeated numbers across views are expected. Two separate posts in the same
  // image cannot occupy one tooth position; show uncertainty rather than merge.
  const seen = new Map<string, ToothNumberAssignment[]>();
  for (let index = 0; index < targets.length; index++) {
    const assignment = assignments[index];
    if (assignment.toothFdi === null) continue;
    const key = `${targets[index].view}:${assignment.toothFdi}`;
    seen.set(key, [...(seen.get(key) ?? []), assignment]);
  }
  for (const duplicates of seen.values()) {
    if (duplicates.length < 2) continue;
    for (const assignment of duplicates) {
      assignment.toothFdi = null;
      assignment.confidence = 0;
      assignment.reason = "The AI assigned the same tooth position to two scanbodies in one view.";
    }
  }
  return { jaw, assignments, notes: result.notes };
}

export const TOOTH_NUMBER_SYSTEM_PROMPT = `Estimate the permanent-tooth FDI positions of ALREADY DETECTED implant scanbodies in dental STL screenshots.
This is a separate tooth-numbering task. Do not detect new scanbodies, change outlines, or return segmentation geometry.
You receive one selected arch-facing image of the jaw, rendered by either the Top or Bottom camera. Supplied pixel outlines identify the targets in that image. Return one assignment for every supplied detectionId. Different scanbodies must not share a tooth number. Do not assume that other camera views are available.
Use the FDI two-digit system, not the Universal 1-32 system. Quadrants are from the PATIENT's perspective: 1 upper right, 2 upper left, 3 lower left, 4 lower right. In each quadrant, positions 1 through 8 run from the central incisor to the third molar. The scanbody occupies a missing-tooth position, not necessarily a visible natural tooth.
Examine the entire dental arch, remaining teeth, midline, canine positions, molar shapes, palate, and gaps. Account for missing teeth; do not number only the visible teeth consecutively. First establish the jaw, arch order, and tooth positions, then assign scanbodies.
The mesh was rotated by PCA. Top and Bottom describe rendering directions, NOT the upper/lower jaw or the patient's anatomical sides. Use supplied jaw and patient-right hints when provided. The patient-right hint refers only to the selected image.
An unmarked, rotated or mirrored scan may not establish patient left/right or exact positions. Make a best-supported estimate when anatomical evidence is sufficient; otherwise return null and explain the missing evidence briefly. Do not invent certainty. Return only the requested JSON.`;
