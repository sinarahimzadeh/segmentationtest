import { z } from "zod";
import type { BoundingBox, ImagePoint } from "./scanbody-types";

/** A separate schema: the model writes just four coordinates, not a contour.
 * A homogeneous number array works with OpenAI, Google and Claude schemas.
 */
export const boxDetectionSchema = z.object({
  views: z.array(z.object({
    view: z.string().describe("The exact VIEW_ID associated with this image."),
    detections: z.array(z.object({
      label: z.enum(["scanbody"]),
      box: z.array(z.number()).length(4).describe(
        "Axis-aligned bounding box as [left, top, right, bottom] in whole-pixel coordinates. Exactly four numbers, not corner points or a contour.",
      ),
      confidence: z.number().describe("Confidence from 0 to 1 that this is a scanbody."),
    })),
  })),
});

export const BOX_SYSTEM_PROMPT = `You inspect monochrome renders of dental intraoral STL meshes.
Your only task is to locate implant scanbodies with simple bounding boxes.

A scanbody is a manufactured post attached to a dental implant. It usually
protrudes from gingiva and has smooth cylindrical, flat-sided, rectangular,
indexed, or cap-like manufactured geometry. It can also look like a large
circular collar or ring with a central opening, flat index, shoulder, or
stepped sides. Several repeated manufactured posts distributed around the
dental arch are strong evidence of scanbodies. Their shading may be very dark
or very bright depending on the camera angle.

Do not mark teeth, gingiva, crowns, empty implant holes, or random mesh
artifacts. Do not reject a manufactured registration post merely because it
resembles a healing abutment. Reject a simple healing cap without scan indexing,
but include repeated protruding indexed bodies used for implant scanning.
A central screw channel or hole inside a manufactured post is part of that
scanbody, not an empty implant hole.

Treat every VIEW_ID independently. Some views show the underside and have no
visible scanbodies. Check the whole image for repeated manufactured posts.
Return a separate tight axis-aligned box around the entire visible portion of
each scanbody, including partially visible ones and any central opening.
Use exactly four whole-pixel coordinates: [left, top, right, bottom], measured
from the image's top-left corner. Left must be less than right; top less than
bottom. Return boxes only. Do not trace outlines, generate polygons, or return
3D points. Use the exact VIEW_ID supplied before each image. An empty detections
array is valid when no scanbody is visible. Include likely scanbodies with
lower confidence when uncertain, but never invent one when no manufactured
post is visible.`;

/** Only normalize a rectangle for drawing; this performs no segmentation. */
export function normalizeBoundingBox(values: number[], imageSize: number): BoundingBox | null {
  if (values.length !== 4 || !values.every(Number.isFinite)) return null;
  const clamp = (value: number) => Math.max(0, Math.min(imageSize, Math.round(value)));
  const left = clamp(Math.min(values[0], values[2]));
  const right = clamp(Math.max(values[0], values[2]));
  const top = clamp(Math.min(values[1], values[3]));
  const bottom = clamp(Math.max(values[1], values[3]));
  return left < right && top < bottom ? [left, top, right, bottom] : null;
}

/** Rectangle corners are only for the existing 2D SVG preview. Box-mode
 * results must never be passed to the 3D triangle mapper as a segmentation.
 */
export function boxPreviewCorners([left, top, right, bottom]: BoundingBox): ImagePoint[] {
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}
