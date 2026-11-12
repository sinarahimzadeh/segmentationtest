import type { CapturedView, ImagePoint } from "./scanbody-types";

export const IMAGE_DIMENSIONS = [1024, 896, 768, 640, 512, 384] as const;
export const JPEG_QUALITIES = [90, 80, 70, 60, 50, 40, 30] as const;
export type ImageDimension = (typeof IMAGE_DIMENSIONS)[number];
export type JpegQuality = (typeof JPEG_QUALITIES)[number];

/** Prepare the actual JPEGs sent to every provider and shown in the previews.
 * Always encode from the original render, never from a previously compressed
 * JPEG. Switching settings back and forth therefore cannot accumulate damage.
 * The original 1024/90 images are returned unchanged for the default settings.
 */
export function prepareImageViews(
  views: CapturedView[],
  sources: ReadonlyMap<string, HTMLCanvasElement>,
  dimension: ImageDimension,
  quality: JpegQuality,
): CapturedView[] {
  if (views.length === 0 || (dimension === 1024 && quality === 90)) return views;
  const canvas = document.createElement("canvas");
  canvas.width = dimension;
  canvas.height = dimension;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not prepare camera images.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  return views.map((view) => {
    const source = sources.get(view.name);
    if (!source) throw new Error(`Missing original camera image: ${view.name}`);
    context.clearRect(0, 0, dimension, dimension);
    context.drawImage(source, 0, 0, dimension, dimension);
    return { ...view, image: canvas.toDataURL("image/jpeg", quality / 100) };
  });
}

/** Keep outlines aligned when the API image, overlay, and hidden triangle-ID
 * map have different dimensions. Old saved runs used 1024-pixel coordinates.
 */
export function scaleOutlinePoints(
  outline: ImagePoint[], fromSize: number, toSize: number,
): ImagePoint[] {
  if (fromSize === toSize) return outline;
  const scale = toSize / fromSize;
  return outline.map(([x, y]) => [x * scale, y * scale]);
}
