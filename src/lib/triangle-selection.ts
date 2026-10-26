import type { ImagePoint, TriangleIdMap } from "./scanbody-types";

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Converts one image outline into the exact visible STL triangle IDs beneath it.
 * The browser fills the polygon into a temporary black-and-white image, then we
 * use that image as a mask over the hidden triangle-ID render.
 */
export function findTriangleIdsInsideOutline(
  outline: ImagePoint[],
  triangleIdMap: TriangleIdMap,
) {
  const selectedTriangleIds = new Set<number>();
  if (outline.length < 3) return selectedTriangleIds;

  let minX = triangleIdMap.width - 1;
  let minY = triangleIdMap.height - 1;
  let maxX = 0;
  let maxY = 0;

  for (const [x, y] of outline) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const startX = clamp(Math.floor(minX), 0, triangleIdMap.width - 1);
  const startY = clamp(Math.floor(minY), 0, triangleIdMap.height - 1);
  const endX = clamp(Math.ceil(maxX), 0, triangleIdMap.width - 1);
  const endY = clamp(Math.ceil(maxY), 0, triangleIdMap.height - 1);
  const maskWidth = endX - startX + 1;
  const maskHeight = endY - startY + 1;
  if (maskWidth <= 0 || maskHeight <= 0) return selectedTriangleIds;

  // Canvas rasterization is much faster than testing every pixel against every
  // polygon edge, especially when an outline contains hundreds of points.
  const canvas = document.createElement("canvas");
  canvas.width = maskWidth;
  canvas.height = maskHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return selectedTriangleIds;

  context.fillStyle = "#ffffff";
  context.beginPath();
  context.moveTo(outline[0][0] - startX, outline[0][1] - startY);
  for (let index = 1; index < outline.length; index += 1) {
    context.lineTo(outline[index][0] - startX, outline[index][1] - startY);
  }
  context.closePath();
  context.fill();

  const mask = context.getImageData(0, 0, maskWidth, maskHeight).data;
  for (let localY = 0; localY < maskHeight; localY += 1) {
    for (let localX = 0; localX < maskWidth; localX += 1) {
      const maskOffset = (localY * maskWidth + localX) * 4;
      if (mask[maskOffset + 3] === 0) continue;

      const imageX = startX + localX;
      const imageY = startY + localY;
      const encodedId =
        triangleIdMap.encodedTriangleIds[
          imageY * triangleIdMap.width + imageX
        ];

      // ID zero is the empty background. Real triangle IDs were stored as +1.
      if (encodedId > 0) selectedTriangleIds.add(encodedId - 1);
    }
  }

  return selectedTriangleIds;
}
