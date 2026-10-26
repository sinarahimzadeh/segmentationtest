export type StlExportMode = "selected" | "remaining";

function shouldIncludeTriangle(
  triangleIndex: number,
  selectedTriangleIds: ReadonlySet<number>,
  mode: StlExportMode,
) {
  const isSelected = selectedTriangleIds.has(triangleIndex);
  return mode === "selected" ? isSelected : !isSelected;
}

/**
 * Writes a binary STL using the untouched coordinates from the uploaded file.
 * Triangle IDs still match after PCA because PCA moves vertices without changing
 * their order, so the exported mesh returns to the scanner's coordinate system.
 */
export function createBinaryStlBlob(
  originalPositions: Float32Array,
  selectedTriangleIds: ReadonlySet<number>,
  mode: StlExportMode,
) {
  const sourceTriangleCount = Math.floor(originalPositions.length / 9);
  let exportedTriangleCount = 0;

  for (
    let triangleIndex = 0;
    triangleIndex < sourceTriangleCount;
    triangleIndex += 1
  ) {
    if (shouldIncludeTriangle(triangleIndex, selectedTriangleIds, mode)) {
      exportedTriangleCount += 1;
    }
  }

  // Binary STL layout: 80-byte header, 4-byte triangle count, then 50 bytes
  // per triangle (normal, three vertices, and a 2-byte attribute field).
  const buffer = new ArrayBuffer(84 + exportedTriangleCount * 50);
  const view = new DataView(buffer);
  const header = new TextEncoder().encode(
    "Scanbody ROI export - original STL coordinates",
  );
  new Uint8Array(buffer, 0, Math.min(header.length, 80)).set(
    header.subarray(0, 80),
  );
  view.setUint32(80, exportedTriangleCount, true);

  let byteOffset = 84;
  for (
    let triangleIndex = 0;
    triangleIndex < sourceTriangleCount;
    triangleIndex += 1
  ) {
    if (!shouldIncludeTriangle(triangleIndex, selectedTriangleIds, mode)) {
      continue;
    }

    const positionOffset = triangleIndex * 9;
    const ax = originalPositions[positionOffset];
    const ay = originalPositions[positionOffset + 1];
    const az = originalPositions[positionOffset + 2];
    const bx = originalPositions[positionOffset + 3];
    const by = originalPositions[positionOffset + 4];
    const bz = originalPositions[positionOffset + 5];
    const cx = originalPositions[positionOffset + 6];
    const cy = originalPositions[positionOffset + 7];
    const cz = originalPositions[positionOffset + 8];

    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const acx = cx - ax;
    const acy = cy - ay;
    const acz = cz - az;
    let normalX = aby * acz - abz * acy;
    let normalY = abz * acx - abx * acz;
    let normalZ = abx * acy - aby * acx;
    const normalLength = Math.hypot(normalX, normalY, normalZ);
    if (normalLength > 0) {
      normalX /= normalLength;
      normalY /= normalLength;
      normalZ /= normalLength;
    }

    for (const value of [
      normalX,
      normalY,
      normalZ,
      ax,
      ay,
      az,
      bx,
      by,
      bz,
      cx,
      cy,
      cz,
    ]) {
      view.setFloat32(byteOffset, value, true);
      byteOffset += 4;
    }
    view.setUint16(byteOffset, 0, true);
    byteOffset += 2;
  }

  return new Blob([buffer], { type: "model/stl" });
}

/** Starts a normal browser download and immediately releases its temporary URL. */
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
