import type {
  ImagePoint,
  TriangleIdMap,
  ViewDetections,
} from "./scanbody-types";
import type { BufferAttribute } from "three";

const MIN_AGREEING_VIEWS = 2;
const MIN_SUPPORT_RATIO = 0.5;
const MIN_MEAN_CONFIDENCE = 0.6;
const SINGLE_VISIBLE_VIEW_CONFIDENCE = 0.92;

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

type ViewEvidence = {
  detections: ViewDetections;
  triangleIdMap: TriangleIdMap;
};

type Point3 = [number, number, number];

type PreparedDetection = {
  view: string;
  confidence: number;
  triangleIds: Set<number>;
  center: Point3;
  radius: number;
};

type DetectionGroup = {
  detections: PreparedDetection[];
  views: Set<string>;
  center: Point3;
  totalWeight: number;
};

function distance3(a: Point3, b: Point3) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function median(values: number[]) {
  if (values.length === 0) return 0;
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0
    ? (values[middle - 1] + values[middle]) / 2
    : values[middle];
}

function getTriangleCenter(
  triangleId: number,
  positions: BufferAttribute,
): Point3 {
  const firstVertex = triangleId * 3;
  return [
    (positions.getX(firstVertex) +
      positions.getX(firstVertex + 1) +
      positions.getX(firstVertex + 2)) /
      3,
    (positions.getY(firstVertex) +
      positions.getY(firstVertex + 1) +
      positions.getY(firstVertex + 2)) /
      3,
    (positions.getZ(firstVertex) +
      positions.getZ(firstVertex + 1) +
      positions.getZ(firstVertex + 2)) /
      3,
  ];
}

/** Uses medians so a few jaw triangles inside a rough outline cannot drag the
 * estimated scanbody center far away from the manufactured body. */
function estimateDetectionCenter(
  triangleIds: ReadonlySet<number>,
  positions: BufferAttribute,
) {
  const maximumSamples = 800;
  const sampleStep = Math.max(
    1,
    Math.floor(triangleIds.size / maximumSamples),
  );
  const centers: Point3[] = [];
  let itemIndex = 0;

  for (const triangleId of triangleIds) {
    if (itemIndex % sampleStep === 0) {
      centers.push(getTriangleCenter(triangleId, positions));
    }
    itemIndex += 1;
  }
  if (centers.length === 0) return null;

  const center: Point3 = [
    median(centers.map((point) => point[0])),
    median(centers.map((point) => point[1])),
    median(centers.map((point) => point[2])),
  ];
  const distances = centers
    .map((point) => distance3(point, center))
    .sort((a, b) => a - b);
  const radiusIndex = Math.min(
    distances.length - 1,
    Math.floor(distances.length * 0.8),
  );

  return { center, radius: distances[radiusIndex] ?? 0 };
}

function getModelDiagonal(positions: BufferAttribute) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (let vertex = 0; vertex < positions.count; vertex += 1) {
    const x = positions.getX(vertex);
    const y = positions.getY(vertex);
    const z = positions.getZ(vertex);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  return Math.max(Math.hypot(maxX - minX, maxY - minY, maxZ - minZ), 1);
}

/** Matches 2D detections from separate cameras by their estimated 3D center. */
function groupDetectionsByPosition(
  detections: PreparedDetection[],
  modelDiagonal: number,
) {
  const typicalRadius = median(detections.map((detection) => detection.radius));
  const groupingDistance = clamp(
    typicalRadius * 1.75,
    modelDiagonal * 0.04,
    modelDiagonal * 0.09,
  );
  const groups: DetectionGroup[] = [];

  // Strong detections establish the candidate centers first.
  const sortedDetections = [...detections].sort(
    (a, b) => b.confidence - a.confidence,
  );
  for (const detection of sortedDetections) {
    let bestGroup: DetectionGroup | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const group of groups) {
      // Two detections from one image must remain two different objects.
      if (group.views.has(detection.view)) continue;
      const distance = distance3(group.center, detection.center);
      if (distance <= groupingDistance && distance < bestDistance) {
        bestGroup = group;
        bestDistance = distance;
      }
    }

    if (!bestGroup) {
      groups.push({
        detections: [detection],
        views: new Set([detection.view]),
        center: [...detection.center],
        totalWeight: Math.max(detection.confidence, 0.01),
      });
      continue;
    }

    const weight = Math.max(detection.confidence, 0.01);
    const newTotalWeight = bestGroup.totalWeight + weight;
    bestGroup.center = [
      (bestGroup.center[0] * bestGroup.totalWeight +
        detection.center[0] * weight) /
        newTotalWeight,
      (bestGroup.center[1] * bestGroup.totalWeight +
        detection.center[1] * weight) /
        newTotalWeight,
      (bestGroup.center[2] * bestGroup.totalWeight +
        detection.center[2] * weight) /
        newTotalWeight,
    ];
    bestGroup.totalWeight = newTotalWeight;
    bestGroup.detections.push(detection);
    bestGroup.views.add(detection.view);
  }

  // A one-camera object is much more likely to be random image noise. Real
  // scanbody candidates must be independently located in at least two views.
  return groups.filter((group) => group.views.size >= MIN_AGREEING_VIEWS);
}

function createVertexKey(
  positions: BufferAttribute,
  vertex: number,
  tolerance: number,
) {
  return `${Math.round(positions.getX(vertex) / tolerance)},${Math.round(
    positions.getY(vertex) / tolerance,
  )},${Math.round(positions.getZ(vertex) / tolerance)}`;
}

function getTriangleNormal(
  triangleId: number,
  positions: BufferAttribute,
): Point3 {
  const firstVertex = triangleId * 3;
  const ax = positions.getX(firstVertex);
  const ay = positions.getY(firstVertex);
  const az = positions.getZ(firstVertex);
  const abx = positions.getX(firstVertex + 1) - ax;
  const aby = positions.getY(firstVertex + 1) - ay;
  const abz = positions.getZ(firstVertex + 1) - az;
  const acx = positions.getX(firstVertex + 2) - ax;
  const acy = positions.getY(firstVertex + 2) - ay;
  const acz = positions.getZ(firstVertex + 2) - az;
  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  const length = Math.hypot(nx, ny, nz) || 1;
  return [nx / length, ny / length, nz / length];
}

function getTriangleAverageEdgeLength(
  triangleId: number,
  positions: BufferAttribute,
) {
  const firstVertex = triangleId * 3;
  const points: Point3[] = [0, 1, 2].map((corner) => [
    positions.getX(firstVertex + corner),
    positions.getY(firstVertex + corner),
    positions.getZ(firstVertex + corner),
  ]);
  return (
    distance3(points[0], points[1]) +
    distance3(points[1], points[2]) +
    distance3(points[2], points[0])
  ) / 3;
}

function createEdgeKey(firstVertexKey: string, secondVertexKey: string) {
  return firstVertexKey < secondVertexKey
    ? `${firstVertexKey}|${secondVertexKey}`
    : `${secondVertexKey}|${firstVertexKey}`;
}

/**
 * Expands a clean, conservative scanbody core over nearby connected triangles.
 * Growth is deliberately bounded by physical distance and mesh-edge steps.
 * Smooth unmarked neighbors may fill outline holes; sharper neighbors must at
 * least have been included by one of the original image detections.
 */
function growSelectionAcrossSurface(
  coreTriangleIds: ReadonlySet<number>,
  hintedTriangleIds: ReadonlySet<number>,
  positions: BufferAttribute,
  groupCenter: Point3,
  estimatedGroupRadius: number,
  modelDiagonal: number,
) {
  if (coreTriangleIds.size === 0) return new Set<number>();

  const triangleCount = Math.floor(positions.count / 3);
  const spatialRadius = clamp(
    estimatedGroupRadius * 1.35,
    modelDiagonal * 0.025,
    modelDiagonal * 0.075,
  );
  const tolerance = Math.max(modelDiagonal * 0.00001, 1e-7);
  const candidateTriangleIds: number[] = [];

  // Restrict adjacency construction to a small region around this scanbody.
  // This prevents a smooth bridge in the full jaw from carrying growth away.
  for (let triangleId = 0; triangleId < triangleCount; triangleId += 1) {
    if (
      distance3(getTriangleCenter(triangleId, positions), groupCenter) <=
      spatialRadius
    ) {
      candidateTriangleIds.push(triangleId);
    }
  }

  const neighbors = new Map<number, number[]>();
  const firstTriangleAtEdge = new Map<string, number>();
  for (const triangleId of candidateTriangleIds) {
    const firstVertex = triangleId * 3;
    const vertexKeys = [0, 1, 2].map((corner) =>
      createVertexKey(positions, firstVertex + corner, tolerance),
    );
    const edgeKeys = [
      createEdgeKey(vertexKeys[0], vertexKeys[1]),
      createEdgeKey(vertexKeys[1], vertexKeys[2]),
      createEdgeKey(vertexKeys[2], vertexKeys[0]),
    ];

    for (const edgeKey of edgeKeys) {
      const existingTriangle = firstTriangleAtEdge.get(edgeKey);
      if (existingTriangle === undefined) {
        firstTriangleAtEdge.set(edgeKey, triangleId);
        continue;
      }
      const existingNeighbors = neighbors.get(existingTriangle);
      if (existingNeighbors) existingNeighbors.push(triangleId);
      else neighbors.set(existingTriangle, [triangleId]);
      const currentNeighbors = neighbors.get(triangleId);
      if (currentNeighbors) currentNeighbors.push(existingTriangle);
      else neighbors.set(triangleId, [existingTriangle]);
    }
  }

  const sampledEdgeLengths: number[] = [];
  const edgeSampleStep = Math.max(
    1,
    Math.floor(coreTriangleIds.size / 500),
  );
  let coreIndex = 0;
  for (const triangleId of coreTriangleIds) {
    if (coreIndex % edgeSampleStep === 0) {
      sampledEdgeLengths.push(
        getTriangleAverageEdgeLength(triangleId, positions),
      );
    }
    coreIndex += 1;
  }
  const typicalEdgeLength = Math.max(median(sampledEdgeLengths), 1e-7);
  const desiredGrowthDistance = Math.min(
    spatialRadius * 0.28,
    modelDiagonal * 0.012,
  );
  const maximumGrowthSteps = Math.round(
    clamp(desiredGrowthDistance / typicalEdgeLength, 4, 24),
  );
  const smoothNormalSimilarity = Math.cos((55 * Math.PI) / 180);
  const hintedNormalSimilarity = Math.cos((85 * Math.PI) / 180);
  const grownTriangleIds = new Set(coreTriangleIds);
  const queue: Array<{ triangleId: number; depth: number }> = [];
  for (const triangleId of coreTriangleIds) {
    queue.push({ triangleId, depth: 0 });
  }

  let queueIndex = 0;
  while (queueIndex < queue.length) {
    const { triangleId, depth } = queue[queueIndex];
    queueIndex += 1;
    if (depth >= maximumGrowthSteps) continue;

    const sourceNormal = getTriangleNormal(triangleId, positions);
    for (const neighborId of neighbors.get(triangleId) ?? []) {
      if (grownTriangleIds.has(neighborId)) continue;
      const neighborNormal = getTriangleNormal(neighborId, positions);
      const normalSimilarity = Math.abs(
        sourceNormal[0] * neighborNormal[0] +
          sourceNormal[1] * neighborNormal[1] +
          sourceNormal[2] * neighborNormal[2],
      );
      const minimumSimilarity = hintedTriangleIds.has(neighborId)
        ? hintedNormalSimilarity
        : smoothNormalSimilarity;
      if (normalSimilarity < minimumSimilarity) continue;

      grownTriangleIds.add(neighborId);
      queue.push({ triangleId: neighborId, depth: depth + 1 });
    }
  }

  return grownTriangleIds;
}

/** Removes tiny disconnected islands while retaining substantial separated
 * pieces when a valid scanbody was split by imperfect 2D outlines. */
function removeSmallDisconnectedFragments(
  triangleIds: ReadonlySet<number>,
  positions: BufferAttribute,
  modelDiagonal: number,
  groupCenter: Point3,
  estimatedGroupRadius: number,
) {
  const ids = [...triangleIds];
  if (ids.length < 2) return new Set(ids);

  const parents = new Int32Array(ids.length);
  const componentSizes = new Int32Array(ids.length);
  for (let index = 0; index < ids.length; index += 1) {
    parents[index] = index;
    componentSizes[index] = 1;
  }

  const findRoot = (start: number) => {
    let root = start;
    while (parents[root] !== root) root = parents[root];
    let current = start;
    while (parents[current] !== current) {
      const next = parents[current];
      parents[current] = root;
      current = next;
    }
    return root;
  };

  const unite = (first: number, second: number) => {
    let firstRoot = findRoot(first);
    let secondRoot = findRoot(second);
    if (firstRoot === secondRoot) return;
    if (componentSizes[firstRoot] < componentSizes[secondRoot]) {
      [firstRoot, secondRoot] = [secondRoot, firstRoot];
    }
    parents[secondRoot] = firstRoot;
    componentSizes[firstRoot] += componentSizes[secondRoot];
  };

  const tolerance = Math.max(modelDiagonal * 0.00001, 1e-7);
  const firstTriangleAtVertex = new Map<string, number>();
  for (let localIndex = 0; localIndex < ids.length; localIndex += 1) {
    const firstVertex = ids[localIndex] * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const key = createVertexKey(
        positions,
        firstVertex + corner,
        tolerance,
      );
      const existing = firstTriangleAtVertex.get(key);
      if (existing === undefined) {
        firstTriangleAtVertex.set(key, localIndex);
      } else {
        unite(localIndex, existing);
      }
    }
  }

  const components = new Map<number, number[]>();
  for (let localIndex = 0; localIndex < ids.length; localIndex += 1) {
    const root = findRoot(localIndex);
    const component = components.get(root);
    if (component) component.push(ids[localIndex]);
    else components.set(root, [ids[localIndex]]);
  }

  const orderedComponents = [...components.values()].sort(
    (a, b) => b.length - a.length,
  );
  const largestSize = orderedComponents[0]?.length ?? 0;
  const minimumUsefulSize = Math.max(8, Math.ceil(largestSize * 0.03));
  const maximumUsefulDistance = Math.max(
    estimatedGroupRadius * 1.5,
    modelDiagonal * 0.03,
  );
  const cleaned = new Set<number>();
  for (
    let componentIndex = 0;
    componentIndex < orderedComponents.length;
    componentIndex += 1
  ) {
    const component = orderedComponents[componentIndex];
    let closestDistance = Number.POSITIVE_INFINITY;
    for (const triangleId of component) {
      closestDistance = Math.min(
        closestDistance,
        distance3(getTriangleCenter(triangleId, positions), groupCenter),
      );
    }
    const isLargestComponent = componentIndex === 0;
    if (
      !isLargestComponent &&
      (component.length < minimumUsefulSize ||
        closestDistance > maximumUsefulDistance)
    ) {
      continue;
    }
    for (const triangleId of component) cleaned.add(triangleId);
  }

  return cleaned;
}

/**
 * Combines all camera results without allowing one bad view to win by itself.
 *
 * For every triangle we count:
 * - how many camera views can actually see it;
 * - how many of those views place it inside a scanbody outline; and
 * - the confidence of those positive detections.
 *
 * A triangle normally needs support from at least two views and at least half
 * of the views in which it is visible. A triangle visible in only one camera
 * can survive only when that one detection has very high confidence; otherwise
 * real edge surfaces would be deleted simply because no second camera saw them.
 */
export function findGroupedTrianglesWithMultiViewAgreement(
  triangleCount: number,
  positions: BufferAttribute,
  viewEvidence: ViewEvidence[],
) {
  const visibleViewCounts = new Uint8Array(triangleCount);
  const lastVisibleView = new Int16Array(triangleCount);
  lastVisibleView.fill(-1);
  const preparedDetections: PreparedDetection[] = [];

  for (let viewIndex = 0; viewIndex < viewEvidence.length; viewIndex += 1) {
    const { detections, triangleIdMap } = viewEvidence[viewIndex];

    // Count a triangle at most once per view even though it covers many pixels.
    for (const encodedId of triangleIdMap.encodedTriangleIds) {
      if (encodedId === 0) continue;
      const triangleId = encodedId - 1;
      if (
        triangleId >= triangleCount ||
        lastVisibleView[triangleId] === viewIndex
      ) {
        continue;
      }
      lastVisibleView[triangleId] = viewIndex;
      visibleViewCounts[triangleId] += 1;
    }

    for (const detection of detections.detections) {
      const idsUnderOutline = findTriangleIdsInsideOutline(
        detection.outline,
        triangleIdMap,
      );
      const estimate = estimateDetectionCenter(idsUnderOutline, positions);
      if (!estimate) continue;
      preparedDetections.push({
        view: detections.view,
        confidence: detection.confidence,
        triangleIds: idsUnderOutline,
        center: estimate.center,
        radius: estimate.radius,
      });
    }
  }

  const modelDiagonal = getModelDiagonal(positions);
  const groups = groupDetectionsByPosition(
    preparedDetections,
    modelDiagonal,
  );
  const selectedTriangleIds = new Set<number>();
  for (const group of groups) {
    const positiveViewCounts = new Map<number, number>();
    const confidenceSums = new Map<number, number>();
    for (const detection of group.detections) {
      for (const triangleId of detection.triangleIds) {
        if (triangleId >= triangleCount) continue;
        positiveViewCounts.set(
          triangleId,
          (positiveViewCounts.get(triangleId) ?? 0) + 1,
        );
        confidenceSums.set(
          triangleId,
          (confidenceSums.get(triangleId) ?? 0) + detection.confidence,
        );
      }
    }

    const groupTriangleIds = new Set<number>();
    for (const [triangleId, positiveViews] of positiveViewCounts) {
      const visibleViews = Math.max(
        visibleViewCounts[triangleId],
        positiveViews,
      );
      const supportRatio = positiveViews / visibleViews;
      const meanConfidence =
        (confidenceSums.get(triangleId) ?? 0) / positiveViews;
      const hasMultiViewAgreement =
        positiveViews >= MIN_AGREEING_VIEWS &&
        supportRatio >= MIN_SUPPORT_RATIO &&
        meanConfidence >= MIN_MEAN_CONFIDENCE;
      const isHighConfidenceSingleViewSurface =
        visibleViews === 1 &&
        positiveViews === 1 &&
        meanConfidence >= SINGLE_VISIBLE_VIEW_CONFIDENCE;

      if (hasMultiViewAgreement || isHighConfidenceSingleViewSurface) {
        groupTriangleIds.add(triangleId);
      }
    }

    const estimatedGroupRadius = Math.max(
      median(group.detections.map((detection) => detection.radius)),
      modelDiagonal * 0.015,
    );
    const hintedTriangleIds = new Set<number>();
    for (const detection of group.detections) {
      for (const triangleId of detection.triangleIds) {
        hintedTriangleIds.add(triangleId);
      }
    }

    // First remove isolated false-positive seeds. Then expand the reliable core
    // across the neighboring scanbody surface and clean once more afterward.
    const cleanedCore = removeSmallDisconnectedFragments(
      groupTriangleIds,
      positions,
      modelDiagonal,
      group.center,
      estimatedGroupRadius,
    );
    const grownGroup = growSelectionAcrossSurface(
      cleanedCore,
      hintedTriangleIds,
      positions,
      group.center,
      estimatedGroupRadius,
      modelDiagonal,
    );
    const cleanedGroup = removeSmallDisconnectedFragments(
      grownGroup,
      positions,
      modelDiagonal,
      group.center,
      estimatedGroupRadius,
    );
    for (const triangleId of cleanedGroup) {
      selectedTriangleIds.add(triangleId);
    }
  }

  return {
    triangleIds: selectedTriangleIds,
    groupCount: groups.length,
  };
}
