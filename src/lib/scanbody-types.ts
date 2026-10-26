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
    elapsedMs: number;
    inputTokens: number | null;
    outputTokens: number | null;
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

