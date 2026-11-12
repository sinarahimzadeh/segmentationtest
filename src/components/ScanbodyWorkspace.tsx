"use client";

import {
  AlertCircle,
  CheckCircle2,
  Crosshair,
  Download,
  Eye,
  FileUp,
  Hash,
  Layers3,
  LoaderCircle,
  RotateCcw,
  ScanSearch,
  Search,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import styles from "./ScanbodyWorkspace.module.css";
import { captureModelViews } from "@/lib/capture-views";
import {
  IMAGE_DIMENSIONS, JPEG_QUALITIES, prepareImageViews, scaleOutlinePoints,
  type ImageDimension, type JpegQuality,
} from "@/lib/image-input";
import { orientGeometryWithPca } from "@/lib/pca";
import {
  createBinaryStlBlob,
  downloadBlob,
  type StlExportMode,
} from "@/lib/stl-export";
import { findGroupedTrianglesWithMultiViewAgreement } from "@/lib/triangle-selection";
import {
  BENCHMARK_STORAGE_VERSION,
  compareTriangleSelections,
  type SavedBenchmarkRun,
} from "@/lib/benchmark";
import {
  AI_MODEL_OPTIONS,
  getBaselineSelection,
  PROCESSING_SPEEDS,
  getAiModelOption,
  getProcessingSpeeds,
  getProcessingSpeedLabel,
  getSpeedPriceMultiplier,
  type AiModelId,
  type ReasoningEffort,
  type ProcessingSpeed,
} from "@/lib/ai-models";
import {
  CAPTURE_SIZE,
  type AnalysisResult,
  type CapturedView,
  type Detection,
  type TriangleIdMap,
} from "@/lib/scanbody-types";
import type { JawHint, RightSideHint, ToothNumberResult } from "@/lib/tooth-numbering";
import { chooseToothNumberView, type ToothViewChoice } from "@/lib/tooth-number-view";

type ViewerState = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: TrackballControls;
};

type ModelInfo = {
  key: string;
  name: string;
  folderName: string;
  triangles: number;
  dimensions: [number, number, number];
};

type FolderStlChoice = {
  id: string;
  file: File;
  folderName: string;
  relativePath: string;
};

const BASE_COLOR = new THREE.Color(0xcbd2d5);
const HIGHLIGHT_COLOR = new THREE.Color(0xff5a36);

async function createMeshKey(fileBytes: ArrayBuffer) {
  const digest = await crypto.subtle.digest("SHA-256", fileBytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function formatEstimatedCost(cost: number | null) {
  if (cost === null) return "-";
  return cost < 0.01 ? `$${cost.toFixed(4)}` : `$${cost.toFixed(3)}`;
}

function formatStageTime(elapsedMs: number | null | undefined) {
  if (elapsedMs == null) return "—";
  return elapsedMs < 1000 ? `${elapsedMs.toFixed(1)}ms` : `${(elapsedMs / 1000).toFixed(2)}s`;
}

function formatViewRequestTime(elapsedMs: number | undefined, sharedRequest: boolean) {
  if (elapsedMs != null) return `AI: ${formatStageTime(elapsedMs)}`;
  return sharedRequest ? "AI: shared request" : "AI: —";
}

function getParentFolderName(file: File) {
  const parts = file.webkitRelativePath
    .replaceAll("\\", "/")
    .split("/")
    .filter(Boolean);
  return parts.length >= 2 ? parts[parts.length - 2] : "Selected folder";
}

function formatRunCost(
  meta: AnalysisResult["meta"],
) {
  // Billing is determined by the project connected to the API key, and model
  // responses do not tell us whether a call was charged. Always show the paid
  // estimate instead of incorrectly labeling an eligible Gemini model "Free".
  return formatEstimatedCost(meta.estimatedCostUsd);
}

async function loadBenchmarkFiles(meshKey: string) {
  const response = await fetch(
    `/api/benchmarks?meshKey=${encodeURIComponent(meshKey)}`,
    { cache: "no-store" },
  );
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Could not load benchmark files.");
  }
  return payload.runs as SavedBenchmarkRun[];
}

async function saveBenchmarkFile(run: SavedBenchmarkRun) {
  const response = await fetch("/api/benchmarks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(run),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Could not save benchmark file.");
  }
  return payload.savedPath as string;
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material)
      ? child.material
      : [child.material];
    materials.forEach((material) => material.dispose());
  });
}

export default function ScanbodyWorkspace() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ViewerState | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const sphereRef = useRef(new THREE.Sphere());
  const triangleIdMapsRef = useRef(new Map<string, TriangleIdMap>());
  const originalPositionsRef = useRef<Float32Array | null>(null);

  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [capturedViews, setCapturedViews] = useState<CapturedView[]>([]);
  const [sourceCanvases, setSourceCanvases] = useState<ReadonlyMap<string, HTMLCanvasElement>>(() => new Map());
  const [imageDimension, setImageDimension] = useState<ImageDimension>(1024);
  const [jpegQuality, setJpegQuality] = useState<JpegQuality>(90);
  // Only the outgoing images change. PCA, cameras and 1024-pixel ID maps stay
  // fixed, so comparisons always refer to the same original STL triangles.
  const requestViews = useMemo(
    () => prepareImageViews(capturedViews, sourceCanvases, imageDimension, jpegQuality),
    [capturedViews, sourceCanvases, imageDimension, jpegQuality],
  );
  const [pcaElapsedMs, setPcaElapsedMs] = useState<number | null>(null);
  const [captureElapsedMs, setCaptureElapsedMs] = useState<number | null>(null);
  // Preserve the old combined field in saved JSON while recording both stages.
  const preparationElapsedMs = pcaElapsedMs == null || captureElapsedMs == null
    ? null : pcaElapsedMs + captureElapsedMs;
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [toothNumbers, setToothNumbers] = useState<ToothNumberResult | null>(null);
  const [toothJawHint, setToothJawHint] = useState<JawHint>("auto");
  const [toothRightSideHint, setToothRightSideHint] = useState<RightSideHint>("unknown");
  const [toothViewChoice, setToothViewChoice] = useState<ToothViewChoice>("auto");
  const [selectedTriangleIds, setSelectedTriangleIds] = useState<Set<number>>(
    () => new Set(),
  );
  const [selectedObjectCount, setSelectedObjectCount] = useState(0);
  const [baselineRun, setBaselineRun] = useState<SavedBenchmarkRun | null>(null);
  const [modelRuns, setModelRuns] = useState<SavedBenchmarkRun[]>([]);
  const [comparisonSearch, setComparisonSearch] = useState("");
  const [comparisonModel, setComparisonModel] =
    useState<AiModelId>("gpt-6.1-sol");
  const [comparisonEffort, setComparisonEffort] =
    useState<ReasoningEffort>("low");
  const [comparisonSpeed, setComparisonSpeed] = useState<ProcessingSpeed>("standard");
  const [analysisNotice, setAnalysisNotice] = useState<string | null>(null);
  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveResultFiles, setSaveResultFiles] = useState(true);
  const [loadSavedDetections, setLoadSavedDetections] = useState(false);
  // Selected models can run normally without saving or paying for a baseline.
  const [saveModelResultFiles, setSaveModelResultFiles] = useState(false);
  const [compareModelWithBaseline, setCompareModelWithBaseline] = useState(false);
  const [useOpenRouterBaseline, setUseOpenRouterBaseline] = useState(true);
  const [baselineParallelViews, setBaselineParallelViews] = useState(true);
  const [comparisonParallelViews, setComparisonParallelViews] = useState(true);
  const [directClaude, setDirectClaude] = useState(false);
  const [previewViewName, setPreviewViewName] = useState<string | null>(null);
  const [folderStlChoices, setFolderStlChoices] = useState<FolderStlChoice[]>([]);
  const [selectedFolderStlId, setSelectedFolderStlId] = useState("");

  useEffect(() => {
    if (!previewViewName) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPreviewViewName(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [previewViewName]);

  useEffect(() => {
    const container = viewportRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1416);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10_000);
    camera.position.set(80, 65, 100);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.domElement.draggable = false;

    // Keep browser-native dragging out of the 3D viewport. Without this,
    // Chrome can show its no-entry cursor and steal a mesh-rotation gesture.
    const preventNativeDrag = (event: DragEvent) => event.preventDefault();
    renderer.domElement.addEventListener("dragstart", preventNativeDrag);
    container.appendChild(renderer.domElement);

    // TrackballControls does not lock the camera's up direction, so the model
    // can be rotated continuously through every pole and rolled freely.
    const controls = new TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 1.05;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.3;
    controls.staticMoving = false;
    controls.dynamicDampingFactor = 0.12;
    controls.multiTouchRoll = true;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7376, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.1);
    keyLight.position.set(3, 5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x9bc5d7, 1.25);
    fillLight.position.set(-4, 2, -3);
    scene.add(fillLight);

    const resize = () => {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      controls.handleResize();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    let animationFrame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();
    viewerRef.current = { scene, camera, renderer, controls };

    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("dragstart", preventNativeDrag);
      renderer.dispose();
      renderer.domElement.remove();
      viewerRef.current = null;
    };
  }, []);

  const applyTriangleHighlights = useCallback(
    (triangleIds: ReadonlySet<number>) => {
      const mesh = meshRef.current;
      if (!mesh) return;

      const geometry = mesh.geometry;
      const positions = geometry.getAttribute("position") as THREE.BufferAttribute;
      const colors = geometry.getAttribute("color") as THREE.BufferAttribute;
      const triangleCount = Math.floor(positions.count / 3);

      // STLLoader gives us a non-indexed mesh: every three consecutive vertices
      // form one triangle. Color all three corners to highlight that exact face.
      for (
        let triangleIndex = 0;
        triangleIndex < triangleCount;
        triangleIndex += 1
      ) {
        const color = triangleIds.has(triangleIndex)
          ? HIGHLIGHT_COLOR
          : BASE_COLOR;
        const firstVertex = triangleIndex * 3;
        colors.setXYZ(firstVertex, color.r, color.g, color.b);
        colors.setXYZ(firstVertex + 1, color.r, color.g, color.b);
        colors.setXYZ(firstVertex + 2, color.r, color.g, color.b);
      }
      colors.needsUpdate = true;
    },
    [],
  );

  useEffect(() => {
    applyTriangleHighlights(selectedTriangleIds);
  }, [applyTriangleHighlights, selectedTriangleIds]);

  const resetViewer = useCallback(() => {
    const viewer = viewerRef.current;
    const sphere = sphereRef.current;
    if (!viewer || !meshRef.current) return;
    const distance = Math.max(sphere.radius * 2.8, 10);
    viewer.camera.position.set(distance * 0.78, distance * 0.62, distance);
    viewer.camera.up.set(0, 1, 0);
    viewer.camera.near = Math.max(distance / 1000, 0.001);
    viewer.camera.far = distance * 20;
    viewer.camera.updateProjectionMatrix();
    viewer.controls.target.copy(sphere.center);
    viewer.controls.update();
  }, []);

  const loadFile = useCallback(
    async (file: File, folderName: string) => {
      if (!file.name.toLowerCase().endsWith(".stl")) {
        setError("Choose an STL file for this first prototype.");
        return;
      }

      const viewer = viewerRef.current;
      if (!viewer) return;
      setBusyLabel("Preparing mesh");
      setError(null);
      setAnalysisNotice(null);
      setAnalysis(null);
      setPcaElapsedMs(null);
      setCaptureElapsedMs(null);
      setToothNumbers(null);
      setToothJawHint("auto");
      setToothRightSideHint("unknown");
      setToothViewChoice("auto");
      setCapturedViews([]);
      setSelectedTriangleIds(new Set());
      setSelectedObjectCount(0);
      setBaselineRun(null);
      setModelRuns([]);
      setComparisonSearch("");
      triangleIdMapsRef.current.clear();
      setSourceCanvases(new Map());
      originalPositionsRef.current = null;

      try {
        const fileBytes = await file.arrayBuffer();
        const meshKey = await createMeshKey(fileBytes);
        const parsedGeometry = new STLLoader().parse(fileBytes);
        const geometry = parsedGeometry.index
          ? parsedGeometry.toNonIndexed()
          : parsedGeometry;
        if (geometry !== parsedGeometry) parsedGeometry.dispose();

        // Keep original scanner coordinates for STL export. The viewer geometry
        // is rotated by PCA below, but triangle order remains exactly the same.
        const originalPositions = geometry.getAttribute(
          "position",
        ) as THREE.BufferAttribute;
        const originalValues = new Float32Array(originalPositions.count * 3);
        for (let index = 0; index < originalPositions.count; index += 1) {
          originalValues[index * 3] = originalPositions.getX(index);
          originalValues[index * 3 + 1] = originalPositions.getY(index);
          originalValues[index * 3 + 2] = originalPositions.getZ(index);
        }
        originalPositionsRef.current = originalValues;

        // PCA STOPWATCH: start immediately before alignment and stop as soon as
        // it returns. File reading, STL parsing, and viewer setup are excluded.
        const pcaStartedAt = performance.now();
        orientGeometryWithPca(geometry);
        const pcaElapsedMs = performance.now() - pcaStartedAt;
        setPcaElapsedMs(pcaElapsedMs);

        const positions = geometry.getAttribute("position") as THREE.BufferAttribute;
        const colorValues = new Float32Array(positions.count * 3);
        for (let index = 0; index < positions.count; index += 1) {
          colorValues[index * 3] = BASE_COLOR.r;
          colorValues[index * 3 + 1] = BASE_COLOR.g;
          colorValues[index * 3 + 2] = BASE_COLOR.b;
        }
        geometry.setAttribute("color", new THREE.BufferAttribute(colorValues, 3));

        if (meshRef.current) {
          viewer.scene.remove(meshRef.current);
          disposeObject(meshRef.current);
        }

        const material = new THREE.MeshStandardMaterial({
          color: 0xffffff,
          vertexColors: true,
          roughness: 0.7,
          metalness: 0.02,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        viewer.scene.add(mesh);
        meshRef.current = mesh;

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();
        if (geometry.boundingSphere) sphereRef.current.copy(geometry.boundingSphere);
        const size = geometry.boundingBox?.getSize(new THREE.Vector3()) ?? new THREE.Vector3();

        setModelInfo({
          key: meshKey,
          name: file.name,
          folderName,
          triangles: Math.floor(positions.count / 3),
          dimensions: [size.x, size.y, size.z],
        });
        resetViewer();

        setBusyLabel("Rendering 8 views");
        await new Promise((resolve) => requestAnimationFrame(resolve));
        // CAPTURE STOPWATCH: includes renderer setup, all eight JPEG captures,
        // hidden triangle-ID renders/readback, and capture-resource cleanup.
        const captureStartedAt = performance.now();
        const captures = captureModelViews(geometry);
        const captureElapsedMs = performance.now() - captureStartedAt;
        setCaptureElapsedMs(captureElapsedMs);
        triangleIdMapsRef.current = captures.triangleIdMaps;
        setSourceCanvases(captures.sourceCanvases);
        setCapturedViews(captures.views);

        // Load real JSON files from the project's benchmark-output folder.
        try {
          const savedRuns = await loadBenchmarkFiles(meshKey);
          const savedBaseline =
            savedRuns.find((run) => run.kind === "baseline") ?? null;
          const savedModelRuns = savedRuns.filter((run) => run.kind !== "baseline")
            .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
          setBaselineRun(savedBaseline);
          setModelRuns(savedModelRuns);
          const displayedRun = savedBaseline ?? savedModelRuns[0];
          // Keep saved baselines/history available for comparisons, but only
          // restore an old result into the viewer when explicitly enabled.
          if (loadSavedDetections && displayedRun) {
            setAnalysis(displayedRun.analysis);
            setSelectedTriangleIds(
              new Set(displayedRun.selectedTriangleIds),
            );
            setSelectedObjectCount(displayedRun.objectCount);
            setAnalysisNotice(savedBaseline
              ? "Saved Sol Medium baseline loaded for this STL."
              : `${displayedRun.analysis.meta.modelLabel} saved run loaded for this STL.`);
          }
        } catch {
          setAnalysisNotice(
            "The STL loaded, but the saved benchmark files could not be opened.",
          );
        }
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : "Could not read this STL.";
        setError(message);
      } finally {
        setBusyLabel(null);
      }
    },
    [resetViewer, loadSavedDetections],
  );

  const mapAnalysisToTriangles = useCallback((result: AnalysisResult) => {
    const mesh = meshRef.current;
    if (!mesh) return { triangleIds: new Set<number>(), groupCount: 0 };
    const positions = mesh.geometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;
    const triangleCount = Math.floor(positions.count / 3);
    const viewEvidence = result.views.flatMap((viewResult) => {
      const triangleIdMap = triangleIdMapsRef.current.get(viewResult.view);
      return triangleIdMap
        ? [{ detections: {
            ...viewResult,
            detections: viewResult.detections.map((detection) => ({
              ...detection,
              outline: scaleOutlinePoints(detection.outline, result.meta.imageSize ?? CAPTURE_SIZE, triangleIdMap.width),
            })),
          }, triangleIdMap }]
        : [];
    });

    // Views now vote on each visible triangle. This replaces the old raw union
    // where a single inaccurate outline could permanently add mesh noise.
    return findGroupedTrianglesWithMultiViewAgreement(
      triangleCount,
      positions,
      viewEvidence,
    );
  }, []);

  const runAnalysis = useCallback(async (
    runKind: SavedBenchmarkRun["kind"],
  ) => {
    if (capturedViews.length === 0) return;
    if (runKind === "comparison" && !baselineRun) {
      setError("Run the Sol Medium baseline first.");
      return;
    }
    const shouldSaveFiles = runKind === "baseline" ? saveResultFiles : saveModelResultFiles;
    setBusyLabel("Finding scanbodies");
    setError(null);
    setAnalysisNotice(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageSize: imageDimension,
          jpegQuality,
          // The selected run's own save switch controls both its result file and
          // raw-response JSON. Baseline/tooth saving has a separate switch.
          saveFiles: shouldSaveFiles,
          // Each section controls its own request mode independently.
          parallelViews: runKind === "baseline" ? baselineParallelViews : comparisonParallelViews,
          // The toggle changes only Claude comparisons; the baseline stays Sol.
          directClaude:
            runKind !== "baseline" &&
            getAiModelOption(comparisonModel).provider === "anthropic" &&
            directClaude,
          run:
            runKind === "baseline"
              ? { type: "baseline", useOpenRouter: useOpenRouterBaseline }
              : {
                  type: runKind,
                  model: comparisonModel,
                  reasoningEffort: comparisonEffort,
                  processingSpeed: comparisonSpeed,
                },
          // Send every captured camera angle. A scanbody that is unclear in one
          // render can be much easier to recognize from another side.
          views: requestViews.map((view) => ({
            name: view.name,
            image: view.image,
          })),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Analysis failed.");
      const apiResult = payload as AnalysisResult;
      // MAPPING STOPWATCH: starts after the API response has been received and
      // parsed; stops when local mapping, voting, and mesh cleanup return.
      // Highlight rendering, exports, and saving files happen after this timer.
      // The server's existing LLM elapsedMs remains unchanged.
      const mappingStartedAt = performance.now();
      const { triangleIds, groupCount } = mapAnalysisToTriangles(apiResult);
      const mappingElapsedMs = performance.now() - mappingStartedAt;
      const result: AnalysisResult = {
        ...apiResult,
        meta: { ...apiResult.meta, pcaElapsedMs, captureElapsedMs, preparationElapsedMs, mappingElapsedMs },
      };
      setAnalysis(result);
      setToothNumbers(null);
      setToothRightSideHint("unknown");
      setSelectedTriangleIds(triangleIds);
      setSelectedObjectCount(groupCount);
      const returnedDetectionCount = result.views.reduce(
        (total, view) => total + view.detections.length,
        0,
      );
      if (returnedDetectionCount === 0) {
        setAnalysisNotice(
          "The AI model finished successfully but returned zero scanbodies. No triangles were selected.",
        );
      }

      if (!modelInfo) return;
      // Use the same provider selection as the API when labeling saved files.
      const baselineSelection = getBaselineSelection(useOpenRouterBaseline);
      const runModel =
        runKind === "baseline" ? baselineSelection.model : comparisonModel;
      const runEffort =
        runKind === "baseline"
          ? baselineSelection.reasoningEffort
          : comparisonEffort;
      const metrics =
        runKind === "comparison" && baselineRun
          ? compareTriangleSelections(
              baselineRun.selectedTriangleIds,
              triangleIds,
              baselineRun.objectCount,
              groupCount,
            )
          : null;
      const savedRun: SavedBenchmarkRun = {
        id:
          runKind === "baseline"
            ? `baseline:v${BENCHMARK_STORAGE_VERSION}:${modelInfo.key}`
            : `${runKind}:${modelInfo.key}:${Date.now()}:${runModel}:${runEffort}:${result.meta.processingSpeed ?? "standard"}`,
        storageVersion: BENCHMARK_STORAGE_VERSION,
        meshKey: modelInfo.key,
        meshName: modelInfo.name,
        meshFolderName: modelInfo.folderName,
        kind: runKind,
        model: runModel,
        reasoningEffort: runEffort,
        ...(runKind !== "baseline" && result.meta.processingSpeed
          ? { processingSpeed: result.meta.processingSpeed }
          : {}),
        createdAt: new Date().toISOString(),
        analysis: result,
        selectedTriangleIds: [...triangleIds].sort((a, b) => a - b),
        objectCount: groupCount,
        // Keep the baseline JSON unchanged. Other model runs get this clearer
        // field so the final number of separate 3D scanbodies is easy to find.
        ...(runKind !== "baseline"
          ? { detectedScanbodyCount: groupCount }
          : {}),
        metrics,
      };

      // Results always remain available in memory for the current session,
      // regardless of whether writing JSON files is enabled.
      if (runKind === "baseline") {
        // Replacing the baseline invalidates comparisons made against the old
        // one, so those old candidate runs are cleared deliberately.
        setBaselineRun(savedRun);
        setModelRuns((current) => current.filter((run) => run.kind === "standalone"));
      } else {
        setModelRuns((current) => [savedRun, ...current]);
      }

      if (!shouldSaveFiles) {
        setAnalysisNotice(
          runKind === "baseline"
            ? "Sol Medium baseline is ready for this session. File saving is off."
            : runKind === "standalone"
              ? "Segmentation is ready. No baseline comparison or files saved."
              : `Comparison is ready for this session. ${metrics?.sameTrianglePercent.toFixed(1) ?? "0.0"}% same triangles as Sol Medium. File saving is off.`,
        );
      } else {
        try {
          if (runKind === "baseline") {
            const savedPath = await saveBenchmarkFile(savedRun);
            setAnalysisNotice(
              `Sol Medium baseline saved to ${savedPath}`,
            );
          } else {
            const savedPath = await saveBenchmarkFile(savedRun);
            setAnalysisNotice(
              runKind === "standalone"
                ? `Segmentation saved to ${savedPath}. No baseline comparison.`
                : `Comparison saved to ${savedPath}. ${metrics?.sameTrianglePercent.toFixed(1) ?? "0.0"}% same triangles as Sol Medium.`,
            );
          }
        } catch (storageError) {
          const message =
            storageError instanceof Error
              ? storageError.message
              : "file saving failed";
          setAnalysisNotice(
            `Detection finished, but its JSON file could not be saved: ${message}`,
          );
        }
      }
    } catch (analysisError) {
      const message = analysisError instanceof Error ? analysisError.message : "Analysis failed.";
      setError(message);
    } finally {
      setBusyLabel(null);
    }
  }, [
    baselineParallelViews,
    baselineRun,
    capturedViews,
    requestViews,
    imageDimension,
    jpegQuality,
    comparisonEffort,
    comparisonModel,
    comparisonParallelViews,
    comparisonSpeed,
    directClaude,
    mapAnalysisToTriangles,
    modelInfo,
    pcaElapsedMs,
    captureElapsedMs,
    preparationElapsedMs,
    saveResultFiles,
    saveModelResultFiles,
    useOpenRouterBaseline,
  ]);

  // This is the only browser action that calls the tooth-numbering API. Loading
  // an STL, segmenting, and viewing saved comparisons never trigger this call.
  const estimateToothNumbers = useCallback(async () => {
    if (!analysis || !modelInfo || busyLabel || capturedViews.length === 0) return;
    const viewName = chooseToothNumberView(capturedViews.map((view) => view.name), analysis.views, toothViewChoice);
    const referenceView = requestViews.find((view) => view.name === viewName);
    const detections = analysis.views.find((view) => view.view === viewName)?.detections ?? [];
    if (!referenceView || detections.length === 0) {
      setError("Tooth numbering needs a teeth-facing image with detected scanbody outlines. Try the other view.");
      return;
    }
    setBusyLabel("Estimating tooth numbers");
    setError(null);
    setToothNumbers(null);
    try {
      const response = await fetch("/api/tooth-numbers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          meshKey: modelInfo.key, meshName: modelInfo.name,
          imageSize: imageDimension, jpegQuality, useOpenRouter: useOpenRouterBaseline,
          saveFiles: saveResultFiles, jawHint: toothJawHint, rightSideHint: toothRightSideHint,
          referenceView: referenceView.name,
          views: [{
            name: referenceView.name, image: referenceView.image,
            detections: detections
              .map((detection, index) => ({
                detectionId: `${referenceView.name}:${index + 1}`,
                outline: scaleOutlinePoints(detection.outline, analysis.meta.imageSize ?? CAPTURE_SIZE, imageDimension),
              })),
          }],
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Tooth numbering failed.");
      setToothNumbers(result as ToothNumberResult);
      if (result.saveError) setAnalysisNotice(result.saveError);
      else if (result.savedPath) setAnalysisNotice(`Tooth-number JSON saved to ${result.savedPath}`);
    } catch (numberingError) {
      setError(numberingError instanceof Error ? numberingError.message : "Tooth numbering failed.");
    } finally {
      setBusyLabel(null);
    }
  }, [analysis, modelInfo, busyLabel, capturedViews, requestViews, imageDimension, jpegQuality, useOpenRouterBaseline, saveResultFiles, toothJawHint, toothRightSideHint, toothViewChoice]);

  const showSavedRun = useCallback((run: SavedBenchmarkRun) => {
    if (busyLabel) return;
    setAnalysis(run.analysis);
    setToothNumbers(null);
    setToothRightSideHint("unknown");
    setSelectedTriangleIds(new Set(run.selectedTriangleIds));
    setSelectedObjectCount(run.objectCount);
    setAnalysisNotice(
      `${run.analysis.meta.modelLabel} result loaded.`,
    );
  }, [busyLabel]);

  const exportStl = useCallback(
    async (mode: StlExportMode) => {
      const originalPositions = originalPositionsRef.current;
      if (!modelInfo || !originalPositions || selectedTriangleIds.size === 0) {
        return;
      }

      setBusyLabel("Preparing STL download");
      setError(null);
      await new Promise((resolve) => requestAnimationFrame(resolve));

      try {
        const blob = createBinaryStlBlob(
          originalPositions,
          selectedTriangleIds,
          mode,
        );
        const baseName = modelInfo.name.replace(/\.stl$/i, "");
        const suffix =
          mode === "selected" ? "scanbodies" : "without-scanbodies";
        downloadBlob(blob, `${baseName}-${suffix}.stl`);
      } catch (exportError) {
        const message =
          exportError instanceof Error
            ? exportError.message
            : "Could not export this STL.";
        setError(message);
      } finally {
        setBusyLabel(null);
      }
    },
    [modelInfo, selectedTriangleIds],
  );

  const detectionsByView = useMemo(
    () => new Map(analysis?.views.map((view) => [view.view, view.detections.map((detection) => ({
      ...detection,
      outline: scaleOutlinePoints(detection.outline, analysis.meta.imageSize ?? CAPTURE_SIZE, CAPTURE_SIZE),
    }))]) ?? []),
    [analysis],
  );
  const requestTimesByView = useMemo(
    () => new Map(analysis?.meta.viewTimings?.map((timing) => [timing.view, timing.elapsedMs]) ?? []),
    [analysis],
  );
  const sharedImageRequest = analysis?.meta.viewTimings === null;
  const previewView = requestViews.find(
    (view) => view.name === previewViewName,
  );

  const detectionCount = analysis?.views.reduce(
    (total, view) => total + view.detections.length,
    0,
  );
  const comparisonModelOption = getAiModelOption(comparisonModel);
  const toothReferenceName = chooseToothNumberView(capturedViews.map((view) => view.name), analysis?.views ?? [], toothViewChoice);
  const toothReferenceLabel = toothReferenceName === "bottom" ? "Bottom" : "Top";
  const hasToothTargets = Boolean(toothReferenceName && analysis?.views.find((view) => view.view === toothReferenceName)?.detections.length);
  const toothFdiNumbers = [...new Set(toothNumbers?.assignments.flatMap((assignment) =>
    assignment.toothFdi === null ? [] : [assignment.toothFdi]) ?? [])].sort((a, b) => a - b);
  const uncertainToothReasons = [...new Set(toothNumbers?.assignments
    .filter((assignment) => assignment.toothFdi === null)
    .map((assignment) => assignment.reason) ?? [])];
  const latestModelRun = modelRuns[0] ?? null;
  // Ignore separators such as the middle dot, so "terra high" finds
  // "GPT-6.1 Terra · High" without requiring the user to type that symbol.
  const searchTerms = comparisonSearch.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  const visibleModelRuns = modelRuns.filter((run) => {
    const modelText = run.analysis.meta.modelLabel
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ");
    return searchTerms.every((term) => modelText.includes(term));
  });

  return (
    <main className={styles.appShell}>
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <Crosshair size={19} strokeWidth={2.1} />
          <span>Scanbody ROI</span>
        </div>
        <div className={styles.topbarStatus}>
          <span className={styles.localDot} />
          Local mesh processing
        </div>
      </header>

      <div className={styles.workspace}>
        <aside className={styles.controlsPanel}>
          <section className={styles.panelSection}>
            <div className={styles.sectionHeading}>
              <FileUp size={16} />
              <h2>Model</h2>
            </div>
            <label className={styles.saveToggle}>
              <span>
                <strong>Load saved detections</strong>
                <small>When choosing an STL · Off: start without highlights</small>
              </span>
              <input
                type="checkbox"
                checked={loadSavedDetections}
                disabled={Boolean(busyLabel)}
                onChange={(event) => setLoadSavedDetections(event.currentTarget.checked)}
              />
              <span className={styles.toggleTrack} aria-hidden="true"><span /></span>
            </label>
            <label
              className={`${styles.uploadButton} ${
                busyLabel ? styles.uploadDisabled : ""
              }`}
            >
              <Upload size={17} />
              Choose STL folder
              <input
                className={styles.fileInput}
                type="file"
                accept=".stl,model/stl"
                multiple
                ref={(input) => {
                  // Browsers only reveal the parent folder name when the user
                  // selects a folder. A normal file picker hides local paths.
                  if (input) input.webkitdirectory = true;
                }}
                disabled={Boolean(busyLabel)}
                onChange={(event) => {
                  const stlFiles = Array.from(event.currentTarget.files ?? [])
                    .filter((file) => file.name.toLowerCase().endsWith(".stl"))
                    .map((file, index) => ({
                      id: `${file.webkitRelativePath}:${file.size}:${file.lastModified}:${index}`,
                      file,
                      folderName: getParentFolderName(file),
                      relativePath: file.webkitRelativePath || file.name,
                    }))
                    .sort((left, right) =>
                      left.relativePath.localeCompare(right.relativePath),
                    );
                  event.currentTarget.value = "";
                  if (stlFiles.length === 0) {
                    setFolderStlChoices([]);
                    setSelectedFolderStlId("");
                    setError("That folder contains no STL files.");
                    return;
                  }
                  if (stlFiles.length === 1) {
                    setFolderStlChoices([]);
                    setSelectedFolderStlId("");
                    void loadFile(stlFiles[0].file, stlFiles[0].folderName);
                    return;
                  }
                  setError(null);
                  setAnalysisNotice(
                    `Found ${stlFiles.length} STL files. Choose one below.`,
                  );
                  setFolderStlChoices(stlFiles);
                  setSelectedFolderStlId(stlFiles[0].id);
                }}
              />
            </label>

            {folderStlChoices.length > 1 ? (
              <div className={styles.folderStlPicker}>
                <select
                  aria-label="STL file in selected folder"
                  value={selectedFolderStlId}
                  onChange={(event) => setSelectedFolderStlId(event.target.value)}
                  disabled={Boolean(busyLabel)}
                >
                  {folderStlChoices.map((choice) => (
                    <option key={choice.id} value={choice.id}>
                      {choice.relativePath}
                    </option>
                  ))}
                </select>
                <button
                  className={styles.secondaryButton}
                  type="button"
                  disabled={Boolean(busyLabel)}
                  onClick={() => {
                    const choice = folderStlChoices.find(
                      (candidate) => candidate.id === selectedFolderStlId,
                    );
                    if (!choice) return;
                    setFolderStlChoices([]);
                    setSelectedFolderStlId("");
                    void loadFile(choice.file, choice.folderName);
                  }}
                >
                  Load selected STL
                </button>
              </div>
            ) : null}

            {modelInfo ? (
              <div className={styles.modelDetails}>
                <strong title={modelInfo.folderName}>{modelInfo.folderName}</strong>
                <span title={modelInfo.name}>{modelInfo.name}</span>
                <span title={modelInfo.key}>ID: {modelInfo.key.slice(0, 12)}...</span>
                <span>{modelInfo.triangles.toLocaleString()} triangles</span>
                <span>
                  {modelInfo.dimensions.map((value) => value.toFixed(1)).join(" x ")}
                </span>
              </div>
            ) : null}
          </section>

          <section className={styles.panelSection}>
            <div className={styles.sectionHeading}>
              <ScanSearch size={16} />
              <h2>Benchmark</h2>
            </div>
            <label className={styles.saveToggle}>
              <span>
                <strong>Save baseline and tooth results</strong>
                <small>Selected models have their own save switch below</small>
              </span>
              <input
                type="checkbox"
                checked={saveResultFiles}
                disabled={Boolean(busyLabel)}
                onChange={(event) => setSaveResultFiles(event.currentTarget.checked)}
              />
              <span className={styles.toggleTrack} aria-hidden="true">
                <span />
              </span>
            </label>
            {/* This switch changes only the next baseline run. */}
            <label className={styles.saveToggle}>
              <span>
                <strong>Parallel baseline views</strong>
                <small>On: one request per view · Off: all views together</small>
                <small>Separate requests can change cost and results.</small>
              </span>
              <input
                type="checkbox"
                checked={baselineParallelViews}
                disabled={Boolean(busyLabel)}
                onChange={(event) => setBaselineParallelViews(event.currentTarget.checked)}
              />
              <span className={styles.toggleTrack} aria-hidden="true">
                <span />
              </span>
            </label>
            {/* Provider changes apply to the next baseline; saved results stay loaded. */}
            <label className={styles.saveToggle}>
              <span>
                <strong>Use OpenRouter for Sol Medium</strong>
                <small>On: OpenRouter · Off: OpenAI</small>
              </span>
              <input
                type="checkbox"
                checked={useOpenRouterBaseline}
                disabled={Boolean(busyLabel)}
                onChange={(event) => setUseOpenRouterBaseline(event.currentTarget.checked)}
              />
              <span className={styles.toggleTrack} aria-hidden="true">
                <span />
              </span>
            </label>
            <button
              className={styles.analyzeButton}
              type="button"
              onClick={() => void runAnalysis("baseline")}
              disabled={capturedViews.length === 0 || Boolean(busyLabel)}
            >
              {busyLabel === "Finding scanbodies" ? (
                <LoaderCircle className={styles.spin} size={17} />
              ) : (
                <ScanSearch size={17} />
              )}
              {baselineRun ? "Replace" : "Run"} Sol Medium baseline
            </button>

            {baselineRun ? (
              <div className={styles.baselineCard}>
                <div className={styles.savedRunHeading}>
                  <span><CheckCircle2 size={14} />Baseline ready · {baselineRun.analysis.meta.provider === "openrouter" ? "OpenRouter" : "OpenAI"}</span>
                  <button type="button" onClick={() => showSavedRun(baselineRun)}>
                    Show
                  </button>
                </div>
                <div className={styles.savedRunStats}>
                  <span>{baselineRun.selectedTriangleIds.length.toLocaleString()} triangles</span>
                  <span>{baselineRun.objectCount} objects</span>
                  <span>{(baselineRun.analysis.meta.elapsedMs / 1000).toFixed(1)}s</span>
                  <span>{formatRunCost(baselineRun.analysis.meta)}</span>
                </div>
              </div>
            ) : (
              <p className={styles.benchmarkHint}>
                This becomes the fixed answer that every cheaper model is compared with.
              </p>
            )}

            <div className={styles.comparisonDivider}>Run another model</div>
            <div className={styles.selectorGrid}>
              <label>
                <span>AI model</span>
                <select
                  value={comparisonModel}
                  disabled={Boolean(busyLabel)}
                  onChange={(event) => {
                    const model = event.currentTarget.value as AiModelId;
                    const firstEffort = getAiModelOption(model)
                      .comparisonEfforts[0];
                    setComparisonModel(model);
                    setComparisonEffort(firstEffort);
                    setComparisonSpeed("standard");
                  }}
                >
                  <optgroup label="OpenAI">
                    {AI_MODEL_OPTIONS.filter(
                      (option) =>
                        option.provider === "openai" &&
                        option.comparisonEfforts.length > 0,
                    ).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="xAI Grok">
                    {AI_MODEL_OPTIONS.filter(
                      (option) =>
                        option.provider === "xai" &&
                        option.comparisonEfforts.length > 0,
                    ).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Google Gemini">
                    {AI_MODEL_OPTIONS.filter(
                      (option) =>
                        option.provider === "google" &&
                        option.comparisonEfforts.length > 0,
                    ).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Anthropic Claude">
                    {AI_MODEL_OPTIONS.filter(
                      (option) =>
                        option.provider === "anthropic" &&
                        option.comparisonEfforts.length > 0,
                    ).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="OpenRouter">
                    {AI_MODEL_OPTIONS.filter(
                      (option) =>
                        option.provider === "openrouter" &&
                        option.comparisonEfforts.length > 0,
                    ).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </optgroup>
                </select>
              </label>
              <label>
                <span>Thinking</span>
                <select
                  value={comparisonEffort}
                  disabled={Boolean(busyLabel)}
                  onChange={(event) =>
                    setComparisonEffort(
                      event.currentTarget.value as ReasoningEffort,
                    )
                  }
                >
                  {comparisonModelOption.comparisonEfforts.map((effort) => (
                    <option key={effort} value={effort}>
                      {effort[0].toUpperCase() + effort.slice(1)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {comparisonModelOption.speedPriceMultipliers ? (
              <label className={styles.speedSelector}>
                <span>Speed</span>
                <select
                  value={comparisonSpeed}
                  disabled={Boolean(busyLabel)}
                  onChange={(event) => setComparisonSpeed(event.currentTarget.value as ProcessingSpeed)}
                >
                  {PROCESSING_SPEEDS.map((speed) => {
                    const available = getProcessingSpeeds(comparisonModel).includes(speed);
                    return (
                      <option key={speed} value={speed} disabled={!available}>
                        {getProcessingSpeedLabel(speed)}{available
                          ? ` · ${getSpeedPriceMultiplier(comparisonModel, speed)}× token price`
                          : " · unavailable for this model"}
                      </option>
                    );
                  })}
                </select>
                <small>Fast and Ultrafast require that speed tier. An unavailable tier returns an error.</small>
              </label>
            ) : null}
            {comparisonModelOption.provider === "anthropic" ? (
              <label className={styles.saveToggle}>
                <span>
                  <strong>Direct Claude API</strong>
                  <small>On: Anthropic SDK · Off: LangChain</small>
                </span>
                <input
                  type="checkbox"
                  checked={directClaude}
                  disabled={Boolean(busyLabel)}
                  onChange={(event) => setDirectClaude(event.currentTarget.checked)}
                />
                <span className={styles.toggleTrack} aria-hidden="true">
                  <span />
                </span>
              </label>
            ) : null}
            {/* Comparison mode is independent of the baseline switch above. */}
            <label className={styles.saveToggle}>
              <span>
                <strong>Parallel model views</strong>
                <small>On: one request per view · Off: all views together</small>
                <small>Separate requests can change cost and results.</small>
              </span>
              <input
                type="checkbox"
                checked={comparisonParallelViews}
                disabled={Boolean(busyLabel)}
                onChange={(event) => setComparisonParallelViews(event.currentTarget.checked)}
              />
              <span className={styles.toggleTrack} aria-hidden="true">
                <span />
              </span>
            </label>
            <label className={styles.saveToggle}>
              <span>
                <strong>Compare with Sol Medium</strong>
                <small>Off: run normally, without a baseline</small>
              </span>
              <input type="checkbox" checked={compareModelWithBaseline}
                disabled={Boolean(busyLabel)}
                onChange={(event) => setCompareModelWithBaseline(event.currentTarget.checked)} />
              <span className={styles.toggleTrack} aria-hidden="true"><span /></span>
            </label>
            <label className={styles.saveToggle}>
              <span>
                <strong>Save selected model results</strong>
                <small>Off: keep the result only for this session</small>
              </span>
              <input type="checkbox" checked={saveModelResultFiles}
                disabled={Boolean(busyLabel)}
                onChange={(event) => setSaveModelResultFiles(event.currentTarget.checked)} />
              <span className={styles.toggleTrack} aria-hidden="true"><span /></span>
            </label>
            {compareModelWithBaseline && !baselineRun ? (
              <p className={styles.benchmarkHint}>Run a Sol Medium baseline first, or switch comparison off.</p>
            ) : null}
            <button
              className={styles.comparisonButton}
              type="button"
              onClick={() => void runAnalysis(compareModelWithBaseline ? "comparison" : "standalone")}
              disabled={
                (compareModelWithBaseline && !baselineRun) ||
                capturedViews.length === 0 ||
                Boolean(busyLabel)
              }
            >
              {busyLabel === "Finding scanbodies" ? (
                <LoaderCircle className={styles.spin} size={17} />
              ) : (
                <ScanSearch size={17} />
              )}
              {compareModelWithBaseline ? "Run and compare" : "Run segmentation"}
            </button>

            {latestModelRun ? (
              <div className={styles.comparisonCard}>
                <div className={styles.savedRunHeading}>
                  <strong>{latestModelRun.analysis.meta.modelLabel}</strong>
                  <button
                    type="button"
                    onClick={() => showSavedRun(latestModelRun)}
                  >
                    Show
                  </button>
                </div>
                {latestModelRun.metrics ? (
                  <div className={styles.metricGrid}>
                    <span><strong>{latestModelRun.metrics.sameTrianglePercent.toFixed(1)}%</strong>Same triangles</span>
                    <span><strong>{latestModelRun.metrics.baselineFoundPercent.toFixed(1)}%</strong>Inlier</span>
                    <span><strong>{latestModelRun.metrics.missedBaselinePercent.toFixed(1)}%</strong>Missed</span>
                    <span><strong>{latestModelRun.metrics.extraCandidatePercent.toFixed(1)}%</strong>Outlier</span>
                  </div>
                ) : <p className={styles.benchmarkHint}>Standalone run · No baseline comparison</p>}
                <div className={styles.runCostLine}>
                  <span>{(latestModelRun.analysis.meta.elapsedMs / 1000).toFixed(1)} seconds</span>
                  <span>{formatRunCost(latestModelRun.analysis.meta)}</span>
                  <span>{latestModelRun.objectCount} objects</span>
                </div>
              </div>
            ) : null}

            <div className={styles.runHistory}>
                <span>Model runs</span>
                <label className={styles.runHistorySearch}>
                  <Search size={14} aria-hidden="true" />
                  <input
                    type="search"
                    aria-label="Search model runs"
                    placeholder="Search model name"
                    value={comparisonSearch}
                    onChange={(event) => setComparisonSearch(event.target.value)}
                  />
                </label>
                <div className={styles.runHistoryList}>
                  <div className={styles.runHistoryHeader}>
                    <span>Model</span><span>Same</span><span>Inlier</span><span>Outlier</span><span>Time</span><span>Cost</span>
                  </div>
                  {visibleModelRuns.map((run) => (
                    <button key={run.id} type="button" onClick={() => showSavedRun(run)}>
                      <span title={run.analysis.meta.modelLabel}>{run.analysis.meta.modelLabel}</span>
                      <strong>{run.metrics ? `${run.metrics.sameTrianglePercent.toFixed(1)}%` : "—"}</strong>
                      <strong>{run.metrics ? `${run.metrics.baselineFoundPercent.toFixed(1)}%` : "—"}</strong>
                      <strong>{run.metrics ? `${run.metrics.extraCandidatePercent.toFixed(1)}%` : "—"}</strong>
                      <strong>{(run.analysis.meta.elapsedMs / 1000).toFixed(0)}s</strong>
                      <strong>{formatRunCost(run.analysis.meta)}</strong>
                    </button>
                  ))}
                  {visibleModelRuns.length === 0 ? (
                    <p className={styles.runHistoryEmpty}>
                      {!modelInfo
                        ? "Choose an STL to load its saved runs."
                        : modelRuns.length === 0
                          ? "No model runs for this STL yet. Run a model to add a row."
                          : "No matching models. Clear the search to show all runs."}
                    </p>
                  ) : null}
                </div>
            </div>

            <div className={styles.pipeline}>
              <PipelineRow done={Boolean(modelInfo)} label="PCA alignment" />
              <PipelineRow done={capturedViews.length > 0} label="Camera views" />
              <PipelineRow done={Boolean(analysis)} label="Vision ROI" />
              <PipelineRow
                done={selectedTriangleIds.size > 0}
                label="Triangle mapping"
              />
            </div>
          </section>

          <section className={styles.panelSection}>
            <div className={styles.sectionHeading}>
              <h2>Image settings</h2>
            </div>
            <div className={styles.selectorGrid}>
              <label>
                <span>Dimensions</span>
                <select value={imageDimension} disabled={Boolean(busyLabel)} onChange={(event) => {
                  setImageDimension(Number(event.currentTarget.value) as ImageDimension);
                  setToothNumbers(null);
                }}>
                  {IMAGE_DIMENSIONS.map((size) => <option key={size} value={size}>{size} × {size}{size === 1024 ? " (default)" : ""}</option>)}
                </select>
              </label>
              <label>
                <span>JPEG quality</span>
                <select value={jpegQuality} disabled={Boolean(busyLabel)} onChange={(event) => {
                  setJpegQuality(Number(event.currentTarget.value) as JpegQuality);
                  setToothNumbers(null);
                }}>
                  {JPEG_QUALITIES.map((quality) => <option key={quality} value={quality}>{quality}%{quality === 90 ? " (default)" : ""}</option>)}
                </select>
              </label>
            </div>
            <p className={styles.imageSettingsNote}>Applies to the next run. All 8 views are kept.</p>
            {analysis ? <p className={styles.imageSettingsNote}>Displayed result: {analysis.meta.imageSize ?? CAPTURE_SIZE} × {analysis.meta.imageSize ?? CAPTURE_SIZE} · JPEG {analysis.meta.jpegQuality ?? 90}%</p> : null}
          </section>

          <section className={styles.panelSection}>
            <div className={styles.sectionHeading}>
              <Layers3 size={16} />
              <h2>Highlight</h2>
            </div>
            <div className={styles.selectionSummary}>
              <span>Selected triangles</span>
              <strong>{selectedTriangleIds.size.toLocaleString()}</strong>
            </div>
            <div className={styles.selectionSummary}>
              <span>Detected objects</span>
              <strong>{selectedObjectCount}</strong>
            </div>
            <button
              className={styles.secondaryButton}
              type="button"
              onClick={() => setSelectedTriangleIds(new Set())}
              disabled={selectedTriangleIds.size === 0}
            >
              Clear highlight
            </button>
          </section>

          <section className={styles.panelSection}>
            <div className={styles.sectionHeading}>
              <Hash size={16} />
              <h2>Tooth numbers (FDI)</h2>
            </div>
            <p className={styles.benchmarkHint}>
              Separate Sol Medium request via {useOpenRouterBaseline ? "OpenRouter" : "OpenAI"}.
              Sends one image: {toothReferenceName ? toothReferenceLabel : "none available"}, with its scanbody outlines.
            </p>
            <div className={styles.selectorGrid}>
              <label>
                <span>Image showing the teeth</span>
                <select value={toothViewChoice} disabled={Boolean(busyLabel)} onChange={(event) => {
                  setToothViewChoice(event.currentTarget.value as ToothViewChoice);
                  setToothRightSideHint("unknown");
                  setToothNumbers(null);
                }}>
                  <option value="auto">Auto — more scanbodies visible</option>
                  <option value="top">Top image</option>
                  <option value="bottom">Bottom image</option>
                </select>
              </label>
              <label>
                <span>Jaw (optional hint)</span>
                <select value={toothJawHint} disabled={Boolean(busyLabel)} onChange={(event) => {
                  setToothJawHint(event.currentTarget.value as JawHint);
                  setToothNumbers(null);
                }}>
                  <option value="auto">Let AI estimate</option>
                  <option value="upper">Upper jaw</option>
                  <option value="lower">Lower jaw</option>
                </select>
              </label>
              <label>
                <span>Patient right in selected image</span>
                <select value={toothRightSideHint} disabled={Boolean(busyLabel)} onChange={(event) => {
                  setToothRightSideHint(event.currentTarget.value as RightSideHint);
                  setToothNumbers(null);
                }}>
                  <option value="unknown">Unknown</option>
                  <option value="image-left">Image left</option>
                  <option value="image-right">Image right</option>
                </select>
              </label>
            </div>
            <button className={styles.secondaryButton} type="button"
              disabled={!toothReferenceName}
              onClick={() => toothReferenceName && setPreviewViewName(toothReferenceName)}>
              <Eye size={16} /> Preview {toothReferenceName ? toothReferenceLabel : "image"}
            </button>
            <p className={styles.benchmarkHint}>Auto uses existing detections. Check the preview and switch views if needed.</p>
            <button className={styles.comparisonButton} type="button"
              onClick={() => void estimateToothNumbers()}
              disabled={Boolean(busyLabel) || !hasToothTargets}>
              {busyLabel === "Estimating tooth numbers"
                ? <LoaderCircle className={styles.spin} size={17} /> : <Hash size={17} />}
              Estimate tooth numbers
            </button>
            {!hasToothTargets ? <p className={styles.benchmarkHint}>Choose a view with detected scanbodies, or run segmentation first.</p> : null}
            {toothNumbers ? (
              <div className={styles.toothNumberResult} role="status">
                <strong>Estimated scanbody tooth positions</strong>
                {toothFdiNumbers.length > 0 ? (
                  <div className={styles.toothNumberList}>
                    {toothFdiNumbers.map((number) => <span key={number}>{number}</span>)}
                  </div>
                ) : <p>No reliable tooth numbers returned.</p>}
                {toothNumbers.notes ? <p>{toothNumbers.notes}</p> : null}
                {uncertainToothReasons.length > 0 ? (
                  <details>
                    <summary>Some positions are uncertain</summary>
                    {uncertainToothReasons.map((reason) => <p key={reason}>{reason}</p>)}
                  </details>
                ) : null}
                <p>{toothNumbers.meta.modelLabel}</p>
                <div className={styles.runCostLine}>
                  <span>{(toothNumbers.meta.elapsedMs / 1000).toFixed(1)} seconds</span>
                  <span>{formatEstimatedCost(toothNumbers.meta.estimatedCostUsd)}</span>
                </div>
              </div>
            ) : null}
          </section>

          <section className={styles.panelSection}>
            <div className={styles.sectionHeading}>
              <Download size={16} />
              <h2>Export</h2>
            </div>
            <div className={styles.exportActions}>
              <button
                className={styles.exportButton}
                type="button"
                onClick={() => void exportStl("selected")}
                disabled={selectedTriangleIds.size === 0 || Boolean(busyLabel)}
              >
                <Download size={16} />
                Scanbodies STL
              </button>
              <button
                className={styles.exportButton}
                type="button"
                onClick={() => void exportStl("remaining")}
                disabled={selectedTriangleIds.size === 0 || Boolean(busyLabel)}
              >
                <Download size={16} />
                Remaining jaw STL
              </button>
            </div>
          </section>

          {analysisNotice ? (
            <div className={styles.noticeMessage} role="status">
              <AlertCircle size={16} />
              <span>{analysisNotice}</span>
            </div>
          ) : null}

          {error ? (
            <div className={styles.errorMessage} role="alert">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          ) : null}
        </aside>

        <section className={styles.viewerPanel}>
          <div ref={viewportRef} className={styles.viewport} />
          {!modelInfo ? (
            <div className={styles.emptyState}>
              <Layers3 size={30} />
              <strong>Choose an STL to begin</strong>
            </div>
          ) : null}
          {busyLabel && busyLabel !== "Finding scanbodies" ? (
            <div className={styles.busyBadge}>
              <LoaderCircle className={styles.spin} size={16} />
              {busyLabel}
            </div>
          ) : null}
          <button
            className={styles.resetButton}
            type="button"
            title="Reset 3D view"
            aria-label="Reset 3D view"
            onClick={resetViewer}
            disabled={!modelInfo}
          >
            <RotateCcw size={18} />
          </button>
          <div className={styles.viewerLegend}>
            <span><i className={styles.meshSwatch} />Mesh</span>
            <span><i className={styles.roiSwatch} />ROI</span>
          </div>
        </section>

        <aside className={styles.reviewPanel}>
          <div className={styles.reviewHeader}>
            <div>
              <span className={styles.eyebrow}>Review</span>
              <h2>Camera views</h2>
            </div>
            <div className={styles.viewCount}>{capturedViews.length}/8</div>
          </div>

          <div className={styles.thumbnailList}>
            {capturedViews.length === 0 ? (
              <div className={styles.reviewEmpty}>
                <Eye size={22} />
                <span>No views yet</span>
              </div>
            ) : (
              requestViews.map((view) => (
                <ViewThumbnail
                  key={view.name}
                  view={view}
                  detections={detectionsByView.get(view.name) ?? []}
                  requestElapsedMs={requestTimesByView.get(view.name)}
                  sharedRequest={sharedImageRequest}
                  onOpen={() => setPreviewViewName(view.name)}
                />
              ))
            )}
          </div>

          {capturedViews.length > 0 || analysis ? (
            <div className={styles.analysisFooter}>
              <div className={styles.analysisStats}>
                <div title={analysis?.meta.modelLabel}>
                  <span>Current run</span>
                  <strong>{analysis?.meta.modelLabel.replace("GPT-6.1 ", "") ?? "—"}</strong>
                </div>
                <div>
                  <span>2D detections</span>
                  <strong>{detectionCount}</strong>
                </div>
                <div>
                  <span>3D triangles</span>
                  <strong>{selectedTriangleIds.size.toLocaleString()}</strong>
                </div>
                <div>
                  <span>Est. cost</span>
                  <strong>{analysis ? formatRunCost(analysis.meta) : "—"}</strong>
                </div>
              </div>
              <div className={styles.analysisTimings}>
                <div title="PCA alignment only. Uses the saved run's recorded value when available, otherwise the current STL load.">
                  <span>PCA</span>
                  <strong>{formatStageTime(analysis?.meta.pcaElapsedMs ?? pcaElapsedMs)}</strong>
                </div>
                <div title="All eight JPEG captures and hidden triangle-ID renders, including renderer setup, pixel readback, and cleanup.">
                  <span>Screenshots</span>
                  <strong>{formatStageTime(analysis?.meta.captureElapsedMs ?? captureElapsedMs)}</strong>
                </div>
                <div>
                  <span>Model time</span>
                  <strong>{analysis ? `${(analysis.meta.elapsedMs / 1000).toFixed(1)}s` : "—"}</strong>
                </div>
                <div title="Recorded time converting the returned 2D outlines into 3D triangles, including multiview voting and local cleanup.">
                  <span>3D mapping</span>
                  <strong>{formatStageTime(analysis?.meta.mappingElapsedMs)}</strong>
                </div>
              </div>
            </div>
          ) : null}
        </aside>
      </div>

      {previewView ? (
        <div
          className={styles.previewBackdrop}
          role="presentation"
          onClick={() => setPreviewViewName(null)}
        >
          <section
            className={styles.previewDialog}
            role="dialog"
            aria-modal="true"
            aria-label={`${previewView.label} camera view`}
            onClick={(event) => event.stopPropagation()}
          >
            <header className={styles.previewHeader}>
              <div>
                <span className={styles.eyebrow}>Camera view</span>
                <strong>{previewView.label}</strong>
                <small className={styles.previewRequestTime}>
                  {formatViewRequestTime(requestTimesByView.get(previewView.name), sharedImageRequest)}
                </small>
              </div>
              <button
                type="button"
                aria-label="Close camera view"
                title="Close"
                onClick={() => setPreviewViewName(null)}
              >
                <X size={20} />
              </button>
            </header>
            <LargeViewPreview
              view={previewView}
              detections={detectionsByView.get(previewView.name) ?? []}
            />
          </section>
        </div>
      ) : null}
    </main>
  );
}

function PipelineRow({ done, label }: { done: boolean; label: string }) {
  return (
    <div className={styles.pipelineRow}>
      {done ? (
        <CheckCircle2 size={15} className={styles.doneIcon} />
      ) : (
        <span className={styles.pendingDot} />
      )}
      <span>{label}</span>
    </div>
  );
}

function ViewThumbnail({
  view,
  detections,
  requestElapsedMs,
  sharedRequest,
  onOpen,
}: {
  view: CapturedView;
  detections: Detection[];
  requestElapsedMs?: number;
  sharedRequest: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      className={styles.thumbnail}
      type="button"
      onClick={onOpen}
      title={`Open ${view.label} camera view`}
    >
      <div className={styles.thumbnailImage}>
        {/* The image is generated locally from the uploaded STL. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={view.image} alt={`${view.label} STL render`} />
        <svg
          className={styles.outlineLayer}
          viewBox={`0 0 ${CAPTURE_SIZE} ${CAPTURE_SIZE}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {detections.map((detection, index) => (
            <polygon
              key={`${view.name}-${index}`}
              className={styles.detectionOutline}
              points={detection.outline.map(([x, y]) => `${x},${y}`).join(" ")}
            />
          ))}
        </svg>
        {detections.map((detection, index) => {
          const [firstX, firstY] = detection.outline[0] ?? [0, 0];
          return (
            <span
              key={`${view.name}-${index}-confidence`}
              className={styles.confidenceLabel}
              style={{
                left: `${(firstX / CAPTURE_SIZE) * 100}%`,
                top: `${(Math.max(firstY, 110) / CAPTURE_SIZE) * 100}%`,
              }}
            >
              {Math.round(detection.confidence * 100)}
            </span>
          );
        })}
      </div>
      <footer>
        <span>{view.label}</span>
        <span>{detections.length ? `${detections.length} ROI` : "Clear"}</span>
        <small className={styles.viewRequestTime} title={sharedRequest
          ? "All images were sent in one request; separate image times are unavailable."
          : "Elapsed API request time for this image, including provider/network waiting and response parsing. Not pure model inference time."}>
          {formatViewRequestTime(requestElapsedMs, sharedRequest)}
        </small>
      </footer>
    </button>
  );
}

function LargeViewPreview({
  view,
  detections,
}: {
  view: CapturedView;
  detections: Detection[];
}) {
  return (
    <div className={styles.previewCanvas}>
      {/* Generated locally from the STL; opening this preview makes no API call. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={view.image} alt={`${view.label} STL render enlarged`} />
      <svg
        className={styles.outlineLayer}
        viewBox={`0 0 ${CAPTURE_SIZE} ${CAPTURE_SIZE}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {detections.map((detection, index) => (
          <polygon
            key={`${view.name}-large-${index}`}
            className={styles.detectionOutline}
            points={detection.outline.map(([x, y]) => `${x},${y}`).join(" ")}
          />
        ))}
      </svg>
      {detections.map((detection, index) => {
        const [firstX, firstY] = detection.outline[0] ?? [0, 0];
        return (
          <span
            key={`${view.name}-large-${index}-confidence`}
            className={`${styles.confidenceLabel} ${styles.previewConfidence}`}
            style={{
              left: `${(firstX / CAPTURE_SIZE) * 100}%`,
              top: `${(Math.max(firstY, 55) / CAPTURE_SIZE) * 100}%`,
            }}
          >
            {Math.round(detection.confidence * 100)}
          </span>
        );
      })}
    </div>
  );
}

