"use client";

import {
  AlertCircle,
  CheckCircle2,
  Crosshair,
  Download,
  Eye,
  FileUp,
  Layers3,
  LoaderCircle,
  RotateCcw,
  ScanSearch,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import styles from "./ScanbodyWorkspace.module.css";
import { captureModelViews } from "@/lib/capture-views";
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
  BASELINE_SELECTION,
  getAiModelOption,
  type AiModelId,
  type ReasoningEffort,
} from "@/lib/ai-models";
import {
  CAPTURE_SIZE,
  type AnalysisResult,
  type CapturedView,
  type Detection,
  type TriangleIdMap,
} from "@/lib/scanbody-types";

type ViewerState = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
};

type ModelInfo = {
  key: string;
  name: string;
  triangles: number;
  dimensions: [number, number, number];
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

function formatRunCost(
  meta: AnalysisResult["meta"],
  compact = false,
) {
  if (!meta.freeTierEligible) {
    return formatEstimatedCost(meta.estimatedCostUsd);
  }
  return compact
    ? "Free*"
    : `Free tier · ${formatEstimatedCost(meta.estimatedCostUsd)} paid`;
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
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [selectedTriangleIds, setSelectedTriangleIds] = useState<Set<number>>(
    () => new Set(),
  );
  const [selectedObjectCount, setSelectedObjectCount] = useState(0);
  const [baselineRun, setBaselineRun] = useState<SavedBenchmarkRun | null>(null);
  const [comparisonRuns, setComparisonRuns] = useState<SavedBenchmarkRun[]>([]);
  const [comparisonModel, setComparisonModel] =
    useState<AiModelId>("gpt-6.1-sol");
  const [comparisonEffort, setComparisonEffort] =
    useState<ReasoningEffort>("low");
  const [analysisNotice, setAnalysisNotice] = useState<string | null>(null);
  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;

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
    viewer.camera.near = Math.max(distance / 1000, 0.001);
    viewer.camera.far = distance * 20;
    viewer.camera.updateProjectionMatrix();
    viewer.controls.target.copy(sphere.center);
    viewer.controls.update();
  }, []);

  const loadFile = useCallback(
    async (file: File) => {
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
      setCapturedViews([]);
      setSelectedTriangleIds(new Set());
      setSelectedObjectCount(0);
      setBaselineRun(null);
      setComparisonRuns([]);
      triangleIdMapsRef.current.clear();
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

        orientGeometryWithPca(geometry);

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
          triangles: Math.floor(positions.count / 3),
          dimensions: [size.x, size.y, size.z],
        });
        resetViewer();

        setBusyLabel("Rendering 8 views");
        await new Promise((resolve) => requestAnimationFrame(resolve));
        const captures = captureModelViews(geometry);
        triangleIdMapsRef.current = captures.triangleIdMaps;
        setCapturedViews(captures.views);

        // Load real JSON files from the project's benchmark-output folder.
        try {
          const savedRuns = await loadBenchmarkFiles(meshKey);
          const savedBaseline =
            savedRuns.find((run) => run.kind === "baseline") ?? null;
          const savedComparisons = savedRuns.filter(
            (run) => run.kind === "comparison",
          );
          setBaselineRun(savedBaseline);
          setComparisonRuns(savedComparisons);
          if (savedBaseline) {
            setAnalysis(savedBaseline.analysis);
            setSelectedTriangleIds(
              new Set(savedBaseline.selectedTriangleIds),
            );
            setSelectedObjectCount(savedBaseline.objectCount);
            setAnalysisNotice("Saved Sol Medium baseline loaded for this STL.");
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
    [resetViewer],
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
        ? [{ detections: viewResult, triangleIdMap }]
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
    runKind: "baseline" | "comparison",
  ) => {
    if (capturedViews.length === 0) return;
    if (runKind === "comparison" && !baselineRun) {
      setError("Run and save the Sol Medium baseline first.");
      return;
    }
    setBusyLabel("Finding scanbodies");
    setError(null);
    setAnalysisNotice(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageSize: CAPTURE_SIZE,
          run:
            runKind === "baseline"
              ? { type: "baseline" }
              : {
                  type: "comparison",
                  model: comparisonModel,
                  reasoningEffort: comparisonEffort,
                },
          // Send every captured camera angle. A scanbody that is unclear in one
          // render can be much easier to recognize from another side.
          views: capturedViews.map((view) => ({
            name: view.name,
            image: view.image,
          })),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Analysis failed.");
      const result = payload as AnalysisResult;
      const { triangleIds, groupCount } = mapAnalysisToTriangles(result);
      setAnalysis(result);
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
      const runModel =
        runKind === "baseline" ? BASELINE_SELECTION.model : comparisonModel;
      const runEffort =
        runKind === "baseline"
          ? BASELINE_SELECTION.reasoningEffort
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
            : `comparison:${modelInfo.key}:${Date.now()}:${runModel}:${runEffort}`,
        storageVersion: BENCHMARK_STORAGE_VERSION,
        meshKey: modelInfo.key,
        meshName: modelInfo.name,
        kind: runKind,
        model: runModel,
        reasoningEffort: runEffort,
        createdAt: new Date().toISOString(),
        analysis: result,
        selectedTriangleIds: [...triangleIds].sort((a, b) => a - b),
        objectCount: groupCount,
        metrics,
      };

      try {
        if (runKind === "baseline") {
          // Replacing the baseline invalidates comparisons made against the old
          // one, so those old candidate runs are cleared deliberately.
          const savedPath = await saveBenchmarkFile(savedRun);
          setBaselineRun(savedRun);
          setComparisonRuns([]);
          setAnalysisNotice(
            `Sol Medium baseline saved to ${savedPath}`,
          );
        } else {
          const savedPath = await saveBenchmarkFile(savedRun);
          setComparisonRuns((current) => [savedRun, ...current]);
          setAnalysisNotice(
            `Comparison saved to ${savedPath}. ${metrics?.sameTrianglePercent.toFixed(1) ?? "0.0"}% same triangles as Sol Medium.`,
          );
        }
      } catch (storageError) {
        const message =
          storageError instanceof Error
            ? storageError.message
            : "browser storage failed";
        setAnalysisNotice(
          `Detection finished, but its JSON file could not be saved: ${message}`,
        );
      }
    } catch (analysisError) {
      const message = analysisError instanceof Error ? analysisError.message : "Analysis failed.";
      setError(message);
    } finally {
      setBusyLabel(null);
    }
  }, [
    baselineRun,
    capturedViews,
    comparisonEffort,
    comparisonModel,
    mapAnalysisToTriangles,
    modelInfo,
  ]);

  const showSavedRun = useCallback((run: SavedBenchmarkRun) => {
    setAnalysis(run.analysis);
    setSelectedTriangleIds(new Set(run.selectedTriangleIds));
    setSelectedObjectCount(run.objectCount);
    setAnalysisNotice(
      `${run.analysis.meta.modelLabel} result loaded from browser storage.`,
    );
  }, []);

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
    () => new Map(analysis?.views.map((view) => [view.view, view.detections]) ?? []),
    [analysis],
  );

  const detectionCount = analysis?.views.reduce(
    (total, view) => total + view.detections.length,
    0,
  );
  const comparisonModelOption = getAiModelOption(comparisonModel);
  const latestComparison = comparisonRuns[0] ?? null;

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
            <label
              className={`${styles.uploadButton} ${
                busyLabel ? styles.uploadDisabled : ""
              }`}
            >
              <Upload size={17} />
              Choose STL
              <input
                className={styles.fileInput}
                type="file"
                accept=".stl,model/stl"
                disabled={Boolean(busyLabel)}
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  event.currentTarget.value = "";
                  if (file) void loadFile(file);
                }}
              />
            </label>

            {modelInfo ? (
              <div className={styles.modelDetails}>
                <strong title={modelInfo.name}>{modelInfo.name}</strong>
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
                  <span><CheckCircle2 size={14} />Baseline saved</span>
                  <button type="button" onClick={() => showSavedRun(baselineRun)}>
                    Show
                  </button>
                </div>
                <div className={styles.savedRunStats}>
                  <span>{baselineRun.selectedTriangleIds.length.toLocaleString()} triangles</span>
                  <span>{baselineRun.objectCount} objects</span>
                  <span>{(baselineRun.analysis.meta.elapsedMs / 1000).toFixed(1)}s</span>
                  <span>{formatRunCost(baselineRun.analysis.meta, true)}</span>
                </div>
              </div>
            ) : (
              <p className={styles.benchmarkHint}>
                This becomes the fixed answer that every cheaper model is compared with.
              </p>
            )}

            <div className={styles.comparisonDivider}>Compare another run</div>
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
                  }}
                >
                  <optgroup label="OpenAI">
                    {AI_MODEL_OPTIONS.filter(
                      (option) => option.provider === "openai",
                    ).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Google Gemini">
                    {AI_MODEL_OPTIONS.filter(
                      (option) => option.provider === "google",
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
            {comparisonModelOption.freeTierEligible ? (
              <p className={styles.freeTierHint}>
                Free-tier call: $0 until Google&apos;s quota. The paid-price
                equivalent is still saved for later comparison.
              </p>
            ) : null}
            <button
              className={styles.comparisonButton}
              type="button"
              onClick={() => void runAnalysis("comparison")}
              disabled={
                !baselineRun ||
                capturedViews.length === 0 ||
                Boolean(busyLabel)
              }
            >
              {busyLabel === "Finding scanbodies" ? (
                <LoaderCircle className={styles.spin} size={17} />
              ) : (
                <ScanSearch size={17} />
              )}
              Run and compare
            </button>

            {latestComparison?.metrics ? (
              <div className={styles.comparisonCard}>
                <div className={styles.savedRunHeading}>
                  <strong>{latestComparison.analysis.meta.modelLabel}</strong>
                  <button
                    type="button"
                    onClick={() => showSavedRun(latestComparison)}
                  >
                    Show
                  </button>
                </div>
                <div className={styles.metricGrid}>
                  <span><strong>{latestComparison.metrics.sameTrianglePercent.toFixed(1)}%</strong>Same triangles</span>
                  <span><strong>{latestComparison.metrics.baselineFoundPercent.toFixed(1)}%</strong>Baseline found</span>
                  <span><strong>{latestComparison.metrics.missedBaselinePercent.toFixed(1)}%</strong>Missed</span>
                  <span><strong>{latestComparison.metrics.extraCandidatePercent.toFixed(1)}%</strong>Extra</span>
                </div>
                <div className={styles.runCostLine}>
                  <span>{(latestComparison.analysis.meta.elapsedMs / 1000).toFixed(1)} seconds</span>
                  <span>{formatRunCost(latestComparison.analysis.meta)}</span>
                  <span>{latestComparison.objectCount} objects</span>
                </div>
              </div>
            ) : null}

            {comparisonRuns.length > 0 ? (
              <div className={styles.runHistory}>
                <span>Saved comparison table</span>
                <div className={styles.runHistoryHeader}>
                  <span>Model</span><span>Same</span><span>Time</span><span>Cost</span>
                </div>
                {comparisonRuns.slice(0, 8).map((run) => (
                  <button key={run.id} type="button" onClick={() => showSavedRun(run)}>
                    <span>{run.analysis.meta.modelLabel}</span>
                    <strong>{run.metrics?.sameTrianglePercent.toFixed(1) ?? "-"}%</strong>
                    <strong>{(run.analysis.meta.elapsedMs / 1000).toFixed(0)}s</strong>
                    <strong>{formatRunCost(run.analysis.meta, true)}</strong>
                  </button>
                ))}
              </div>
            ) : null}

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
              capturedViews.map((view) => (
                <ViewThumbnail
                  key={view.name}
                  view={view}
                  detections={detectionsByView.get(view.name) ?? []}
                />
              ))
            )}
          </div>

          {analysis ? (
            <div className={styles.analysisFooter}>
              <div title={analysis.meta.modelLabel}>
                <span>Current run</span>
                <strong>{analysis.meta.modelLabel.replace("GPT-6.1 ", "")}</strong>
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
                <strong>{formatRunCost(analysis.meta, true)}</strong>
              </div>
              <div>
                <span>Model time</span>
                <strong>{(analysis.meta.elapsedMs / 1000).toFixed(1)}s</strong>
              </div>
            </div>
          ) : null}
        </aside>
      </div>
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
}: {
  view: CapturedView;
  detections: Detection[];
}) {
  return (
    <article className={styles.thumbnail}>
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
      </footer>
    </article>
  );
}

