"use client";

import {
  AlertCircle,
  CheckCircle2,
  Crosshair,
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
import { findTriangleIdsInsideOutline } from "@/lib/triangle-selection";
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
  name: string;
  triangles: number;
  dimensions: [number, number, number];
};

const BASE_COLOR = new THREE.Color(0xcbd2d5);
const HIGHLIGHT_COLOR = new THREE.Color(0xff5a36);

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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const viewerRef = useRef<ViewerState | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const sphereRef = useRef(new THREE.Sphere());
  const triangleIdMapsRef = useRef(new Map<string, TriangleIdMap>());

  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [capturedViews, setCapturedViews] = useState<CapturedView[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [selectedTriangleIds, setSelectedTriangleIds] = useState<Set<number>>(
    () => new Set(),
  );
  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = viewportRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xedf1f2);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10_000);
    camera.position.set(80, 65, 100);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
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
      setAnalysis(null);
      setCapturedViews([]);
      setSelectedTriangleIds(new Set());
      triangleIdMapsRef.current.clear();

      try {
        const parsedGeometry = new STLLoader().parse(await file.arrayBuffer());
        const geometry = parsedGeometry.index
          ? parsedGeometry.toNonIndexed()
          : parsedGeometry;
        if (geometry !== parsedGeometry) parsedGeometry.dispose();
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
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : "Could not read this STL.";
        setError(message);
      } finally {
        setBusyLabel(null);
      }
    },
    [resetViewer],
  );

  const selectTrianglesFromAnalysis = useCallback((result: AnalysisResult) => {
    const triangleIds = new Set<number>();

    // Each camera view has a same-sized hidden triangle-ID image. Filling the
    // AI outline over that image gives us real STL face numbers, not a 3D box.
    for (const viewResult of result.views) {
      const triangleIdMap = triangleIdMapsRef.current.get(viewResult.view);
      if (!triangleIdMap) continue;

      for (const detection of viewResult.detections) {
        if (detection.confidence < 0.35) continue;
        const idsUnderOutline = findTriangleIdsInsideOutline(
          detection.outline,
          triangleIdMap,
        );
        for (const triangleId of idsUnderOutline) triangleIds.add(triangleId);
      }
    }

    setSelectedTriangleIds(triangleIds);
  }, []);

  const analyze = useCallback(async () => {
    if (capturedViews.length === 0) return;
    setBusyLabel("Finding scanbodies");
    setError(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageSize: CAPTURE_SIZE,
          views: capturedViews.map((view) => ({
            name: view.name,
            image: view.image,
          })),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Analysis failed.");
      const result = payload as AnalysisResult;
      setAnalysis(result);
      selectTrianglesFromAnalysis(result);
    } catch (analysisError) {
      const message = analysisError instanceof Error ? analysisError.message : "Analysis failed.";
      setError(message);
    } finally {
      setBusyLabel(null);
    }
  }, [capturedViews, selectTrianglesFromAnalysis]);

  const detectionsByView = useMemo(
    () => new Map(analysis?.views.map((view) => [view.view, view.detections]) ?? []),
    [analysis],
  );

  const detectionCount = analysis?.views.reduce(
    (total, view) => total + view.detections.length,
    0,
  );

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
            <input
              ref={fileInputRef}
              className={styles.hiddenInput}
              type="file"
              accept=".stl,model/stl"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void loadFile(file);
              }}
            />
            <button
              className={styles.uploadButton}
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={Boolean(busyLabel)}
            >
              <Upload size={17} />
              Choose STL
            </button>

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
              <h2>Detection</h2>
            </div>
            <button
              className={styles.analyzeButton}
              type="button"
              onClick={() => void analyze()}
              disabled={capturedViews.length === 0 || Boolean(busyLabel)}
            >
              {busyLabel === "Finding scanbodies" ? (
                <LoaderCircle className={styles.spin} size={17} />
              ) : (
                <ScanSearch size={17} />
              )}
              Analyze views
            </button>

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
            <button
              className={styles.secondaryButton}
              type="button"
              onClick={() => setSelectedTriangleIds(new Set())}
              disabled={selectedTriangleIds.size === 0}
            >
              Clear highlight
            </button>
          </section>

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
              <div>
                <span>2D detections</span>
                <strong>{detectionCount}</strong>
              </div>
              <div>
                <span>3D triangles</span>
                <strong>{selectedTriangleIds.size.toLocaleString()}</strong>
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

