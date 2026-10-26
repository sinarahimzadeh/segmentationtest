import * as THREE from "three";
import {
  CAPTURE_SIZE,
  type CapturedView,
  type TriangleIdMap,
} from "./scanbody-types";

const MAX_RGB_ID = 0xffffff;

const VIEW_CONFIGS = [
  { name: "top", label: "Top", direction: [0, 1, 0], up: [0, 0, -1] },
  { name: "bottom", label: "Bottom", direction: [0, -1, 0], up: [0, 0, 1] },
  { name: "front", label: "Front", direction: [0, 0, 1], up: [0, 1, 0] },
  { name: "back", label: "Back", direction: [0, 0, -1], up: [0, 1, 0] },
  { name: "left", label: "Left", direction: [-1, 0, 0], up: [0, 1, 0] },
  { name: "right", label: "Right", direction: [1, 0, 0], up: [0, 1, 0] },
  {
    name: "front-left",
    label: "Front left",
    direction: [-1, 0.8, 1],
    up: [0, 1, 0],
  },
  {
    name: "front-right",
    label: "Front right",
    direction: [1, 0.8, 1],
    up: [0, 1, 0],
  },
] as const;

/**
 * Builds a copy of the STL where every triangle has its own exact RGB color.
 * We encode triangle index + 1 because black (zero) is reserved for background.
 */
function createTriangleIdGeometry(geometry: THREE.BufferGeometry) {
  const idGeometry = geometry.clone();
  const positions = idGeometry.getAttribute("position");
  const triangleCount = Math.floor(positions.count / 3);
  if (triangleCount > MAX_RGB_ID) {
    idGeometry.dispose();
    throw new Error(
      `This mesh has ${triangleCount.toLocaleString()} triangles, but RGB triangle IDs support at most ${MAX_RGB_ID.toLocaleString()}.`,
    );
  }

  const colorValues = new Uint8Array(positions.count * 3);
  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += 1) {
    const encodedId = triangleIndex + 1;
    const red = encodedId & 0xff;
    const green = (encodedId >> 8) & 0xff;
    const blue = (encodedId >> 16) & 0xff;

    for (let corner = 0; corner < 3; corner += 1) {
      const vertexOffset = (triangleIndex * 3 + corner) * 3;
      colorValues[vertexOffset] = red;
      colorValues[vertexOffset + 1] = green;
      colorValues[vertexOffset + 2] = blue;
    }
  }

  idGeometry.setAttribute(
    "color",
    new THREE.Uint8BufferAttribute(colorValues, 3, true),
  );
  return idGeometry;
}

/** Reads the RGB render target and decodes it into top-left image coordinates. */
function readTriangleIdMap(
  renderer: THREE.WebGLRenderer,
  renderTarget: THREE.WebGLRenderTarget,
): TriangleIdMap {
  const rgba = new Uint8Array(CAPTURE_SIZE * CAPTURE_SIZE * 4);
  const encodedTriangleIds = new Uint32Array(CAPTURE_SIZE * CAPTURE_SIZE);
  renderer.readRenderTargetPixels(
    renderTarget,
    0,
    0,
    CAPTURE_SIZE,
    CAPTURE_SIZE,
    rgba,
  );

  // WebGL returns rows bottom-first. The AI outline uses normal top-left image
  // coordinates, so flip the rows while decoding RGB back into triangle IDs.
  for (let imageY = 0; imageY < CAPTURE_SIZE; imageY += 1) {
    const webGlY = CAPTURE_SIZE - 1 - imageY;
    for (let x = 0; x < CAPTURE_SIZE; x += 1) {
      const sourceOffset = (webGlY * CAPTURE_SIZE + x) * 4;
      encodedTriangleIds[imageY * CAPTURE_SIZE + x] =
        rgba[sourceOffset] |
        (rgba[sourceOffset + 1] << 8) |
        (rgba[sourceOffset + 2] << 16);
    }
  }

  return {
    width: CAPTURE_SIZE,
    height: CAPTURE_SIZE,
    encodedTriangleIds,
  };
}

export function captureModelViews(geometry: THREE.BufferGeometry) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(1);
  renderer.setSize(CAPTURE_SIZE, CAPTURE_SIZE, false);
  renderer.setClearColor(0xf3f5f6, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const material = new THREE.MeshStandardMaterial({
    color: 0xcbd2d5,
    roughness: 0.7,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  const visualMesh = new THREE.Mesh(geometry, material);
  scene.add(visualMesh);

  // This second mesh is never shown to the user. It produces a hidden image in
  // which every visible pixel tells us exactly which STL triangle is underneath.
  const triangleIdGeometry = createTriangleIdGeometry(geometry);
  const triangleIdMaterial = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  const triangleIdMesh = new THREE.Mesh(
    triangleIdGeometry,
    triangleIdMaterial,
  );
  triangleIdMesh.visible = false;
  scene.add(triangleIdMesh);

  const triangleIdTarget = new THREE.WebGLRenderTarget(
    CAPTURE_SIZE,
    CAPTURE_SIZE,
    {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
    },
  );
  triangleIdTarget.texture.colorSpace = THREE.NoColorSpace;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x667078, 2.1));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
  keyLight.position.set(1, 2, 3);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xb8d7e8, 1.5);
  fillLight.position.set(-3, 1, -2);
  scene.add(fillLight);

  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere ?? new THREE.Sphere();
  const halfSize = Math.max(sphere.radius * 1.12, 1);
  const far = Math.max(sphere.radius * 12, 100);
  const triangleIdMaps = new Map<string, TriangleIdMap>();
  const views: CapturedView[] = [];

  for (const view of VIEW_CONFIGS) {
    const camera = new THREE.OrthographicCamera(
      -halfSize,
      halfSize,
      halfSize,
      -halfSize,
      0.01,
      far,
    );
    const direction = new THREE.Vector3(...view.direction).normalize();
    camera.position.copy(sphere.center).addScaledVector(
      direction,
      Math.max(sphere.radius * 4, 10),
    );
    camera.up.set(view.up[0], view.up[1], view.up[2]);
    camera.lookAt(sphere.center);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);

    renderer.setRenderTarget(null);
    renderer.setClearColor(0xf3f5f6, 1);
    visualMesh.visible = true;
    triangleIdMesh.visible = false;
    renderer.render(scene, camera);
    views.push({
      name: view.name,
      label: view.label,
      image: renderer.domElement.toDataURL("image/jpeg", 0.9),
    });

    renderer.setRenderTarget(triangleIdTarget);
    renderer.setClearColor(0x000000, 1);
    visualMesh.visible = false;
    triangleIdMesh.visible = true;
    renderer.clear();
    renderer.render(scene, camera);
    triangleIdMaps.set(
      view.name,
      readTriangleIdMap(renderer, triangleIdTarget),
    );
  }

  renderer.setRenderTarget(null);
  material.dispose();
  triangleIdMaterial.dispose();
  triangleIdGeometry.dispose();
  triangleIdTarget.dispose();
  renderer.dispose();
  return { views, triangleIdMaps };
}

