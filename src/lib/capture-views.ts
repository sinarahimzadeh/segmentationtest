import * as THREE from "three";
import { CAPTURE_SIZE, type CapturedView } from "./scanbody-types";

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
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

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
  const cameras = new Map<string, THREE.OrthographicCamera>();
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

    renderer.render(scene, camera);
    views.push({
      name: view.name,
      label: view.label,
      image: renderer.domElement.toDataURL("image/jpeg", 0.9),
    });
    cameras.set(view.name, camera.clone());
  }

  material.dispose();
  renderer.dispose();
  return { views, cameras };
}

