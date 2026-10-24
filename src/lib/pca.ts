import * as THREE from "three";

type EigenPair = {
  value: number;
  vector: THREE.Vector3;
};

function jacobiEigen3(input: number[][]): EigenPair[] {
  const matrix = input.map((row) => [...row]);
  const vectors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  for (let iteration = 0; iteration < 60; iteration += 1) {
    let p = 0;
    let q = 1;
    let largest = Math.abs(matrix[p][q]);

    for (const [row, column] of [
      [0, 2],
      [1, 2],
    ]) {
      if (Math.abs(matrix[row][column]) > largest) {
        largest = Math.abs(matrix[row][column]);
        p = row;
        q = column;
      }
    }

    if (largest < 1e-10) break;

    const angle = 0.5 * Math.atan2(
      2 * matrix[p][q],
      matrix[q][q] - matrix[p][p],
    );
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);

    for (let index = 0; index < 3; index += 1) {
      const pValue = matrix[p][index];
      const qValue = matrix[q][index];
      matrix[p][index] = cosine * pValue - sine * qValue;
      matrix[q][index] = sine * pValue + cosine * qValue;
    }

    for (let index = 0; index < 3; index += 1) {
      const pValue = matrix[index][p];
      const qValue = matrix[index][q];
      matrix[index][p] = cosine * pValue - sine * qValue;
      matrix[index][q] = sine * pValue + cosine * qValue;
    }

    for (let index = 0; index < 3; index += 1) {
      const pValue = vectors[index][p];
      const qValue = vectors[index][q];
      vectors[index][p] = cosine * pValue - sine * qValue;
      vectors[index][q] = sine * pValue + cosine * qValue;
    }
  }

  return [0, 1, 2]
    .map((index) => ({
      value: matrix[index][index],
      vector: new THREE.Vector3(
        vectors[0][index],
        vectors[1][index],
        vectors[2][index],
      ).normalize(),
    }))
    .sort((left, right) => right.value - left.value);
}

export function orientGeometryWithPca(geometry: THREE.BufferGeometry) {
  const positions = geometry.getAttribute("position") as THREE.BufferAttribute;
  const sampleStep = Math.max(1, Math.floor(positions.count / 30_000));
  const mean = new THREE.Vector3();
  let sampleCount = 0;

  for (let index = 0; index < positions.count; index += sampleStep) {
    mean.x += positions.getX(index);
    mean.y += positions.getY(index);
    mean.z += positions.getZ(index);
    sampleCount += 1;
  }
  mean.multiplyScalar(1 / sampleCount);

  const covariance = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];

  for (let index = 0; index < positions.count; index += sampleStep) {
    const x = positions.getX(index) - mean.x;
    const y = positions.getY(index) - mean.y;
    const z = positions.getZ(index) - mean.z;
    covariance[0][0] += x * x;
    covariance[0][1] += x * y;
    covariance[0][2] += x * z;
    covariance[1][1] += y * y;
    covariance[1][2] += y * z;
    covariance[2][2] += z * z;
  }

  covariance[1][0] = covariance[0][1];
  covariance[2][0] = covariance[0][2];
  covariance[2][1] = covariance[1][2];

  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      covariance[row][column] /= sampleCount;
    }
  }

  const eigen = jacobiEigen3(covariance);
  const xAxis = eigen[0].vector;
  const zAxis = eigen[1].vector;
  const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();
  const source = new THREE.Vector3();

  for (let index = 0; index < positions.count; index += 1) {
    source.set(
      positions.getX(index) - mean.x,
      positions.getY(index) - mean.y,
      positions.getZ(index) - mean.z,
    );
    positions.setXYZ(
      index,
      source.dot(xAxis),
      source.dot(yAxis),
      source.dot(zAxis),
    );
  }

  positions.needsUpdate = true;
  geometry.deleteAttribute("normal");
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return {
    axes: [xAxis, yAxis, zAxis],
    variances: eigen.map((item) => item.value),
  };
}

