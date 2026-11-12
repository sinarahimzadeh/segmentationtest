import assert from "node:assert/strict";
import { test } from "node:test";
import { IMAGE_DIMENSIONS, JPEG_QUALITIES, prepareImageViews, scaleOutlinePoints } from "../src/lib/image-input.ts";

const views = Array.from({ length: 8 }, (_, index) => ({ name: `view-${index}`, label: `View ${index}`, image: `original-jpeg-${index}` }));
const sources = new Map(views.map((view) => [view.name, { original: view.name }]));

test("changing dimensions and JPEG quality prepares all eight images from original pixels; defaults restore the exact original JPEGs", (t) => {
  const draws = [];
  const encodes = [];
  const canvas = {
    width: 0, height: 0,
    getContext: () => ({ clearRect() {}, drawImage: (...args) => draws.push(args) }),
    toDataURL: (format, quality) => {
      encodes.push({ format, quality, width: canvas.width, height: canvas.height });
      return `data:${format};${canvas.width}x${canvas.height};quality=${quality}`;
    },
  };
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: () => canvas } });
  t.after(() => {
    if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
    else delete globalThis.document;
  });
  for (const size of IMAGE_DIMENSIONS) {
    for (const quality of JPEG_QUALITIES) {
      const result = prepareImageViews(views, sources, size, quality);
      assert.equal(result.length, 8);
      if (size === 1024 && quality === 90) {
        assert.equal(result, views);
        continue;
      }
      assert.deepEqual(result.map((view) => view.name), views.map((view) => view.name));
      assert(result.every((view) => view.image === `data:image/jpeg;${size}x${size};quality=${quality / 100}`));
      assert.deepEqual(draws.slice(-8).map((args) => args[0]), [...sources.values()]);
      assert(draws.slice(-8).every((args) => args[3] === size && args[4] === size));
    }
  }
  const expectedEncodes = (IMAGE_DIMENSIONS.length * JPEG_QUALITIES.length - 1) * views.length;
  assert.equal(encodes.length, expectedEncodes);
  assert.equal(prepareImageViews(views, sources, 1024, 90), views);
  assert.equal(encodes.length, expectedEncodes, "restoring defaults must not recompress any JPEG");
});

test("reduced-image outlines land on the same 1024-pixel positions for 3D mapping, previews and tooth requests", () => {
  const original = [[128, 256], [768, 128], [1024, 1024]];
  for (const size of IMAGE_DIMENSIONS) {
    const modelOutline = original.map(([x, y]) => [x * size / 1024, y * size / 1024]);
    assert.deepEqual(scaleOutlinePoints(modelOutline, size, 1024), original);
    assert.deepEqual(scaleOutlinePoints(original, 1024, size), modelOutline);
  }
  assert.equal(scaleOutlinePoints(original, 1024, 1024), original, "old 1024-pixel results stay unchanged");
});
