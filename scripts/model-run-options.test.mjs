import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import ts from "typescript";
import { getAiModelOption, getBaselineSelection } from "../src/lib/ai-models.ts";
import { BENCHMARK_STORAGE_VERSION, compareTriangleSelections } from "../src/lib/benchmark.ts";
import { scaleOutlinePoints } from "../src/lib/image-input.ts";
import { chooseToothNumberView } from "../src/lib/tooth-number-view.ts";

const meshKey = "a".repeat(64);
const analysis = {
  views: [{ view: "top", detections: [{ label: "scanbody", outline: [[1, 1], [2, 1], [2, 2]], confidence: 0.9 }] }],
  meta: { model: "openrouter-gpt-6.1-terra", modelLabel: "Terra Low", provider: "openrouter",
    reasoningEffort: "low", elapsedMs: 100, inputTokens: 100, cachedInputTokens: 0,
    viewTimings: [{ view: "top", elapsedMs: 87.5 }],
    outputTokens: 20, estimatedCostUsd: 0.001, freeTierEligible: false },
};
const baseline = { selectedTriangleIds: [1, 2, 3], objectCount: 1 };

// Exercise the actual browser callback without WebGL, a real STL, or paid calls.
// Extracting it keeps these checks tied to the behavior used by the two toggles.
function loadRunAnalysis(context, callbackName = "runAnalysis") {
  const filename = fileURLToPath(new URL("../src/components/ScanbodyWorkspace.tsx", import.meta.url));
  const source = ts.createSourceFile(filename, readFileSync(filename, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let callback;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === callbackName) {
      callback = node.initializer.arguments[0].getText(source);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert(callback);
  const compiled = ts.transpileModule(`module.exports = ${callback};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  const loadedModule = { exports: {} };
  new Function(...Object.keys(context), "module", compiled)(...Object.values(context), loadedModule);
  return loadedModule.exports;
}

function client(save, existingBaseline, imageDimension = 1024, jpegQuality = 90, outlineInputs = { min: "16", max: "40" }, boxesOnly = false) {
  const state = { requests: [], saves: [], modelRuns: [], error: null };
  const clock = { elapsed: 0 };
  const selection = new Set(Array.from({ length: 8 }, (_, index) => `view-${index}`));
  const context = {
    capturedViews: Array.from({ length: 8 }, (_, index) => ({ name: `view-${index}`, image: `data:image/jpeg;base64,${index}` })),
    requestViews: Array.from({ length: 8 }, (_, index) => ({ name: `view-${index}`, image: `data:image/jpeg;base64,selected-${imageDimension}-${jpegQuality}-${index}` })),
    selectedViewNames: selection,
    boundingBoxesOnly: boxesOnly,
    imageDimension, jpegQuality,
    outlineMinPointsInput: outlineInputs.min, outlineMaxPointsInput: outlineInputs.max,
    baselineRun: existingBaseline,
    saveResultFiles: !save, saveModelResultFiles: save,
    baselineParallelViews: true, comparisonParallelViews: true,
    comparisonModel: "openrouter-gpt-6.1-terra", comparisonEffort: "low", comparisonSpeed: "standard",
    directClaude: false, useOpenRouterBaseline: true,
    pcaElapsedMs: 100.5, captureElapsedMs: 1134,
    preparationElapsedMs: 1234.5,
    performance: { now: () => clock.elapsed },
    modelInfo: { key: meshKey, name: "test.stl", folderName: "161" },
    getAiModelOption, getBaselineSelection, BENCHMARK_STORAGE_VERSION, compareTriangleSelections,
    CAPTURE_SIZE: 1024,
    mapAnalysisToTriangles: () => {
      assert.equal(boxesOnly, false, "box-only mode must never perform local 3D mapping");
      clock.elapsed += 27.5;
      return { triangleIds: new Set([1, 2]), groupCount: 1 };
    },
    fetch: async (url, options) => {
      assert.equal(url, "/api/analyze");
      state.requests.push(JSON.parse(options.body));
      clock.elapsed += 5000;
      const request = state.requests.at(-1);
      return new Response(JSON.stringify({ ...analysis, meta: { ...analysis.meta, imageSize: imageDimension, jpegQuality,
        detectionMode: request.boundingBoxesOnly ? "box" : "outline",
        selectedViews: request.views.map((view) => view.name),
        outlineMinPointsPerScanbody: request.outlineMinPoints, outlineMaxPointsPerScanbody: request.outlineMaxPoints } }));
    },
    saveBenchmarkFile: async (run) => { state.saves.push(run); return `${run.kind}.json`; },
    setModelRuns: (update) => { state.modelRuns = typeof update === "function" ? update(state.modelRuns) : update; },
  };
  for (const name of ["BusyLabel", "Error", "AnalysisNotice", "Analysis", "ToothNumbers", "ToothRightSideHint", "SelectedTriangleIds", "SelectedObjectCount", "BaselineRun"]) {
    context[`set${name}`] = (value) => { state[name[0].toLowerCase() + name.slice(1)] = value; };
  }
  return { state, run: loadRunAnalysis(context), selection };
}

test("all four save/compare combinations use the selected model's own save setting", async () => {
  for (const compare of [false, true]) {
    for (const save of [false, true]) {
      const { state, run } = client(save, compare ? baseline : null);
      const kind = compare ? "comparison" : "standalone";
      await run(kind);
      assert.equal(state.error, null);
      assert.equal(state.requests.length, 1);
      assert.equal(state.requests[0].run.type, kind);
      assert.equal(state.requests[0].run.model, "openrouter-gpt-6.1-terra");
      assert.equal(state.requests[0].run.reasoningEffort, "low");
      assert.equal(state.requests[0].saveFiles, save);
      assert.equal(state.requests[0].views.length, 8);
      assert.equal(state.requests[0].imageSize, 1024);
      assert.equal(state.requests[0].jpegQuality, 90);
      assert.equal(state.requests[0].outlineMinPoints, 16);
      assert.equal(state.requests[0].outlineMaxPoints, 40);
      assert.equal(state.saves.length, save ? 1 : 0);
      assert.equal(state.modelRuns[0].kind, kind);
      assert.equal(state.modelRuns[0].metrics === null, !compare);
      // Browser mapping excludes API waiting and preserves the server's LLM time.
      assert.equal(state.modelRuns[0].analysis.meta.mappingElapsedMs, 27.5);
      assert.equal(state.modelRuns[0].analysis.meta.pcaElapsedMs, 100.5);
      assert.equal(state.modelRuns[0].analysis.meta.captureElapsedMs, 1134);
      assert.equal(state.modelRuns[0].analysis.meta.preparationElapsedMs, 1234.5);
      assert.equal(state.modelRuns[0].analysis.meta.elapsedMs, 100);
      assert.deepEqual(state.modelRuns[0].analysis.meta.viewTimings, [{ view: "top", elapsedMs: 87.5 }]);
      assert.deepEqual([...state.selectedTriangleIds], [1, 2]);
    }
  }
});

test("both baseline and selected-model requests send the selected images and record their settings", async () => {
  for (const [size, quality] of [[896, 80], [768, 70]]) {
    for (const kind of ["baseline", "standalone"]) {
      const { state, run } = client(kind !== "baseline", null, size, quality);
      await run(kind);
      assert.equal(state.error, null);
      assert.equal(state.requests.length, 1);
      const request = state.requests[0];
      assert.equal(request.imageSize, size);
      assert.equal(request.jpegQuality, quality);
      assert.equal(request.views.length, 8);
      assert(request.views.every((view) => view.image.includes(`selected-${size}-${quality}-`)), "send prepared images, not the original 1024/90 JPEGs");
      const saved = state.saves[0];
      assert.equal(saved.analysis.meta.imageSize, size);
      assert.equal(saved.analysis.meta.jpegQuality, quality);
    }
  }
});

test("comparison off ignores an existing baseline; comparison on alone requires one", async () => {
  const standalone = client(false, baseline);
  await standalone.run("standalone");
  assert.equal(standalone.state.modelRuns[0].metrics, null);
  const comparison = client(false, null);
  await comparison.run("comparison");
  assert.equal(comparison.state.requests.length, 0);
  assert.match(comparison.state.error, /baseline first/);
});

test("only checked views are submitted and their names are preserved when saving each run kind", async () => {
  for (const kind of ["baseline", "comparison", "standalone"]) {
    for (const indices of [[0], [0, 2, 5], [1, 2, 3, 4, 6, 7]]) {
      const { state, run, selection } = client(kind !== "baseline", kind === "comparison" ? baseline : null);
      selection.clear();
      for (const index of indices) selection.add(`view-${index}`);
      await run(kind);
      assert.equal(state.error, null);
      assert.equal(state.requests.length, 1);
      const names = indices.map((index) => `view-${index}`);
      assert.deepEqual(state.requests[0].views.map((view) => view.name), names);
      assert.deepEqual(state.saves[0].analysis.meta.selectedViews, names);
      assert.deepEqual(state.analysis.meta.selectedViews, names);
      assert(state.requests[0].views.every((view) => view.image.includes(`selected-1024-90-${view.name.slice(5)}`)));
    }
  }
});

test("an empty camera selection cannot start any paid segmentation request", async () => {
  for (const kind of ["baseline", "comparison", "standalone"]) {
    const { state, run, selection } = client(false, kind === "comparison" ? baseline : null);
    selection.clear();
    await run(kind);
    assert.equal(state.requests.length, 0);
    assert.equal(state.saves.length, 0);
    assert.match(state.error, /Select at least one camera view/);
  }
});

test("outline inputs apply to baseline, comparison and standalone runs; a blank minimum means three points", async () => {
  for (const kind of ["baseline", "comparison", "standalone"]) {
    for (const inputs of [{ min: "", max: "8" }, { min: "6", max: "12" }, { min: "8", max: "8" }]) {
      const { state, run } = client(kind !== "baseline", kind === "comparison" ? baseline : null, 1024, 90, inputs);
      await run(kind);
      assert.equal(state.error, null);
      assert.equal(state.requests.length, 1);
      const minimum = inputs.min === "" ? 3 : Number(inputs.min);
      assert.equal(state.requests[0].outlineMinPoints, minimum);
      assert.equal(state.requests[0].outlineMaxPoints, Number(inputs.max));
      assert.equal(state.saves[0].analysis.meta.outlineMinPointsPerScanbody, minimum);
      assert.equal(state.saves[0].analysis.meta.outlineMaxPointsPerScanbody, Number(inputs.max));
    }
  }
});

test("invalid outline bounds cannot start a paid request", async () => {
  for (const inputs of [{ min: "16", max: "8" }, { min: "2", max: "40" },
    { min: "3.5", max: "40" }, { min: "3", max: "10001" }, { min: "", max: "" }]) {
    const { state, run } = client(false, null, 1024, 90, inputs);
    await run("standalone");
    assert.match(state.error, /Outline points/);
    assert.equal(state.requests.length, 0);
    assert.equal(state.saves.length, 0);
  }
});

test("box-only mode skips mapping/comparison and cannot replace a baseline through either run button", async () => {
  for (const kind of ["baseline", "comparison", "standalone"]) {
    for (const save of [false, true]) {
      // Each entry point retains its own save switch. Outline fields are
      // deliberately invalid because they must be ignored in box mode.
      const { state, run, selection } = client(kind === "baseline" ? !save : save,
        kind === "baseline" ? baseline : null, 768, 70, { min: "bad", max: "" }, true);
      selection.clear(); selection.add("view-0"); selection.add("view-2");
      await run(kind);
      assert.equal(state.error, null);
      assert.equal(state.requests.length, 1);
      const request = state.requests[0];
      assert.equal(request.boundingBoxesOnly, true);
      assert.equal(request.outlineMinPoints, undefined);
      assert.equal(request.outlineMaxPoints, undefined);
      assert.deepEqual(request.views.map((view) => view.name), ["view-0", "view-2"]);
      assert.equal(state.analysis.meta.mappingElapsedMs, null);
      assert.equal(state.analysis.meta.elapsedMs, 100, "LLM time must remain visible");
      assert.deepEqual([...state.selectedTriangleIds], []);
      assert.equal(state.selectedObjectCount, 0);
      assert.equal(state.baselineRun, undefined, "the baseline setter must not run for a box trial");
      assert.equal(state.modelRuns.length, 1);
      assert.equal(state.modelRuns[0].kind, "standalone");
      assert.equal(state.modelRuns[0].metrics, null);
      assert.equal(state.modelRuns[0].detectedScanbodyCount, undefined);
      assert.equal(state.modelRuns[0].analysis.meta.detectionMode, "box");
      assert.equal(state.saves.length, save ? 1 : 0);
    }
  }
});

test("tooth numbering rescales an existing result to the newly selected image dimensions", async () => {
  const requests = [];
  const errors = [];
  const state = {
    analysis: { ...analysis, views: [{ view: "top", detections: [{ label: "scanbody", confidence: 0.9,
      outline: [[96, 192], [576, 96], [768, 768]] }] }], meta: { ...analysis.meta, imageSize: 768 } },
    modelInfo: { key: meshKey, name: "test.stl" }, busyLabel: null,
    capturedViews: [{ name: "top", image: "original-1024-jpeg" }],
    requestViews: [{ name: "top", image: "selected-896-quality-70-jpeg" }],
    imageDimension: 896, jpegQuality: 70, useOpenRouterBaseline: true, saveResultFiles: false,
    toothJawHint: "auto", toothRightSideHint: "unknown", toothViewChoice: "auto",
    chooseToothNumberView, scaleOutlinePoints, CAPTURE_SIZE: 1024,
    setBusyLabel() {}, setToothNumbers() {}, setAnalysisNotice() {},
    setError: (message) => { if (message) errors.push(message); },
    fetch: async (url, options) => {
      assert.equal(url, "/api/tooth-numbers");
      requests.push(JSON.parse(options.body));
      return new Response(JSON.stringify({ assignments: [], savedPath: null }));
    },
  };
  await loadRunAnalysis(state, "estimateToothNumbers")();
  assert.deepEqual(errors, []);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].imageSize, 896);
  assert.equal(requests[0].jpegQuality, 70);
  assert.equal(requests[0].views.length, 1);
  assert.equal(requests[0].views[0].image, "selected-896-quality-70-jpeg");
  assert.deepEqual(requests[0].views[0].detections[0].outline, [[112, 224], [672, 112], [896, 896]]);
});

function loadFileRoutes(files) {
  const filename = fileURLToPath(new URL("../src/app/api/benchmarks/route.ts", import.meta.url));
  const require = createRequire(filename);
  const compiled = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const io = {
    mkdir: async () => {},
    writeFile: async (path, value) => { files.set(path, value); },
    readFile: async (path) => { if (!files.has(path)) throw new Error("ENOENT"); return files.get(path); },
    unlink: async (path) => { files.delete(path); },
    readdir: async (path, options) => [...files.keys()].filter((file) => dirname(file) === path).map((file) => {
      const name = file.slice(path.length + 1);
      return options?.withFileTypes ? { name, isFile: () => true } : name;
    }),
  };
  const loadedModule = { exports: {} };
  const localRequire = (name) => name === "node:fs/promises" ? io : require(name.startsWith("@/")
    ? fileURLToPath(new URL(`../src/${name.slice(2)}.ts`, import.meta.url)) : name);
  new Function("require", "module", "exports", compiled)(localRequire, loadedModule, loadedModule.exports);
  return loadedModule.exports;
}

test("standalone results save/load without a baseline and survive baseline replacement", async () => {
  const files = new Map();
  const routes = loadFileRoutes(files);
  const run = { id: "test", storageVersion: 1, meshKey, meshName: "test.stl", kind: "standalone",
    model: "openrouter-gpt-6.1-terra", reasoningEffort: "low", createdAt: "2026-11-11T12:00:00.000Z",
    analysis: { ...analysis, meta: { ...analysis.meta, selectedViews: ["top", "front", "left"], imageSize: 768, jpegQuality: 70, pcaElapsedMs: 100.5, captureElapsedMs: 1134, preparationElapsedMs: 1234.5, mappingElapsedMs: 27.5 } },
    selectedTriangleIds: [1, 2], objectCount: 1, detectedScanbodyCount: 1, metrics: null };
  const post = (value) => routes.POST(new Request("http://offline/api/benchmarks", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value),
  }));
  const get = async () => (await (await routes.GET(new Request(`http://offline/api/benchmarks?meshKey=${meshKey}`))).json()).runs;
  const response = await post(run);
  assert.equal(response.status, 200);
  assert.match((await response.json()).savedPath, /\/runs\//);
  const loadedRun = (await get())[0];
  assert.equal(loadedRun.kind, "standalone");
  assert.deepEqual(loadedRun.analysis.meta.selectedViews, ["top", "front", "left"]);
  assert.equal(loadedRun.analysis.meta.imageSize, 768);
  assert.equal(loadedRun.analysis.meta.jpegQuality, 70);
  assert.equal(loadedRun.analysis.meta.pcaElapsedMs, 100.5);
  assert.equal(loadedRun.analysis.meta.captureElapsedMs, 1134);
  assert.equal(loadedRun.analysis.meta.preparationElapsedMs, 1234.5);
  assert.equal(loadedRun.analysis.meta.mappingElapsedMs, 27.5);
  assert.deepEqual(loadedRun.analysis.meta.viewTimings, [{ view: "top", elapsedMs: 87.5 }]);
  const standalonePath = [...files.keys()][0];
  const standaloneBytes = files.get(standalonePath);
  assert.equal((await post({ ...run, kind: "baseline" })).status, 200);
  const metrics = compareTriangleSelections([1, 2, 3], new Set([1, 2]), 1, 1);
  assert.equal((await post({ ...run, kind: "comparison", metrics })).status, 200);
  assert.deepEqual((await get()).map((value) => value.kind).sort(), ["baseline", "comparison", "standalone"]);
  assert.equal((await post({ ...run, kind: "baseline" })).status, 200);
  assert.deepEqual((await get()).map((value) => value.kind).sort(), ["baseline", "standalone"]);
  assert.equal(files.get(standalonePath), standaloneBytes);
  assert.equal((await post({ ...run, metrics })).status, 400);
  const hydrogenRun = { ...run, id: "hydrogen-xhigh", model: "gemini-4-hydrogen", reasoningEffort: "xhigh",
    analysis: { ...run.analysis, meta: { ...run.analysis.meta, model: "gemini-4-hydrogen", provider: "google",
      modelLabel: "Gemini 4 Hydrogen · XHigh", reasoningEffort: "xhigh" } } };
  assert.equal((await post(hydrogenRun)).status, 200);
  const hydrogenSaved = (await get()).find((value) => value.id === hydrogenRun.id);
  assert.equal(hydrogenSaved.model, "gemini-4-hydrogen");
  assert.equal(hydrogenSaved.reasoningEffort, "xhigh");
  assert.equal(hydrogenSaved.analysis.meta.reasoningEffort, "xhigh");
});

test("box files round-trip separately and cannot overwrite a baseline or delete comparisons", async () => {
  const files = new Map();
  const routes = loadFileRoutes(files);
  const normal = { id: "normal", storageVersion: 1, meshKey, meshName: "test.stl", kind: "baseline",
    model: "openrouter-gpt-6.1-terra", reasoningEffort: "low", createdAt: "2026-11-17T12:00:00.000Z",
    analysis, selectedTriangleIds: [1, 2], objectCount: 1, metrics: null };
  const post = (run) => routes.POST(new Request("http://offline/api/benchmarks", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(run),
  }));
  assert.equal((await post(normal)).status, 200);
  const comparison = { ...normal, id: "comparison", kind: "comparison", metrics: compareTriangleSelections([1, 2], new Set([1]), 1, 1) };
  assert.equal((await post(comparison)).status, 200);
  const originalFiles = new Map(files);
  const box = { ...normal, id: "boxes", kind: "standalone", selectedTriangleIds: [], objectCount: 0,
    analysis: { ...analysis,
      views: [{ view: "top", detections: [{ label: "scanbody", box: [10, 20, 30, 40],
        outline: [[10, 20], [30, 20], [30, 40], [10, 40]], confidence: 0.9 }] }],
      meta: { ...analysis.meta, detectionMode: "box", mappingElapsedMs: null } } };
  const response = await post(box);
  assert.equal(response.status, 200);
  const { savedPath } = await response.json();
  assert.match(savedPath, /[\\/]boxes[\\/]/);
  for (const [path, bytes] of originalFiles) assert.equal(files.get(path), bytes);
  const get = async () => (await (await routes.GET(new Request(`http://offline/api/benchmarks?meshKey=${meshKey}`))).json()).runs;
  const restored = (await get()).find((run) => run.id === "boxes");
  assert.equal(restored.analysis.meta.detectionMode, "box");
  assert.deepEqual(restored.analysis.views[0].detections[0].box, [10, 20, 30, 40]);
  assert.equal(restored.analysis.meta.mappingElapsedMs, null);
  assert.equal((await post({ ...box, kind: "baseline" })).status, 400);
  assert.equal((await post({ ...box, kind: "comparison", metrics: comparison.metrics })).status, 400);
  assert.equal((await post({ ...box, selectedTriangleIds: [1] })).status, 400);
  assert.equal(files.size, originalFiles.size + 1);
  const boxBytes = files.get(savedPath);
  assert.equal((await post(normal)).status, 200);
  assert.equal(files.get(savedPath), boxBytes, "replacing a normal baseline must preserve box trials");
});
