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

function client(save, existingBaseline, imageDimension = 1024, jpegQuality = 90) {
  const state = { requests: [], saves: [], modelRuns: [], error: null };
  const clock = { elapsed: 0 };
  const context = {
    capturedViews: Array.from({ length: 8 }, (_, index) => ({ name: `view-${index}`, image: `data:image/jpeg;base64,${index}` })),
    requestViews: Array.from({ length: 8 }, (_, index) => ({ name: `view-${index}`, image: `data:image/jpeg;base64,selected-${imageDimension}-${jpegQuality}-${index}` })),
    imageDimension, jpegQuality,
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
      clock.elapsed += 27.5;
      return { triangleIds: new Set([1, 2]), groupCount: 1 };
    },
    fetch: async (url, options) => {
      assert.equal(url, "/api/analyze");
      state.requests.push(JSON.parse(options.body));
      clock.elapsed += 5000;
      return new Response(JSON.stringify({ ...analysis, meta: { ...analysis.meta, imageSize: imageDimension, jpegQuality } }));
    },
    saveBenchmarkFile: async (run) => { state.saves.push(run); return `${run.kind}.json`; },
    setModelRuns: (update) => { state.modelRuns = typeof update === "function" ? update(state.modelRuns) : update; },
  };
  for (const name of ["BusyLabel", "Error", "AnalysisNotice", "Analysis", "ToothNumbers", "ToothRightSideHint", "SelectedTriangleIds", "SelectedObjectCount", "BaselineRun"]) {
    context[`set${name}`] = (value) => { state[name[0].toLowerCase() + name.slice(1)] = value; };
  }
  return { state, run: loadRunAnalysis(context) };
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
    analysis: { ...analysis, meta: { ...analysis.meta, imageSize: 768, jpegQuality: 70, pcaElapsedMs: 100.5, captureElapsedMs: 1134, preparationElapsedMs: 1234.5, mappingElapsedMs: 27.5 } },
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
});
