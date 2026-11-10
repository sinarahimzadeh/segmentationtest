import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import ts from "typescript";
import { analyzeViewBatches } from "../src/lib/parallel-view-analysis.ts";

// No real API calls or API keys. Deferred responses prove concurrency without
// depending on timing, provider load, or paid segmentation runs.
const views = Array.from({ length: 8 }, (_, index) => ({
  name: `view-${index}`,
  image: `data:image/jpeg;base64,image-${index}`,
}));
const resultFor = (batch, overrides = {}) => ({
  parsed: { views: batch.map(({ name }) => ({ view: name, detections: [] })) },
  inputTokens: 100, cachedInputTokens: 2, outputTokens: 10,
  serviceTier: "default", reportedCostUsd: 0.01, estimatedCostUsd: 0.01,
  ...overrides,
});

test("all eight start together; reverse completion preserves view order and totals", async () => {
  const finish = [];
  const pending = analyzeViewBatches(views, true, (batch) => {
    assert.equal(batch.length, 1);
    const index = views.findIndex((view) => view === batch[0]);
    return new Promise((resolve) => { finish[index] = () => resolve(resultFor(batch)); });
  });
  assert.equal(finish.length, 8, "every call must start before any completes");
  for (const complete of finish.toReversed()) complete();
  const result = await pending;
  assert.deepEqual(result.parsed.views.map((view) => view.view), views.map((view) => view.name));
  assert.equal(result.viewRequestCount, 8);
  assert.equal(result.inputTokens, 800);
  assert.equal(result.cachedInputTokens, 16);
  assert.equal(result.outputTokens, 80);
  assert(Math.abs(result.estimatedCostUsd - 0.08) < 1e-12);
  assert(Math.abs(result.reportedCostUsd - 0.08) < 1e-12);
});

test("off keeps all eight images in one request", async () => {
  let calls = 0;
  const result = await analyzeViewBatches(views, false, async (batch) => {
    calls++;
    assert.deepEqual(batch, views);
    return resultFor(batch);
  });
  assert.equal(calls, 1);
  assert.equal(result.viewRequestCount, 1);
  assert.equal(result.inputTokens, 100);
  assert.equal(result.estimatedCostUsd, 0.01);
});

test("a failed view waits for other calls and never returns an incomplete success", async () => {
  const finish = [];
  let returned = false;
  const pending = analyzeViewBatches(views, true, (batch) => new Promise((resolve, reject) => {
    finish.push(() => batch[0].name === "view-0"
      ? reject(new Error("provider overloaded"))
      : resolve(resultFor(batch)));
  })).finally(() => { returned = true; });
  const rejected = assert.rejects(pending, /1\/8.*No partial result was saved.*view-0: provider overloaded/);
  finish[0]();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(returned, false);
  for (const complete of finish.slice(1)) complete();
  await rejected;
});

test("missing, duplicate or wrong output IDs fail; an explicit empty detection is valid", async () => {
  for (const returnedViews of [[], [{ view: "wrong", detections: [] }], [
    { view: "view-0", detections: [] }, { view: "view-0", detections: [] },
  ]]) {
    await assert.rejects(analyzeViewBatches(views.slice(0, 1), true, async (batch) =>
      resultFor(batch, { parsed: { views: returnedViews } })), /exactly the requested VIEW_ID/);
  }
  const result = await analyzeViewBatches(views.slice(0, 1), true, async (batch) => resultFor(batch));
  assert.deepEqual(result.parsed.views[0].detections, []);
});

test("missing usage stays unknown; mixed receipts still sum each request's cost", async () => {
  const result = await analyzeViewBatches(views.slice(0, 2), true, async (batch) => resultFor(batch,
    batch[0].name === "view-0"
      ? { inputTokens: null, reportedCostUsd: null, estimatedCostUsd: 0.03, serviceTier: null }
      : { reportedCostUsd: 0.02, estimatedCostUsd: 0.02 }));
  assert.equal(result.inputTokens, null);
  assert.equal(result.reportedCostUsd, null);
  assert.equal(result.estimatedCostUsd, 0.05);
  assert.equal(result.serviceTier, "mixed");
});

test("duplicate input IDs are rejected before a provider call", async () => {
  let calls = 0;
  await assert.rejects(analyzeViewBatches([views[0], views[0]], true, async (batch) => {
    calls++;
    return resultFor(batch);
  }), /unique names/);
  assert.equal(calls, 0);
});

// Load the real Next route with TypeScript's compiler, resolving its @/ imports
// locally. SDKs remain real; only their HTTP transport is mocked below.
function loadRoute() {
  const filename = fileURLToPath(new URL("../src/app/api/analyze/route.ts", import.meta.url));
  const localRequire = createRequire(filename);
  const compiled = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const route = { exports: {} };
  const requireLocal = (name) => localRequire(name.startsWith("@/")
    ? fileURLToPath(new URL(`../src/${name.slice(2)}.ts`, import.meta.url)) : name);
  new Function("require", "module", "exports", compiled)(requireLocal, route, route.exports);
  return route.exports.POST;
}

test("real route keeps prompt/settings, maps outlines, and totals eight independent receipts", async (t) => {
  const settings = {
    OPENROUTER_API_KEY: "offline-test-key",
    SCANBODY_OUTLINE_POINTS: "40",
    SCANBODY_OUTLINE_MIN_POINTS: "16",
  };
  const previousSettings = Object.fromEntries(Object.keys(settings).map((key) => [key, process.env[key]]));
  Object.assign(process.env, settings);
  t.after(() => {
    for (const [key, value] of Object.entries(previousSettings)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });
  const POST = loadRoute();
  const requests = [];
  t.mock.method(globalThis, "fetch", async (url, init) => {
    assert.equal(String(url), "https://openrouter.ai/api/v1/chat/completions");
    const body = JSON.parse(init.body);
    requests.push(body);
    const content = body.messages[1].content;
    const ids = content.filter((item) => item.type === "text" && item.text.startsWith("VIEW_ID: "))
      .map((item) => item.text.slice("VIEW_ID: ".length));
    const tier = body.service_tier ?? "default";
    return new Response(JSON.stringify({
      id: `offline-${requests.length}`, model: body.model, object: "chat.completion", created: 1,
      service_tier: tier,
      choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: JSON.stringify({
        views: ids.map((view) => {
          // Simulate a short simple outline and a longer detailed outline.
          const count = Number(view.slice("view-".length)) % 2 === 0 ? 16 : 40;
          const outline = Array.from({ length: count }, (_, index) => {
            const angle = index * Math.PI * 2 / count;
            return `${Math.round(100 + 30 * Math.cos(angle))},${Math.round(100 + 30 * Math.sin(angle))}`;
          }).join(" ");
          return { view, detections: [{ label: "scanbody", outline, confidence: 0.9 }] };
        }),
      }) } }],
      usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120,
        prompt_tokens_details: { cached_tokens: 10 }, cost: 0.0123 },
    }), { headers: { "Content-Type": "application/json" } });
  });
  async function analyze(parallelViews, run = { type: "baseline", useOpenRouter: true }) {
    const response = await POST(new Request("http://offline/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageSize: 1024, saveFiles: false, parallelViews, run, views }),
    }));
    const result = await response.json();
    assert.equal(response.status, 200, result.error);
    return result;
  }
  const grouped = await analyze(false);
  const groupedRequest = requests[0];
  assert.match(groupedRequest.messages[0].content, /between 16 and 40/);
  assert.match(groupedRequest.messages[1].content[0].text, /between 16 and 40/);
  assert.match(JSON.stringify(groupedRequest.response_format), /between 16 and 40/);
  assert.doesNotMatch(JSON.stringify(groupedRequest.messages), /exactly 40/);
  const parallel = await analyze(true);
  assert.equal(requests.length, 9);
  assert.deepEqual(parallel.views, grouped.views);
  assert.deepEqual(parallel.views.map((view) => view.detections[0].outline.length), [16, 40, 16, 40, 16, 40, 16, 40]);
  assert.equal(parallel.meta.outlineMinPointsPerScanbody, 16);
  assert.equal(parallel.meta.outlineMaxPointsPerScanbody, 40);
  for (let index = 0; index < 8; index++) {
    const body = requests[index + 1];
    assert.deepEqual(body.messages[0], groupedRequest.messages[0]);
    assert.deepEqual(body.reasoning, { effort: "medium" });
    assert.equal(body.model, "openai/gpt-6.1-sol");
    assert.deepEqual(body.response_format, groupedRequest.response_format);
    const images = body.messages[1].content.filter((item) => item.type === "image_url");
    assert.equal(images.length, 1);
    assert.equal(images[0].image_url.url, views[index].image);
    assert.equal(images[0].image_url.detail, "high");
  }
  assert.equal(parallel.meta.inputTokens, 800);
  assert.equal(parallel.meta.outputTokens, 160);
  assert.equal(parallel.meta.cachedInputTokens, 80);
  assert(Math.abs(parallel.meta.estimatedCostUsd - 0.0984) < 1e-12);
  assert.equal(parallel.meta.reportedCostUsd, parallel.meta.estimatedCostUsd);
  assert.equal(parallel.meta.viewRequestCount, 8);
  assert.equal(parallel.meta.parallelViews, true);
  assert.match(parallel.meta.modelLabel, /Parallel$/);
  assert.equal(grouped.meta.viewRequestCount, 1);
  assert.equal(grouped.meta.estimatedCostUsd, 0.0123);
  // Parallelism is independent of the Fast/Ultrafast tier and never drops it.
  await analyze(true, { type: "comparison", model: "openrouter-gpt-6.1-sol", reasoningEffort: "high", processingSpeed: "ultrafast" });
  for (const body of requests.slice(9)) {
    assert.equal(body.service_tier, "ultrafast");
    assert.deepEqual(body.provider.only, ["openai/ultrafast"]);
    assert.equal(body.provider.allow_fallbacks, false);
    assert.deepEqual(body.reasoning, { effort: "high" });
  }
  // Equal bounds restore a fixed point count without contradictory instructions.
  process.env.SCANBODY_OUTLINE_MIN_POINTS = "40";
  const fixed = await analyze(false);
  assert.match(requests.at(-1).messages[0].content, /exactly 40/);
  assert.doesNotMatch(requests.at(-1).messages[0].content, /fewest points|do not pad/);
  assert.equal(fixed.meta.outlineMinPointsPerScanbody, 40);
  // Reject invalid settings before any potentially paid provider request.
  process.env.SCANBODY_OUTLINE_MIN_POINTS = "41";
  const callsBefore = requests.length;
  const invalid = await POST(new Request("http://offline/api/analyze", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageSize: 1024, saveFiles: false, parallelViews: true,
      run: { type: "baseline", useOpenRouter: true }, views }),
  }));
  assert.equal(invalid.status, 400);
  assert.match((await invalid.json()).error, /SCANBODY_OUTLINE_MIN_POINTS/);
  assert.equal(requests.length, callsBefore);
});
