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

test("Hydrogen sends images at every documented thinking level and includes thinking in its cost", async (t) => {
  const previousKey = process.env.GOOGLE_API_KEY;
  process.env.GOOGLE_API_KEY = "offline-google-key";
  t.after(() => {
    if (previousKey === undefined) delete process.env.GOOGLE_API_KEY;
    else process.env.GOOGLE_API_KEY = previousKey;
  });
  const POST = loadRoute();
  const requests = [];
  t.mock.method(globalThis, "fetch", async (url, init) => {
    assert.match(String(url), /^https:\/\/generativelanguage\.googleapis\.com\/v1beta\/models\/gemini-(4-hydrogen|4-argon|3\.8-flash):generateContent$/);
    const body = JSON.parse(init.body);
    const parts = body.contents.flatMap((content) => content.parts);
    const ids = parts.filter((part) => part.text?.startsWith("VIEW_ID: ")).map((part) => part.text.slice(9));
    const usage = { promptTokenCount: ids.length * 100, cachedContentTokenCount: ids.length * 10,
      candidatesTokenCount: ids.length * 20, thoughtsTokenCount: ids.length * 30, totalTokenCount: ids.length * 150 };
    requests.push({ body, usage });
    return new Response(JSON.stringify({
      candidates: [{ index: 0, finishReason: "STOP", content: { role: "model", parts: [{ text: JSON.stringify({
        views: ids.map((view) => ({ view, detections: [{ label: "scanbody", outline: "10,10 20,10 20,20", confidence: 0.9 }] })),
      }) }] } }], usageMetadata: usage,
    }), { headers: { "Content-Type": "application/json" } });
  });
  const call = async (model, effort, parallelViews) => {
    const response = await POST(new Request("http://offline/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageSize: 768, jpegQuality: 80, outlineMinPoints: 3, outlineMaxPoints: 14,
        saveFiles: false, parallelViews,
        run: { type: "standalone", model, reasoningEffort: effort }, views }),
    }));
    return { status: response.status, result: await response.json() };
  };
  let referencePrompt;
  for (const effort of ["low", "medium", "high", "xhigh"]) {
    for (const parallelViews of [false, true]) {
      const before = requests.length;
      const { status, result } = await call("gemini-4-hydrogen", effort, parallelViews);
      assert.equal(status, 200, result.error);
      assert.equal(result.meta.model, "gemini-4-hydrogen");
      assert.equal(result.meta.provider, "google");
      assert.equal(result.meta.reasoningEffort, effort);
      assert.equal(result.meta.imageSize, 768);
      assert.equal(result.meta.jpegQuality, 80);
      assert.equal(result.meta.outlineMaxPointsPerScanbody, 14);
      assert.equal(result.meta.viewRequestCount, parallelViews ? 8 : 1);
      assert.deepEqual(result.views.map((view) => view.view), views.map((view) => view.name));
      const sent = requests.slice(before);
      assert.equal(sent.length, parallelViews ? 8 : 1);
      for (const { body } of sent) {
        assert.equal(body.generationConfig.thinkingConfig.thinkingLevel, effort.toUpperCase());
        assert.equal(body.generationConfig.maxOutputTokens, 65536);
        assert.equal(body.generationConfig.responseMimeType, "application/json");
        assert(body.generationConfig.responseSchema);
        assert.doesNotMatch(JSON.stringify(body.generationConfig.responseSchema), /"const":/);
        const parts = body.contents.flatMap((content) => content.parts);
        const images = parts.filter((part) => part.inlineData);
        assert.equal(images.length, parallelViews ? 1 : 8, "Gemini 4 must receive images, not just the text prompt");
        assert(images.every((part) => part.inlineData.mimeType === "image/jpeg"));
        assert.match(parts[0].text, /768 by 768/);
        assert.match(parts[0].text, /between 3 and 14/);
        referencePrompt ??= body.systemInstruction;
        assert.deepEqual(body.systemInstruction, referencePrompt);
      }
      assert.equal(result.meta.inputTokens, 800);
      assert.equal(result.meta.cachedInputTokens, 80);
      assert.equal(result.meta.outputTokens, 400, "160 visible + 240 thinking tokens are all billable output");
      assert(Math.abs(result.meta.estimatedCostUsd - (720 * 1.5 + 80 * 0.075 + 400 * 7.5) / 1_000_000) < 1e-12);
      if (effort === "xhigh") assert.match(result.meta.modelLabel, /XHigh/);
    }
  }
  // Existing Google models retain their existing levels and integration path.
  for (const model of ["gemini-4-argon", "gemini-3.8-flash"]) {
    const { status, result } = await call(model, "medium", false);
    assert.equal(status, 200, result.error);
    assert.equal(result.meta.outputTokens, 160);
    assert.equal(requests.at(-1).body.generationConfig.thinkingConfig.thinkingLevel, "MEDIUM");
  }
  const beforeInvalid = requests.length;
  for (const [model, effort] of [["gemini-4-hydrogen", "minimal"], ["gemini-4-argon", "xhigh"], ["gpt-6.1-terra", "xhigh"]]) {
    assert.equal((await call(model, effort, true)).status, 400);
  }
  assert.equal(requests.length, beforeInvalid, "unsupported thinking levels must not reach any provider");
});

test("all eight start together; reverse completion preserves view order, individual times and totals", async (t) => {
  let clockNow = 0;
  t.mock.method(performance, "now", () => clockNow);
  const finish = [];
  const pending = analyzeViewBatches(views, true, (batch) => {
    assert.equal(batch.length, 1);
    const index = views.findIndex((view) => view === batch[0]);
    return new Promise((resolve) => { finish[index] = () => resolve(resultFor(batch)); });
  });
  assert.equal(finish.length, 8, "every call must start before any completes");
  for (const complete of finish.toReversed()) {
    clockNow += 10;
    complete();
    await Promise.resolve();
  }
  const result = await pending;
  assert.deepEqual(result.parsed.views.map((view) => view.view), views.map((view) => view.name));
  assert.equal(result.viewRequestCount, 8);
  assert.deepEqual(result.viewTimings, views.map((view, index) => ({ view: view.name, elapsedMs: (8 - index) * 10 })));
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
  assert.equal(result.viewTimings, null, "a shared eight-image request has no individual image times");
  const single = await analyzeViewBatches(views.slice(0, 1), false, async (batch) => resultFor(batch));
  assert.equal(single.viewTimings.length, 1);
  assert.equal(single.viewTimings[0].view, views[0].name);
  assert(single.viewTimings[0].elapsedMs >= 0);
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
          const prompt = body.messages[0].content;
          const range = /between (\d+) and (\d+)/.exec(prompt);
          const fixedCount = /exactly (\d+)/.exec(prompt);
          const count = range
            ? Number(range[Number(view.slice("view-".length)) % 2 === 0 ? 1 : 2])
            : Number(fixedCount[1]);
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
  async function analyze(parallelViews, run = { type: "baseline", useOpenRouter: true }, imageSettings = {}) {
    const response = await POST(new Request("http://offline/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageSize: 1024, ...imageSettings, saveFiles: false, parallelViews, run, views }),
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
  assert.deepEqual(parallel.meta.viewTimings.map((timing) => timing.view), views.map((view) => view.name));
  assert(parallel.meta.viewTimings.every((timing) => Number.isFinite(timing.elapsedMs) && timing.elapsedMs >= 0));
  assert.match(parallel.meta.modelLabel, /Parallel$/);
  assert.equal(grouped.meta.viewRequestCount, 1);
  assert.equal(grouped.meta.viewTimings, null);
  assert.equal(grouped.meta.estimatedCostUsd, 0.0123);
  // Parallelism is independent of the Fast/Ultrafast tier and never drops it.
  await analyze(true, { type: "comparison", model: "openrouter-gpt-6.1-sol", reasoningEffort: "high", processingSpeed: "ultrafast" });
  for (const body of requests.slice(9)) {
    assert.equal(body.service_tier, "ultrafast");
    assert.deepEqual(body.provider.only, ["openai/ultrafast"]);
    assert.equal(body.provider.allow_fallbacks, false);
    assert.deepEqual(body.reasoning, { effort: "high" });
  }
  // A selected model can use exactly the same eight-view path without a baseline.
  const standalone = await analyze(true, {
    type: "standalone", model: "openrouter-gpt-6.1-terra", reasoningEffort: "low", processingSpeed: "standard",
  });
  assert.equal(standalone.meta.model, "openrouter-gpt-6.1-terra");
  assert.equal(standalone.meta.viewRequestCount, 8);
  for (const request of requests.slice(-8)) {
    assert.equal(request.model, "openai/gpt-6.1-terra");
    assert.deepEqual(request.reasoning, { effort: "low" });
  }
  const reduced = await analyze(true, { type: "baseline", useOpenRouter: true }, { imageSize: 768, jpegQuality: 70 });
  assert.equal(reduced.meta.imageSize, 768);
  assert.equal(reduced.meta.jpegQuality, 70);
  assert.equal(reduced.meta.viewRequestCount, 8);
  for (const request of requests.slice(-8)) {
    assert.match(request.messages[1].content[0].text, /Every image is 768 by 768/);
    assert.equal(request.messages[1].content.filter((item) => item.type === "image_url").length, 1);
  }
  // Equal bounds restore a fixed point count without contradictory instructions.
  process.env.SCANBODY_OUTLINE_MIN_POINTS = "40";
  const fixed = await analyze(false);
  assert.match(requests.at(-1).messages[0].content, /exactly 40/);
  assert.doesNotMatch(requests.at(-1).messages[0].content, /fewest points|do not pad/);
  assert.equal(fixed.meta.outlineMinPointsPerScanbody, 40);
  // UI settings override both environment defaults in every parallel request.
  const custom = await analyze(true, { type: "standalone", model: "openrouter-gpt-6.1-terra", reasoningEffort: "low" },
    { outlineMinPoints: 3, outlineMaxPoints: 8 });
  assert.equal(custom.meta.outlineMinPointsPerScanbody, 3);
  assert.equal(custom.meta.outlineMaxPointsPerScanbody, 8);
  assert.deepEqual(custom.views.map((view) => view.detections[0].outline.length), [3, 8, 3, 8, 3, 8, 3, 8]);
  for (const request of requests.slice(-8)) {
    assert.match(request.messages[0].content, /between 3 and 8/);
    assert.match(request.messages[1].content[0].text, /between 3 and 8/);
    assert.match(JSON.stringify(request.response_format), /between 3 and 8/);
  }
  const fixedUi = await analyze(false, { type: "baseline", useOpenRouter: true }, { outlineMinPoints: 8, outlineMaxPoints: 8 });
  assert.match(requests.at(-1).messages[0].content, /exactly 8/);
  assert.doesNotMatch(requests.at(-1).messages[0].content, /fewest points|do not pad/);
  assert.equal(fixedUi.meta.outlineMaxPointsPerScanbody, 8);
  const beforeInvalidBounds = requests.length;
  for (const bounds of [{ outlineMinPoints: 16, outlineMaxPoints: 8 }, { outlineMinPoints: 2, outlineMaxPoints: 40 },
    { outlineMinPoints: 3, outlineMaxPoints: 10001 }, { outlineMinPoints: 3.5, outlineMaxPoints: 40 }]) {
    const invalidBounds = await POST(new Request("http://offline/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageSize: 1024, saveFiles: false, run: { type: "baseline", useOpenRouter: true }, views, ...bounds }),
    }));
    assert.equal(invalidBounds.status, 400);
  }
  assert.equal(requests.length, beforeInvalidBounds);
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
