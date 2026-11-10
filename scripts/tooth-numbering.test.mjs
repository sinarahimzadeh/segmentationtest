import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import ts from "typescript";
import { isPermanentFdi, normalizeToothAssignments } from "../src/lib/tooth-numbering.ts";
import { chooseToothNumberView } from "../src/lib/tooth-number-view.ts";

const assignment = (detectionId, toothFdi, confidence = 0.9) => ({ detectionId, toothFdi, confidence, reason: "Anatomical estimate" });
const target = (detectionId, view = "top") => ({ detectionId, view });

test("arch-facing choice prefers more visible posts, allows overrides, and handles reversed orientation", () => {
  const detection = (confidence) => ({ label: "scanbody", outline: [[1, 1], [2, 1], [2, 2]], confidence });
  const detected = [
    { view: "top", detections: [detection(0.99)] },
    { view: "bottom", detections: [detection(0.9), detection(0.9)] },
  ];
  assert.equal(chooseToothNumberView(["top", "bottom"], detected, "auto"), "bottom");
  assert.equal(chooseToothNumberView(["top", "bottom"], detected, "top"), "top");
  assert.equal(chooseToothNumberView(["bottom"], detected, "auto"), "bottom");
  assert.equal(chooseToothNumberView(["bottom"], detected, "top"), null);
  assert.equal(chooseToothNumberView(["front"], detected, "auto"), null);
  assert.equal(chooseToothNumberView(["top", "bottom"], [
    { view: "top", detections: [detection(0.6)] },
    { view: "bottom", detections: [detection(0.95)] },
  ], "auto"), "bottom");
});

test("FDI accepts only permanent two-digit tooth positions", () => {
  for (const number of [11, 18, 21, 28, 31, 38, 41, 48]) assert(isPermanentFdi(number));
  for (const number of [0, 1, 19, 20, 29, 49, 51, 99, 16.5]) assert(!isPermanentFdi(number));
});

test("numbering preserves cross-view IDs, keeps omissions uncertain, and rejects wrong jaws", () => {
  const result = normalizeToothAssignments({ jaw: "upper", notes: "Estimate", assignments: [
    assignment("top:1", 16), assignment("front:1", 16), assignment("top:2", 46),
  ] }, [target("top:1"), target("front:1", "front"), target("top:2"), target("top:3")], "upper");
  assert.deepEqual(result.assignments.map((item) => item.toothFdi), [16, 16, null, null]);
  assert.match(result.assignments[2].reason, /invalid or did not match/);
  assert.match(result.assignments[3].reason, /did not identify/);
});

test("conflicting same-view numbers stay uncertain and never alter the raw response", () => {
  const raw = { jaw: "upper", notes: "", assignments: [assignment("top:1", 16), assignment("top:2", 16)] };
  const result = normalizeToothAssignments(raw, [target("top:1"), target("top:2")], "auto");
  assert.deepEqual(result.assignments.map((item) => item.toothFdi), [null, null]);
  assert.equal(raw.assignments[0].toothFdi, 16);
  assert.throws(() => normalizeToothAssignments({ ...raw, assignments: [assignment("invented", 16)] }, [target("top:1")], "auto"), /unknown detection ID/);
});

function loadRoute(files) {
  const filename = fileURLToPath(new URL("../src/app/api/tooth-numbers/route.ts", import.meta.url));
  const localRequire = createRequire(filename);
  const compiled = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const route = { exports: {} };
  const requireLocal = (name) => name === "node:fs/promises" ? {
    mkdir: async () => {}, writeFile: async (path, value) => { files.set(path, value); },
  } : localRequire(name.startsWith("@/")
    ? fileURLToPath(new URL(`../src/${name.slice(2)}.ts`, import.meta.url)) : name);
  new Function("require", "module", "exports", compiled)(requireLocal, route, route.exports);
  return route.exports.POST;
}

test("separate real route sends only the selected arch view and its matching outlines", async (t) => {
  const previousKey = process.env.OPENROUTER_API_KEY;
  process.env.OPENROUTER_API_KEY = "offline-test-key";
  t.after(() => {
    if (previousKey === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = previousKey;
  });
  const files = new Map();
  const POST = loadRoute(files);
  const requests = [];
  t.mock.method(globalThis, "fetch", async (url, init) => {
    assert.equal(String(url), "https://openrouter.ai/api/v1/chat/completions");
    const body = JSON.parse(init.body);
    requests.push(body);
    assert.equal(body.model, "openai/gpt-6.1-sol");
    assert.deepEqual(body.reasoning, { effort: "medium" });
    assert.equal(body.response_format.json_schema.name, "scanbody_tooth_numbers");
    const images = body.messages[1].content.filter((item) => item.type === "image_url");
    assert.equal(images.length, 1);
    assert(images.every((item) => item.image_url.detail === "high"));
    const texts = body.messages[1].content.filter((item) => item.type === "text").map((item) => item.text).join("\n");
    const selectedName = /VIEW_ID: (top|bottom)/.exec(texts)?.[1];
    assert(selectedName);
    assert.equal(images[0].image_url.url, `data:image/jpeg;base64,${selectedName}`);
    assert(!texts.includes("front:1"), "another view's outlines must never be sent");
    assert(!texts.includes(`${selectedName === "top" ? "bottom" : "top"}:1`));
    assert.match(body.messages[0].content, /Do not detect new scanbodies/);
    return new Response(JSON.stringify({
      id: "offline", model: body.model, object: "chat.completion", created: 1, service_tier: "default",
      choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: JSON.stringify({
        jaw: "upper", assignments: [assignment(`${selectedName}:1`, 16)], notes: "Some positions are uncertain.",
      }) } }],
      usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120, cost: 0.0123 },
    }), { headers: { "Content-Type": "application/json" } });
  });
  const views = ["top", "bottom", "front", "back", "left", "right", "front-left", "front-right"].map((name) => ({
    name, image: `data:image/jpeg;base64,${name}`,
    detections: ["top", "bottom", "front"].includes(name)
      ? [{ detectionId: `${name}:1`, outline: [[10, 10], [20, 10], [20, 20]] }] : [],
  }));
  const body = { meshKey: "a".repeat(64), meshName: "example.stl", imageSize: 1024,
    useOpenRouter: true, saveFiles: false, jawHint: "upper", rightSideHint: "image-left", views };
  async function call(overrides = {}) {
    return POST(new Request("http://offline/api/tooth-numbers", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...body, ...overrides }) }));
  }
  const response = await call();
  const result = await response.json();
  assert.equal(response.status, 200, result.error);
  assert.equal(requests.length, 1);
  assert.equal(result.assignments[0].toothFdi, 16);
  assert.equal(result.assignments.length, 1);
  assert.equal(result.meta.imageCount, 1);
  assert.deepEqual(result.meta.viewNames, ["top"]);
  assert.equal(result.meta.estimatedCostUsd, 0.0123);
  assert.equal(files.size, 0, "saving off must never write a file");
  const saved = await (await call({ saveFiles: true })).json();
  assert.equal(files.size, 1);
  assert.match(saved.savedPath, /[\\/]tooth-numbers[\\/]/);
  const data = JSON.parse(files.get(saved.savedPath));
  assert.equal(data.rawResponse.assignments[0].toothFdi, 16);
  assert.deepEqual(data.targets.map((item) => item.detectionId), ["top:1"]);
  assert.equal(data.result.savedPath, saved.savedPath);
  assert(!files.get(saved.savedPath).includes("data:image/"));
  const bottomResponse = await call({ referenceView: "bottom" });
  const bottom = await bottomResponse.json();
  assert.equal(bottomResponse.status, 200, bottom.error);
  assert.deepEqual(bottom.meta.viewNames, ["bottom"]);
  assert.equal(bottom.meta.imageCount, 1);
  assert.equal(bottom.assignments[0].detectionId, "bottom:1");
  const before = requests.length;
  assert.equal((await call({ meshKey: "../elsewhere" })).status, 400);
  assert.equal((await call({ views: views.map((view) => ({ ...view, detections: [] })) })).status, 400);
  assert.equal((await call({ views: views.filter((view) => view.name !== "top") })).status, 400);
  assert.equal((await call({ views: views.map((view) => view.name === "top" ? { ...view, detections: [] } : view) })).status, 400);
  assert.equal((await call({ referenceView: "bottom", views: views.filter((view) => view.name !== "bottom") })).status, 400);
  assert.equal((await call({ referenceView: "front" })).status, 400);
  assert.equal(requests.length, before, "invalid requests must not reach the API");
});
