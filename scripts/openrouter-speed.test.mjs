import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";
import {
  getProcessingSpeeds, getSpeedPriceMultiplier, getModelRunLabel, estimatePaidCostUsd,
} from "../src/lib/ai-models.ts";
import {
  createOpenRouterReceiptFetch, getOpenRouterSpeedRouting,
} from "../src/lib/openrouter-speed.ts";
import { createOpenAiSpeedFetch, getOpenAiServiceTier } from "../src/lib/openai-speed.ts";

// All transport is mocked. These tests never contact any provider or use real keys.
const schema = z.object({ views: z.array(z.object({ view: z.string() })) });
const messages = [new SystemMessage("Find scanbodies"), new HumanMessage({
  content: Array.from({ length: 8 }, (_, index) => [
    { type: "text", text: `VIEW_ID: view-${index}` },
    { type: "image_url", image_url: { url: `data:image/jpeg;base64,${index}`, detail: "high" } },
  ]).flat(),
})];

function mockCompletion(tier) {
  return new Response(JSON.stringify({
    id: "offline-test", model: "openai/gpt-6.1-sol", object: "chat.completion", created: 1,
    service_tier: tier,
    choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: '{"views":[{"view":"top"}]}' } }],
    usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120, cost: 0.00123 },
  }), { headers: { "Content-Type": "application/json" } });
}

function detector(speed, receipt, transport) {
  return new ChatOpenAI({
    model: "openai/gpt-6.1-sol", apiKey: "offline-test-key", maxRetries: 0,
    configuration: { baseURL: "https://openrouter.ai/api/v1", fetch: createOpenRouterReceiptFetch(receipt, speed, transport) },
    modelKwargs: { ...getOpenRouterSpeedRouting(speed), reasoning: { effort: "medium" } },
  }).withStructuredOutput(schema, { name: "scanbody_roi_detection", includeRaw: true });
}

for (const speed of ["fast", "ultrafast"]) {
  test(`${speed} sends all images and requires its exact endpoint`, async () => {
    const receipt = { serviceTier: null, reportedCostUsd: null };
    const tier = speed === "fast" ? "priority" : "ultrafast";
    let calls = 0;
    const result = await detector(speed, receipt, async (_url, options) => {
      calls++;
      const body = JSON.parse(options.body);
      assert.equal(body.service_tier, tier);
      assert.deepEqual(body.provider.only, [`openai/${speed}`]);
      assert.equal(body.provider.allow_fallbacks, false);
      assert.equal(body.provider.require_parameters, true);
      assert.equal(body.reasoning.effort, "medium");
      const images = body.messages[1].content.filter((part) => part.type === "image_url");
      assert.equal(images.length, 8);
      assert(images.every((image) => image.image_url.detail === "high"));
      assert.equal(body.response_format.type, "json_schema");
      return mockCompletion(tier);
    }).invoke(messages);
    assert.equal(calls, 1);
    assert.deepEqual(result.parsed, { views: [{ view: "top" }] });
    assert.equal(receipt.serviceTier, tier);
    assert.equal(receipt.reportedCostUsd, 0.00123);
  });
}

test("a downgraded or unconfirmed tier is rejected without retry", async () => {
  for (const [speed, returnedTier] of [["fast", "default"], ["ultrafast", "priority"], ["ultrafast", null]]) {
    let calls = 0;
    const receipt = { serviceTier: null, reportedCostUsd: null };
    await assert.rejects(detector(speed, receipt, async () => {
      calls++;
      return mockCompletion(returnedTier);
    }).invoke(messages), /did not confirm/);
    assert.equal(calls, 1);
    assert.equal(receipt.reportedCostUsd, null);
  }
});

test("unavailable tier returns an error without a normal-speed retry", async () => {
  let calls = 0;
  await assert.rejects(detector("ultrafast", {}, async () => {
    calls++;
    return new Response('{"error":{"message":"No endpoints available"}}', { status: 503 });
  }).invoke(messages), /No endpoints available/);
  assert.equal(calls, 1);
});

test("only supported models expose tiers; saved labels distinguish speed", () => {
  for (const model of ["gpt-6.1-sol", "openrouter-gpt-6.1-sol"]) {
    assert.deepEqual(getProcessingSpeeds(model), ["standard", "fast", "ultrafast"]);
  }
  for (const model of ["gpt-6.1-terra", "gpt-6-sol", "gpt-6-luna", "gpt-5.5", "openrouter-gpt-6.1-terra", "openrouter-gpt-6-sol", "openrouter-gpt-6-luna", "openrouter-gpt-5.5"]) {
    assert.deepEqual(getProcessingSpeeds(model), ["standard", "fast"]);
    assert.throws(() => getSpeedPriceMultiplier(model, "ultrafast"));
  }
  for (const model of ["gemini-3.8-flash", "openrouter-qwen-3-8-flash"]) {
    assert.deepEqual(getProcessingSpeeds(model), ["standard"]);
    assert.throws(() => getSpeedPriceMultiplier(model, "fast"));
  }
  assert.match(getModelRunLabel("openrouter-gpt-6.1-sol", "medium", "ultrafast"), /Medium · Ultrafast$/);
});

test("cost fallback uses each model's tier prices", () => {
  for (const [model, multiplier] of [["gpt-6.1-sol", 2], ["gpt-6.1-terra", 2], ["gpt-6-sol", 2], ["gpt-6-luna", 2], ["gpt-5.5", 2.5], ["openrouter-gpt-6.1-sol", 2], ["openrouter-gpt-5.5", 2.5]]) {
    const usage = { modelId: model, inputTokens: 10000, cachedInputTokens: 1000, outputTokens: 6000 };
    assert.equal(estimatePaidCostUsd({ ...usage, processingSpeed: "fast" }), estimatePaidCostUsd(usage) * multiplier);
  }
  assert.equal(getSpeedPriceMultiplier("openrouter-gpt-6.1-sol", "ultrafast"), 6);
  assert.equal(getSpeedPriceMultiplier("gpt-6.1-sol", "ultrafast"), 6);
  assert.deepEqual(getOpenRouterSpeedRouting("standard"), { service_tier: "default" });
});

test("direct OpenAI verifies Fast/Priority and Ultrafast receipts without slower retries", async () => {
  for (const [speed, tier, accepted] of [
    ["standard", "default", true], ["fast", "fast", true], ["fast", "priority", true],
    ["ultrafast", "ultrafast", true], ["fast", "default", false],
    ["ultrafast", "fast", false], ["ultrafast", null, false],
  ]) {
    let calls = 0;
    const receipt = { serviceTier: null };
    const fetchReceipt = createOpenAiSpeedFetch(receipt, speed, async () => {
      calls++;
      return new Response(JSON.stringify({ service_tier: tier }), { headers: { "Content-Type": "application/json" } });
    });
    const response = await fetchReceipt("https://api.openai.com/v1/responses");
    assert.equal(response.status, accepted ? 200 : 400);
    assert.equal(receipt.serviceTier, accepted ? tier : null);
    if (!accepted) assert.match((await response.json()).error.message, /OpenAI did not confirm/);
    assert.equal(calls, 1);
    assert.equal(getOpenAiServiceTier(speed), speed === "standard" ? "default" : speed);
  }
});
