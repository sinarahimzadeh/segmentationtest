import type { ProcessingSpeed } from "./ai-models";

export type OpenRouterReceipt = {
  serviceTier: string | null;
  reportedCostUsd: number | null;
};

/** Pin the actual endpoint: service_tier alone permits slower fallbacks. */
export function getOpenRouterSpeedRouting(speed: ProcessingSpeed) {
  if (speed === "standard") return { service_tier: "default" };
  return {
    service_tier: speed === "fast" ? "priority" : "ultrafast",
    provider: {
      only: [speed === "fast" ? "openai/fast" : "openai/ultrafast"],
      allow_fallbacks: false,
      require_parameters: true,
    },
  };
}

/** Capture billing/tier before LangChain normalizes away provider metadata. */
export function createOpenRouterReceiptFetch(
  receipt: OpenRouterReceipt,
  requiredSpeed?: ProcessingSpeed,
  transport: typeof fetch = fetch,
): typeof fetch {
  return async (input, init) => {
    const response = await transport(input, init);
    if (!response.ok) return response;
    const body = await response.clone().json();
    // OpenRouter can return an error object even with an HTTP 200 response.
    if (body.error) return new Response(JSON.stringify(body), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
    const tier = typeof body.service_tier === "string" ? body.service_tier : null;
    if (requiredSpeed && requiredSpeed !== "standard") {
      const expectedTier = requiredSpeed === "fast" ? "priority" : "ultrafast";
      if (tier !== expectedTier && !(requiredSpeed === "fast" && tier === "fast")) {
        // An HTTP error preserves the explanation through the OpenAI SDK;
        // throwing here would be wrapped as an unhelpful "Connection error".
        return new Response(JSON.stringify({ error: {
          message: `OpenRouter did not confirm ${requiredSpeed} processing (reported ${tier ?? "no tier"}). The result was rejected; no slower-tier retry was made.`,
        } }), { status: 400, headers: { "Content-Type": "application/json" } });
      }
    }
    receipt.serviceTier = tier;
    const cost = body.usage?.cost;
    receipt.reportedCostUsd = typeof cost === "number" && Number.isFinite(cost) && cost >= 0 ? cost : null;
    return response;
  };
}
