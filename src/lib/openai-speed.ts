import type { ProcessingSpeed } from "./ai-models";

export type OpenAiSpeedReceipt = { serviceTier: string | null };

/** Explicit Standard also overrides an account whose default tier is Fast. */
export function getOpenAiServiceTier(speed: ProcessingSpeed) {
  return speed === "standard" ? "default" : speed;
}

/** Read the native Responses receipt before LangChain normalizes the response. */
export function createOpenAiSpeedFetch(
  receipt: OpenAiSpeedReceipt,
  requiredSpeed: ProcessingSpeed,
  transport: typeof fetch = fetch,
): typeof fetch {
  return async (input, init) => {
    const response = await transport(input, init);
    if (!response.ok) return response;
    const body = await response.clone().json();
    const tier = typeof body.service_tier === "string" ? body.service_tier : null;
    const confirmed = requiredSpeed === "fast"
      ? tier === "fast" || tier === "priority" // Older models report Priority.
      : tier === "ultrafast";
    if (requiredSpeed !== "standard" && !confirmed) {
      // Return an SDK-readable error; never label a downgraded run as Fast.
      return new Response(JSON.stringify({ error: {
        message: `OpenAI did not confirm ${requiredSpeed} processing (reported ${tier ?? "no tier"}). The result was rejected; no slower-tier retry was made.`,
      } }), { status: 400, headers: { "Content-Type": "application/json" } });
    }
    receipt.serviceTier = tier;
    return response;
  };
}
