import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import type { ContentBlockParam } from "@anthropic-ai/sdk/resources/messages";
import { z } from "zod";
import type { AiModelId, ReasoningEffort } from "./ai-models";

type ClaudeModelId = Extract<AiModelId, `claude-${string}`>;
type ClaudeEffort = Extract<ReasoningEffort, "low" | "medium" | "high">;

/** Direct Anthropic SDK experiment. This module does not import LangChain. */
export async function analyzeClaudeDirect<Schema extends z.ZodType>({
  apiKey,
  model,
  effort,
  systemPrompt,
  imageInstruction,
  views,
  detectionSchema,
}: {
  apiKey: string;
  model: ClaudeModelId;
  effort: ClaudeEffort;
  systemPrompt: string;
  imageInstruction: string;
  views: { name: string; image: string }[];
  detectionSchema: Schema;
}): Promise<{
  parsed: z.infer<Schema>;
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
}> {
  const content: ContentBlockParam[] = [
    { type: "text", text: imageInstruction },
  ];

  for (const view of views) {
    // The browser captures base64 JPEGs. Preserve each image's bytes and order.
    const prefix = /^data:(image\/(?:jpeg|png|gif|webp));base64,/i.exec(view.image);
    if (!prefix) throw new Error(`Unsupported image format in ${view.name}.`);
    content.push({ type: "text", text: `VIEW_ID: ${view.name}` });
    content.push({
      type: "image",
      source: {
        type: "base64",
        media_type: prefix[1].toLowerCase() as
          | "image/jpeg"
          | "image/png"
          | "image/gif"
          | "image/webp",
        data: view.image.slice(prefix[0].length),
      },
    });
  }

  const client = new Anthropic({ apiKey, maxRetries: 1 });
  // Streaming is internal: the app receives one completed JSON response.
  // Anthropic requires max_tokens; 128K is the model's permitted ceiling,
  // including thinking, not a target or a minimum billed amount.
  const message = await client.messages.stream({
    model,
    max_tokens: 128_000,
    thinking: { type: "adaptive" },
    output_config: {
      effort,
      format: zodOutputFormat(detectionSchema),
    },
    system: systemPrompt,
    messages: [{ role: "user", content }],
  }).finalMessage();

  if (message.stop_reason !== "end_turn" || !message.parsed_output) {
    throw new Error(`Claude stopped without complete JSON (${message.stop_reason}).`);
  }

  const cacheRead = message.usage.cache_read_input_tokens ?? 0;
  const cacheCreation = message.usage.cache_creation_input_tokens ?? 0;
  return {
    parsed: message.parsed_output,
    // Match the total input-token convention used by LangChain's Claude path.
    inputTokens: message.usage.input_tokens + cacheRead + cacheCreation,
    cachedInputTokens: cacheRead,
    outputTokens: message.usage.output_tokens,
  };
}
