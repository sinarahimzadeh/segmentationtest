import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOpenAI } from "@langchain/openai";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";
import { z } from "zod";
import { analyzeClaudeDirect } from "@/lib/direct-claude";
import { MistralChatModel } from "@/lib/mistral-chat";
import { ScanbodyChatXAI } from "@/lib/xai-chat";
import { analyzeViewBatches, type ViewBatchResult } from "@/lib/parallel-view-analysis";
import { createOpenRouterReceiptFetch, getOpenRouterSpeedRouting, type OpenRouterReceipt } from "@/lib/openrouter-speed";
import {
  AI_MODEL_IDS,
  PROCESSING_SPEEDS,
  getBaselineSelection,
  estimatePaidCostUsd,
  getAiModelOption,
  getModelRunLabel,
  getProcessingSpeeds,
  isAllowedComparisonSelection,
  type AiModelId,
  type AiProvider,
  type ReasoningEffort,
  type ProcessingSpeed,
} from "@/lib/ai-models";

/**
 * @langchain/google-genai 2.3.4 predates Gemini 4 and its internal multimodal
 * check only recognizes model names through `gemini-3`. Google officially
 * supports image input for Gemini 4 Argon, so extend that check locally until
 * LangChain adds Gemini 4 to its own model-name matcher.
 */
class CurrentGeminiChatModel extends ChatGoogleGenerativeAI {
  override get _isMultimodalModel() {
    return this.model.startsWith("gemini-4") || super._isMultimodalModel;
  }
}

type ClaudeModelId = Extract<AiModelId, `claude-${string}`>;
type ClaudeEffort = Extract<ReasoningEffort, "low" | "medium" | "high">;
type MistralEffort = Extract<ReasoningEffort, "none" | "low" | "medium" | "high">;

const requestSchema = z.object({
  imageSize: z.number().int().min(256).max(2048),
  saveFiles: z.boolean().optional().default(true),
  // Older clients keep grouped requests unless they explicitly opt in.
  parallelViews: z.boolean().default(false),
  // Only Claude comparisons may opt out of LangChain. Default preserves old runs.
  directClaude: z.boolean().optional().default(false),
  run: z.discriminatedUnion("type", [
    z.object({
      type: z.literal("baseline"),
      useOpenRouter: z.boolean().default(true),
    }),
    z.object({
      // Standalone runs use the same model settings without requiring a baseline.
      type: z.enum(["comparison", "standalone"]),
      model: z.enum(AI_MODEL_IDS),
      reasoningEffort: z.enum(["default", "none", "minimal", "low", "medium", "high"]),
      processingSpeed: z.enum(PROCESSING_SPEEDS).default("standard"),
    }),
  ]),
  views: z
    .array(
      z.object({
        name: z.string().min(1).max(40),
        image: z.string().startsWith("data:image/"),
      }),
    )
    .min(1)
    .max(10),
});

function resolveModelSelection(run: z.infer<typeof requestSchema>["run"]): {
  model: AiModelId;
  provider: AiProvider;
  reasoningEffort: ReasoningEffort;
  processingSpeed: ProcessingSpeed;
} {
  if (run.type === "baseline") {
    const baseline = getBaselineSelection(run.useOpenRouter);
    return {
      ...baseline,
      processingSpeed: "standard",
      provider: getAiModelOption(baseline.model).provider,
    };
  }
  if (!isAllowedComparisonSelection(run.model, run.reasoningEffort)) {
    throw new Error(
      `${run.model} does not support ${run.reasoningEffort} in this benchmark.`,
    );
  }
  if (!getProcessingSpeeds(run.model).includes(run.processingSpeed)) {
    throw new Error(`${run.model} does not offer ${run.processingSpeed} processing.`);
  }
  return {
    model: run.model,
    provider: getAiModelOption(run.model).provider,
    reasoningEffort: run.reasoningEffort,
    processingSpeed: run.processingSpeed,
  };
}

const DEFAULT_OUTLINE_POINTS_PER_SCANBODY = 40;
const DEFAULT_MIN_OUTLINE_POINTS_PER_SCANBODY = 16;
const MAX_OUTLINE_POINTS_PER_SCANBODY = 10_000;
type OutlinePointRange = { min: number; max: number };

function getOutlinePointRange(): OutlinePointRange {
  // The existing setting is now the maximum, rather than a mandatory count.
  const pointCount = Number(
    process.env.SCANBODY_OUTLINE_POINTS ??
      DEFAULT_OUTLINE_POINTS_PER_SCANBODY,
  );

  if (
    !Number.isInteger(pointCount) ||
    pointCount < 3 ||
    pointCount > MAX_OUTLINE_POINTS_PER_SCANBODY
  ) {
    throw new Error(
      `SCANBODY_OUTLINE_POINTS must be a whole number from 3 to ${MAX_OUTLINE_POINTS_PER_SCANBODY}.`,
    );
  }

  const minimum = Number(
    process.env.SCANBODY_OUTLINE_MIN_POINTS ??
      Math.min(DEFAULT_MIN_OUTLINE_POINTS_PER_SCANBODY, pointCount),
  );
  if (!Number.isInteger(minimum) || minimum < 3 || minimum > pointCount) {
    throw new Error(
      "SCANBODY_OUTLINE_MIN_POINTS must be a whole number from 3 to SCANBODY_OUTLINE_POINTS.",
    );
  }
  return { min: minimum, max: pointCount };
}

function outlinePointInstruction(range: OutlinePointRange) {
  return range.min === range.max
    ? `exactly ${range.max}`
    : `between ${range.min} and ${range.max}`;
}

function createDetectionSchema(range: OutlinePointRange) {
  return z.object({
    views: z.array(
      z.object({
        view: z
          .string()
          .describe("The exact VIEW_ID associated with this image."),
        detections: z.array(
          z.object({
            // Use a one-value enum instead of z.literal(). Zod converts a
            // literal to JSON Schema `const`, which Gemini rejects. Both
            // OpenAI and Gemini accept the equivalent single-value `enum`.
            label: z.enum(["scanbody"]),
            // One compact string avoids repeating the JSON keys "x" and "y"
            // for every point. The app parses it back into normal coordinates.
            outline: z.string().describe(
              `Use ${outlinePointInstruction(range)} ordered integer pixel pairs tracing the visible boundary.${range.min < range.max ? " Choose the count separately for each outline, using fewer for simple shapes and more for detailed shapes." : ""} Format only as "x,y x,y x,y".`,
            ),
            confidence: z
              .number()
              .describe("Confidence from 0 to 1 that this is a scanbody."),
          }),
        ),
      }),
    ),
  });
}

/**
 * Converts the compact LLM outline back into coordinates used by the existing
 * triangle mapper. The regex tolerates harmless spacing differences so one
 * formatting mistake does not waste an otherwise successful paid response.
 */
function parseCompactOutline(value: string) {
  const pairs = value.match(/-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?/g) ?? [];
  return pairs.map((pair) => {
    const [x, y] = pair.split(",").map((coordinate) => Number(coordinate.trim()));
    return [x, y] as [number, number];
  });
}

function createSystemPrompt(range: OutlinePointRange) {
  return `You inspect monochrome renders of dental intraoral STL meshes.
Your only task is to locate implant scanbodies as regions of interest.

A scanbody is a manufactured post attached to a dental implant. In a scan it
usually protrudes from gingiva and has smooth cylindrical, flat-sided,
rectangular, indexed, or cap-like manufactured geometry. It can also look like
a large circular collar or ring with a central opening, flat index, shoulder,
or stepped sides. Several repeated manufactured posts distributed around the
dental arch are strong evidence that they are scanbodies. Their shading may be
very dark or very bright depending on the camera angle.

Do not mark teeth, gingiva, crowns, empty implant holes, or random mesh
artifacts. Do not reject a manufactured registration post merely because it
resembles a healing abutment. A low-profile simple healing cap without scan
indexing should be rejected, but a repeated protruding indexed body used for
implant scanning should be marked.

IMPORTANT: A scanbody can have a central screw channel or hole. Outline the
outer silhouette of the manufactured body and ignore its internal opening. An
empty implant hole means a hole with no manufactured post around it. In an
occlusal or top view, large repeated circular or polygonal posts with central
openings distributed around the arch are the primary targets. Examine the full
image for this repeated manufactured pattern before deciding that a view is
empty.

The images are different camera views of the same mesh. Some images can show
the underside, where no scanbody is visible. Treat every view independently.
For every visible scanbody, trace its visible outer boundary using
${outlinePointInstruction(range)} ordered whole-pixel points.${range.min < range.max ? `
Choose the number
separately for each scanbody in each view. Use the fewest points that follow
the actual boundary closely. Simple silhouettes need fewer points; detailed
silhouettes need more. Preserve curves, flat sides, corners, steps, and notches,
placing more points where the boundary changes direction. Use up to ${range.max}
points when needed for detail; do not pad a simple outline to the maximum count.` : ""}
Put all points into the single outline
string using only this compact format: "430,250 438,229 455,215". Do not put
brackets, x/y keys, semicolons, or explanations inside the string. Follow the
real shape closely, including flat sides and corners. Do not repeat the first
point at the end. Use the exact VIEW_ID supplied before each image. Return every
visible scanbody separately, including partially visible ones. When a likely
scanbody is visible but you are not fully certain, include it with lower
confidence instead of returning an empty view. An empty top or occlusal view is
valid only after checking that no repeated protruding manufactured posts are
present. Never invent a detection when no manufactured post is visible.`;
}

export async function POST(request: Request) {
  try {
    const body = requestSchema.parse(await request.json());
    const outlinePointRange = getOutlinePointRange();
    const detectionSchema = createDetectionSchema(outlinePointRange);
    const selection = resolveModelSelection(body.run);
    // Honor the baseline provider toggle even if that provider's key fails.
    const allowOpenRouterFallback = body.run.type !== "baseline" && Boolean(process.env.OPENROUTER_API_KEY);
    const directClaude = body.directClaude && selection.provider === "anthropic";
    const isOpenRouterOpenAi = selection.provider === "openrouter" &&
      getAiModelOption(selection.model).apiModelName?.startsWith("openai/");

    if (selection.provider === "openai" && !process.env.OPENAI_API_KEY && !allowOpenRouterFallback) {
      throw new Error(
        body.run.type === "baseline"
          ? "OPENAI_API_KEY is missing. Add it to .env.local or enable OpenRouter for the baseline."
          : "OPENAI_API_KEY or OPENROUTER_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }
    if (selection.provider === "google" && !process.env.GOOGLE_API_KEY) {
      throw new Error(
        "GOOGLE_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }
    if (selection.provider === "anthropic" && !process.env.ANTHROPIC_API_KEY) {
      throw new Error(
        "ANTHROPIC_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }
    if (selection.provider === "mistral" && !process.env.MISTRAL_API_KEY) {
      throw new Error(
        "MISTRAL_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }
    if (selection.provider === "openrouter" && !process.env.OPENROUTER_API_KEY) {
      throw new Error(
        "OPENROUTER_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }
    if (selection.provider === "xai" && !process.env.XAI_API_KEY) {
      throw new Error(
        "XAI_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }

    const imageInstruction = `Every image is ${body.imageSize} by ${body.imageSize} pixels. Return each outline as one compact string containing ${outlinePointInstruction(outlinePointRange)} ordered whole-number pixel pairs in "x,y x,y" format.${outlinePointRange.min < outlinePointRange.max ? " Choose the fewest points needed to follow each scanbody's actual boundary closely." : ""}`;
    const systemPrompt = createSystemPrompt(outlinePointRange);
    type RawDetection = z.infer<typeof detectionSchema>["views"][number]["detections"][number];
    async function analyzeBatch(batchViews: typeof body.views): Promise<ViewBatchResult<RawDetection>> {
      // Every concurrent request needs its own client and mutable billing receipt.
      const openRouterReceipt: OpenRouterReceipt = { serviceTier: null, reportedCostUsd: null };
      const detector = directClaude || selection.provider === "xai"
        ? null
        : selection.provider === "openai"
          ? (!process.env.OPENAI_API_KEY && allowOpenRouterFallback
              ? new ChatOpenAI({
                  model: `openai/${selection.model}`,
                  apiKey: process.env.OPENROUTER_API_KEY,
                  maxRetries: 1,
                  configuration: {
                    baseURL: "https://openrouter.ai/api/v1",
                  },
                  modelKwargs: {
                    ...(selection.reasoningEffort !== "default"
                      ? { reasoning: { effort: selection.reasoningEffort } }
                      : {}),
                  },
                })
              : new ChatOpenAI({
                  model: selection.model,
                  apiKey: process.env.OPENAI_API_KEY,
                  maxRetries: 1,
                  useResponsesApi: true,
                  reasoning: { effort: selection.reasoningEffort as "low" | "medium" | "high" },
                })
            ).withStructuredOutput(detectionSchema, {
              name: "scanbody_roi_detection",
              strict: Boolean(process.env.OPENAI_API_KEY),
              includeRaw: true,
            })
          : selection.provider === "google"
            ? new CurrentGeminiChatModel({
                model: selection.model,
                apiKey: process.env.GOOGLE_API_KEY,
                maxRetries: 1,
                maxOutputTokens: 65_536,
                thinkingConfig: {
                  // The installed LangChain type has not yet added MINIMAL, but the
                  // current Gemini API supports it for Flash-Lite.
                  thinkingLevel: selection.reasoningEffort.toUpperCase() as
                    | "LOW"
                    | "MEDIUM"
                    | "HIGH",
                },
              }).withStructuredOutput(detectionSchema, {
                name: "scanbody_roi_detection",
                includeRaw: true,
              })
            : selection.provider === "mistral"
              ? new MistralChatModel({
                  model: selection.model,
                  apiKey: process.env.MISTRAL_API_KEY,
                  maxRetries: 1,
                  // Large 3 uses its default behavior, so omit reasoning_effort.
                  reasoningEffort:
                    selection.reasoningEffort === "default"
                      ? undefined
                      : (selection.reasoningEffort as MistralEffort),
                }).withStructuredOutput(detectionSchema, {
                  name: "scanbody_roi_detection",
                  includeRaw: true,
                })
              : selection.provider === "openrouter"
                ? new ChatOpenAI({
                    model:
                      getAiModelOption(selection.model).apiModelName ??
                      selection.model,
                    apiKey: process.env.OPENROUTER_API_KEY,
                    // A tier mismatch must not trigger another potentially paid call.
                    maxRetries: selection.processingSpeed === "standard" ? 1 : 0,
                    configuration: {
                      baseURL: "https://openrouter.ai/api/v1",
                      fetch: createOpenRouterReceiptFetch(openRouterReceipt, isOpenRouterOpenAi ? selection.processingSpeed : undefined),
                    },
                    modelKwargs: {
                      ...(isOpenRouterOpenAi ? getOpenRouterSpeedRouting(selection.processingSpeed) : {}),
                      ...(selection.reasoningEffort !== "default"
                        ? { reasoning: { effort: selection.reasoningEffort } }
                        : {}),
                    },
                  }).withStructuredOutput(detectionSchema, {
                    name: "scanbody_roi_detection",
                    includeRaw: true,
                  })
                : new ChatAnthropic({
                // Provider validation above guarantees this branch only receives
                // a selectable Claude ID from the shared model whitelist.
                model: selection.model as ClaudeModelId,
                apiKey: process.env.ANTHROPIC_API_KEY,
                maxRetries: 1,
                // Fable 5.1 and the selectable Claude 5.5 models support 128K output.
                // This is only a ceiling; Anthropic bills tokens actually used.
                maxTokens: 128_000,
                // Anthropic's SDK rejects large non-streaming requests before
                // sending them. LangChain collects this internal stream and the
                // route still returns one normal final JSON response to the UI.
                streaming: true,
                // Fable 5.1 and the Claude 5.5 family use adaptive thinking and
                // the native effort parameter for Low, Medium, and High.
                thinking: { type: "adaptive" },
                outputConfig: {
                  effort: selection.reasoningEffort as ClaudeEffort,
                },
              }).withStructuredOutput(detectionSchema, {
                name: "scanbody_roi_detection",
                // Anthropic's native JSON-schema output avoids extra tool-call
                // wrapper tokens and returns the same parsed shape as other providers.
                method: "jsonSchema",
                includeRaw: true,
              });

      let parsed: z.infer<typeof detectionSchema>;
      let inputTokens: number | null;
      let cachedInputTokens: number | null;
      let outputTokens: number | null;
      if (directClaude) {
        // Direct SDK branch uses the same prompt and the requested view batch.
        const result = await analyzeClaudeDirect({
          apiKey: process.env.ANTHROPIC_API_KEY!,
          model: selection.model as ClaudeModelId,
          effort: selection.reasoningEffort as ClaudeEffort,
          systemPrompt,
          imageInstruction,
          views: batchViews,
          detectionSchema,
        });
        parsed = result.parsed;
        inputTokens = result.inputTokens;
        cachedInputTokens = result.cachedInputTokens;
        outputTokens = result.outputTokens;
      } else {
        const content: HumanMessage["content"] = [
          { type: "text", text: imageInstruction },
        ];
        for (const view of batchViews) {
          content.push({ type: "text", text: `VIEW_ID: ${view.name}` });
          content.push({
            type: "image_url",
            image_url: { url: view.image, detail: "high" },
          });
        }
        const messages = [
          new SystemMessage(systemPrompt),
          new HumanMessage({ content }),
        ];
        if (selection.provider === "xai") {
          // LangChain's xAI Responses model carries this view batch. Ask xAI for
          // native JSON Schema output, then validate the returned JSON locally.
          const grok = new ScanbodyChatXAI({
            model: selection.model,
            apiKey: process.env.XAI_API_KEY,
            reasoning: { effort: selection.reasoningEffort as "low" | "medium" | "high" },
            store: false,
          });
          const format = {
            type: "json_schema" as const,
            name: "scanbody_roi_detection",
            schema: z.toJSONSchema(detectionSchema),
            strict: true,
          };
          const result = await grok.invoke(messages, { text: { format } });
          if (typeof result.content !== "string") {
            throw new Error("Grok returned an unexpected response format.");
          }
          parsed = detectionSchema.parse(JSON.parse(result.content));
          inputTokens = result.usage_metadata?.input_tokens ?? null;
          cachedInputTokens = result.usage_metadata?.input_token_details?.cache_read ?? null;
          outputTokens = result.usage_metadata?.output_tokens ?? null;
        } else {
          let result: {
            parsed: z.infer<typeof detectionSchema>;
            raw: unknown;
          };
          try {
            result = await detector!.invoke(messages);
          } catch (detectorError) {
            const isQuotaOrAuthError =
              selection.provider === "openai" &&
              allowOpenRouterFallback &&
              detectorError instanceof Error &&
              /quota|credit|insufficient|billing|429|402|401/i.test(detectorError.message);
            if (isQuotaOrAuthError) {
              const fallbackOpenRouterDetector = new ChatOpenAI({
                model: `openai/${selection.model}`,
                apiKey: process.env.OPENROUTER_API_KEY,
                maxRetries: 1,
                configuration: {
                  baseURL: "https://openrouter.ai/api/v1",
                },
                modelKwargs: {
                  ...(selection.reasoningEffort !== "default"
                    ? { reasoning: { effort: selection.reasoningEffort } }
                    : {}),
                },
              }).withStructuredOutput(detectionSchema, {
                name: "scanbody_roi_detection",
                includeRaw: true,
              });
              result = await fallbackOpenRouterDetector.invoke(messages);
            } else {
              throw detectorError;
            }
          }
          parsed = result.parsed;
          const usage = (
            result.raw as {
              usage_metadata?: {
                input_tokens?: number;
                output_tokens?: number;
                input_token_details?: { cache_read?: number };
              };
            }
          ).usage_metadata;
          inputTokens = usage?.input_tokens ?? null;
          cachedInputTokens = usage?.input_token_details?.cache_read ?? null;
          outputTokens = usage?.output_tokens ?? null;
        }
      }
      return {
        parsed,
        inputTokens,
        cachedInputTokens,
        outputTokens,
        ...openRouterReceipt,
        estimatedCostUsd: openRouterReceipt.reportedCostUsd ?? estimatePaidCostUsd({
          modelId: selection.model,
          inputTokens,
          cachedInputTokens,
          outputTokens,
          processingSpeed: selection.processingSpeed,
        }),
      };
    }

    const startedAt = performance.now();
    const {
      parsed, inputTokens, cachedInputTokens, outputTokens, estimatedCostUsd,
      reportedCostUsd, serviceTier, viewRequestCount, viewTimings,
    } = await analyzeViewBatches(body.views, body.parallelViews, analyzeBatch);
    const elapsedMs = Math.round(performance.now() - startedAt);

    // TEMPORARY DEBUG SAVE — REMOVE LATER:
    // This saves the exact structured JSON returned by the LLM, before the app
    // converts the compact outline string into [x, y] arrays or performs 3D cleanup.
    // In parallel mode it combines the views from all real responses unchanged.
    // To remove this feature later, delete this commented block and the two
    // node:fs/promises / node:path imports at the top of this file.
    if (body.saveFiles) {
      const debugOutputDirectory = join(process.cwd(), "debug-output");
      await mkdir(debugOutputDirectory, { recursive: true });
      await writeFile(
        join(debugOutputDirectory, "latest-llm-response.json"),
        JSON.stringify(parsed, null, 2),
        "utf8",
      );
    }

    const validViewNames = new Set(body.views.map((view) => view.name));
    const clampCoordinate = (coordinate: number) =>
      Math.max(0, Math.min(body.imageSize, coordinate));

    const views = parsed.views
      .filter((view) => validViewNames.has(view.view))
      .map((view) => ({
        view: view.view,
        detections: view.detections.flatMap((detection) => {
          const outline = parseCompactOutline(detection.outline);
          // Three points are the minimum needed to form a polygon. Ignore only
          // a malformed detection instead of failing and losing the whole run.
          if (outline.length < 3) return [];
          return [{
            label: detection.label,
            outline: outline.map(
              ([x, y]) =>
                [clampCoordinate(x), clampCoordinate(y)] as [number, number],
            ),
            confidence: Math.max(0, Math.min(1, detection.confidence)),
          }];
        }),
      }));

    return NextResponse.json({
      views,
      meta: {
        model: selection.model,
        modelLabel: `${getModelRunLabel(selection.model, selection.reasoningEffort, selection.processingSpeed)}${directClaude ? " · Direct API" : ""}${body.parallelViews ? " · Parallel" : ""}`,
        provider: selection.provider,
        ...(selection.provider === "anthropic"
          ? { integration: directClaude ? "anthropic-sdk" : "langchain" }
          : {}),
        reasoningEffort: selection.reasoningEffort,
        ...(isOpenRouterOpenAi ? { processingSpeed: selection.processingSpeed } : {}),
        ...(selection.provider === "openrouter" ? { serviceTier, reportedCostUsd } : {}),
        parallelViews: body.parallelViews,
        viewRequestCount,
        viewTimings,
        // Keep the old field as the configured ceiling for existing consumers.
        outlinePointsPerScanbody: outlinePointRange.max,
        outlineMinPointsPerScanbody: outlinePointRange.min,
        outlineMaxPointsPerScanbody: outlinePointRange.max,
        elapsedMs,
        inputTokens,
        cachedInputTokens,
        outputTokens,
        estimatedCostUsd,
        // Show OpenRouter's reported cost or the calculated paid cost. Keep this legacy
        // field only so old saved JSON files remain compatible.
        freeTierEligible: false,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Analysis failed.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

