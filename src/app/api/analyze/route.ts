import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOpenAI } from "@langchain/openai";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";
import { z } from "zod";
import {
  AI_MODEL_IDS,
  BASELINE_SELECTION,
  estimatePaidCostUsd,
  getAiModelOption,
  getModelRunLabel,
  isAllowedComparisonSelection,
  type AiModelId,
  type AiProvider,
  type ReasoningEffort,
} from "@/lib/ai-models";

const requestSchema = z.object({
  imageSize: z.number().int().min(256).max(2048),
  run: z.discriminatedUnion("type", [
    z.object({ type: z.literal("baseline") }),
    z.object({
      type: z.literal("comparison"),
      model: z.enum(AI_MODEL_IDS),
      reasoningEffort: z.enum(["minimal", "low", "medium", "high"]),
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
} {
  if (run.type === "baseline") {
    return {
      ...BASELINE_SELECTION,
      provider: getAiModelOption(BASELINE_SELECTION.model).provider,
    };
  }
  if (!isAllowedComparisonSelection(run.model, run.reasoningEffort)) {
    throw new Error(
      `${run.model} does not support ${run.reasoningEffort} in this benchmark.`,
    );
  }
  return {
    model: run.model,
    provider: getAiModelOption(run.model).provider,
    reasoningEffort: run.reasoningEffort,
  };
}

const DEFAULT_OUTLINE_POINTS_PER_SCANBODY = 40;
const MAX_OUTLINE_POINTS_PER_SCANBODY = 10_000;

function getOutlinePointCount() {
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

  return pointCount;
}

function createDetectionSchema(pointCount: number) {
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
            outline: z
              .array(
                z.object({
                  x: z.number().describe("Horizontal pixel coordinate."),
                  y: z.number().describe("Vertical pixel coordinate."),
                }),
              )
              .length(pointCount)
              .describe(
                `Exactly ${pointCount} ordered {x, y} pixel coordinates tracing the visible scanbody boundary.`,
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

function createSystemPrompt(pointCount: number) {
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
For every visible scanbody, trace its visible outer boundary with exactly
${pointCount} ordered points. Follow the real shape closely, including flat
sides and corners. Do not repeat the first point at the end. Use the exact
VIEW_ID supplied before each image. Return every visible scanbody separately,
including partially visible ones. When a likely scanbody is visible but you are
not fully certain, include it with lower confidence instead of returning an
empty view. An empty top or occlusal view is valid only after checking that no
repeated protruding manufactured posts are present. Never invent a detection
when no manufactured post is visible.`;
}

export async function POST(request: Request) {
  try {
    const body = requestSchema.parse(await request.json());
    const outlinePointCount = getOutlinePointCount();
    const detectionSchema = createDetectionSchema(outlinePointCount);
    const selection = resolveModelSelection(body.run);
    const modelOption = getAiModelOption(selection.model);

    if (selection.provider === "openai" && !process.env.OPENAI_API_KEY) {
      throw new Error(
        "OPENAI_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }
    if (selection.provider === "google" && !process.env.GOOGLE_API_KEY) {
      throw new Error(
        "GOOGLE_API_KEY is missing. Add it to .env.local and restart the server.",
      );
    }

    const detector =
      selection.provider === "openai"
        ? new ChatOpenAI({
            model: selection.model,
            apiKey: process.env.OPENAI_API_KEY,
            maxRetries: 1,
            useResponsesApi: true,
            reasoning: { effort: selection.reasoningEffort },
          }).withStructuredOutput(detectionSchema, {
            name: "scanbody_roi_detection",
            strict: true,
            includeRaw: true,
          })
        : new ChatGoogleGenerativeAI({
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
          });

    const content: HumanMessage["content"] = [
      {
        type: "text",
        text: `Every image is ${body.imageSize} by ${body.imageSize} pixels. Return each outline as exactly ${outlinePointCount} ordered objects with numeric x and y pixel coordinates.`,
      },
    ];

    for (const view of body.views) {
      content.push({ type: "text", text: `VIEW_ID: ${view.name}` });
      content.push({
        type: "image_url",
        image_url: { url: view.image, detail: "high" },
      });
    }

    const startedAt = performance.now();
    const result = await detector.invoke([
      new SystemMessage(createSystemPrompt(outlinePointCount)),
      new HumanMessage({ content }),
    ]);
    const elapsedMs = Math.round(performance.now() - startedAt);

    // TEMPORARY DEBUG SAVE — REMOVE LATER:
    // This saves the exact structured JSON returned by the LLM, before the app
    // converts {x, y} outline points into [x, y] arrays or performs 3D cleanup.
    // To remove this feature later, delete this commented block and the two
    // node:fs/promises / node:path imports at the top of this file.
    const debugOutputDirectory = join(process.cwd(), "debug-output");
    await mkdir(debugOutputDirectory, { recursive: true });
    await writeFile(
      join(debugOutputDirectory, "latest-llm-response.json"),
      JSON.stringify(result.parsed, null, 2),
      "utf8",
    );

    const validViewNames = new Set(body.views.map((view) => view.name));
    const clampCoordinate = (coordinate: number) =>
      Math.max(0, Math.min(body.imageSize, coordinate));

    const views = result.parsed.views
      .filter((view) => validViewNames.has(view.view))
      .map((view) => ({
        view: view.view,
        detections: view.detections.map((detection) => ({
          label: detection.label,
          outline: detection.outline.map(
            ({ x, y }) =>
              [clampCoordinate(x), clampCoordinate(y)] as [number, number],
          ),
          confidence: Math.max(0, Math.min(1, detection.confidence)),
        })),
      }));

    const usage = (
      result.raw as {
        usage_metadata?: {
          input_tokens?: number;
          output_tokens?: number;
          input_token_details?: { cache_read?: number };
        };
      }
    ).usage_metadata;
    const inputTokens = usage?.input_tokens ?? null;
    const cachedInputTokens = usage?.input_token_details?.cache_read ?? null;
    const outputTokens = usage?.output_tokens ?? null;
    return NextResponse.json({
      views,
      meta: {
        model: selection.model,
        modelLabel: getModelRunLabel(
          selection.model,
          selection.reasoningEffort,
        ),
        provider: selection.provider,
        reasoningEffort: selection.reasoningEffort,
        outlinePointsPerScanbody: outlinePointCount,
        elapsedMs,
        inputTokens,
        cachedInputTokens,
        outputTokens,
        estimatedCostUsd: estimatePaidCostUsd({
          modelId: selection.model,
          inputTokens,
          cachedInputTokens,
          outputTokens,
        }),
        freeTierEligible: modelOption.freeTierEligible,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Analysis failed.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

