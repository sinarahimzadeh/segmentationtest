import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { NextResponse } from "next/server";
import { z } from "zod";

const requestSchema = z.object({
  imageSize: z.number().int().min(256).max(2048),
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

const DEFAULT_OUTLINE_POINTS_PER_SCANBODY = 500;
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
            label: z.literal("scanbody"),
            outline: z
              .array(z.tuple([z.number(), z.number()]))
              .length(pointCount)
              .describe(
                `Exactly ${pointCount} ordered [x, y] pixel coordinates tracing the visible scanbody boundary.`,
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
rectangular, indexed, or cap-like manufactured geometry. Do not mark teeth,
gingiva, crowns, holes, healing abutments, or random mesh artifacts.

The images are different camera views of the same mesh. Some images can show
the underside, where no scanbody is visible. Treat every view independently.
For every visible scanbody, trace its visible outer boundary with exactly
${pointCount} ordered points. Follow the real shape closely, including flat
sides and corners. Do not repeat the first point at the end. Use the exact
VIEW_ID supplied before each image. Never invent a detection when none is visible.`;
}

export async function POST(request: Request) {
  if (!process.env.OPENAI_API_KEY) {
    return NextResponse.json(
      {
        error:
          "OPENAI_API_KEY is missing. Add it to .env.local and restart the server.",
      },
      { status: 503 },
    );
  }

  try {
    const body = requestSchema.parse(await request.json());
    const outlinePointCount = getOutlinePointCount();
    const detectionSchema = createDetectionSchema(outlinePointCount);
    const modelName = process.env.OPENAI_VISION_MODEL || "gpt-6.1-sol";
    const model = new ChatOpenAI({
      model: modelName,
      apiKey: process.env.OPENAI_API_KEY,
      maxRetries: 1,
      timeout: 55_000,
      useResponsesApi: true,
    });
    const detector = model.withStructuredOutput(detectionSchema, {
      name: "scanbody_roi_detection",
      strict: true,
      includeRaw: true,
    });

    const content: HumanMessage["content"] = [
      {
        type: "text",
        text: `Every image is ${body.imageSize} by ${body.imageSize} pixels. Return each outline as exactly ${outlinePointCount} ordered [x, y] pixel coordinates.`,
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
            ([x, y]) =>
              [clampCoordinate(x), clampCoordinate(y)] as [number, number],
          ),
          confidence: Math.max(0, Math.min(1, detection.confidence)),
        })),
      }));

    const usage = (
      result.raw as {
        usage_metadata?: { input_tokens?: number; output_tokens?: number };
      }
    ).usage_metadata;
    return NextResponse.json({
      views,
      meta: {
        model: modelName,
        outlinePointsPerScanbody: outlinePointCount,
        elapsedMs,
        inputTokens: usage?.input_tokens ?? null,
        outputTokens: usage?.output_tokens ?? null,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Analysis failed.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

