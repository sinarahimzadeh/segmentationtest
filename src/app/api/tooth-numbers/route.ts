import { HumanMessage, SystemMessage, type AIMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";
import { z } from "zod";
import { estimatePaidCostUsd, getBaselineSelection, getModelRunLabel } from "@/lib/ai-models";
import { createOpenRouterReceiptFetch, getOpenRouterSpeedRouting, type OpenRouterReceipt } from "@/lib/openrouter-speed";
import {
  JAW_HINTS, RIGHT_SIDE_HINTS, normalizeToothAssignments,
  toothNumberResponseSchema, TOOTH_NUMBER_SYSTEM_PROMPT,
} from "@/lib/tooth-numbering";

const requestSchema = z.object({
  meshKey: z.string().regex(/^[a-f0-9]{64}$/),
  meshName: z.string().min(1).max(500),
  imageSize: z.number().int().min(256).max(2048),
  jpegQuality: z.number().int().min(1).max(100).default(90),
  useOpenRouter: z.boolean().default(true),
  saveFiles: z.boolean().default(true),
  jawHint: z.enum(JAW_HINTS).default("auto"),
  rightSideHint: z.enum(RIGHT_SIDE_HINTS).default("unknown"),
  referenceView: z.enum(["top", "bottom"]).default("top"),
  views: z.array(z.object({
    name: z.string().min(1).max(40),
    image: z.string().startsWith("data:image/"),
    detections: z.array(z.object({
      detectionId: z.string().min(1).max(80),
      outline: z.array(z.tuple([z.number().finite(), z.number().finite()])).min(3).max(10_000),
    })).max(100),
  })).min(1).max(10),
});

/** Standalone request: never calls /api/analyze or changes benchmark files. */
export async function POST(request: Request) {
  try {
    const body = requestSchema.parse(await request.json());
    // Send only the chosen arch-facing image, even if an older client sends eight.
    const referenceView = body.views.find((view) => view.name === body.referenceView);
    if (!referenceView) throw new Error(`Tooth numbering requires the selected ${body.referenceView} camera image.`);
    const targets = referenceView.detections.map((detection) => ({
      detectionId: detection.detectionId, view: referenceView.name,
    }));
    if (targets.length === 0) throw new Error(`No scanbody outlines were detected in the ${body.referenceView} image. Choose the other view or run segmentation first.`);
    if (new Set(body.views.map((view) => view.name)).size !== body.views.length ||
        new Set(targets.map((target) => target.detectionId)).size !== targets.length) {
      throw new Error("View names and detection IDs must be unique.");
    }
    const selection = getBaselineSelection(body.useOpenRouter);
    const apiKey = body.useOpenRouter ? process.env.OPENROUTER_API_KEY : process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error(`${body.useOpenRouter ? "OPENROUTER_API_KEY" : "OPENAI_API_KEY"} is missing in .env.local.`);
    const receipt: OpenRouterReceipt = { serviceTier: null, reportedCostUsd: null };
    const detector = new ChatOpenAI({
      model: body.useOpenRouter ? "openai/gpt-6.1-sol" : selection.model,
      apiKey,
      maxRetries: 0,
      ...(body.useOpenRouter ? {
        configuration: {
          baseURL: "https://openrouter.ai/api/v1",
          fetch: createOpenRouterReceiptFetch(receipt, "standard"),
        },
        modelKwargs: { ...getOpenRouterSpeedRouting("standard"), reasoning: { effort: "medium" } },
      } : {
        useResponsesApi: true,
        reasoning: { effort: "medium" },
      }),
    }).withStructuredOutput(toothNumberResponseSchema, {
      name: "scanbody_tooth_numbers", includeRaw: true, strict: true,
    });
    const content: HumanMessage["content"] = [{
      type: "text",
      text: `The selected ${body.referenceView} image is ${body.imageSize}×${body.imageSize} pixels. Jaw hint: ${body.jawHint}. Patient right in this image: ${body.rightSideHint}. Outlines are ordered x,y pixel pairs. Use the complete dental arch visible in this image to establish anatomy.`,
    }];
    // One image and only its matching outlines; never add another camera angle.
    content.push({ type: "text", text: `VIEW_ID: ${referenceView.name}\nDetected scanbodies: ${JSON.stringify(referenceView.detections)}` });
    content.push({ type: "image_url", image_url: { url: referenceView.image, detail: "high" } });
    const startedAt = performance.now();
    const result = await detector.invoke([
      new SystemMessage(TOOTH_NUMBER_SYSTEM_PROMPT), new HumanMessage({ content }),
    ]);
    const numbering = normalizeToothAssignments(toothNumberResponseSchema.parse(result.parsed), targets, body.jawHint);
    const usage = (result.raw as AIMessage).usage_metadata;
    const inputTokens = usage?.input_tokens ?? null;
    const outputTokens = usage?.output_tokens ?? null;
    const output = {
      ...numbering,
      meta: {
        modelLabel: getModelRunLabel(selection.model, "medium"),
        imageCount: 1, viewNames: [referenceView.name],
        imageSize: body.imageSize, jpegQuality: body.jpegQuality,
        elapsedMs: Math.round(performance.now() - startedAt), inputTokens, outputTokens,
        estimatedCostUsd: receipt.reportedCostUsd ?? estimatePaidCostUsd({
          modelId: selection.model, inputTokens, outputTokens,
          cachedInputTokens: usage?.input_token_details?.cache_read ?? null,
        }),
      },
      savedPath: null as string | null,
      saveError: undefined as string | undefined,
    };
    if (body.saveFiles) {
      // Separate files; never overwrite the segmentation's baseline, comparison,
      // or latest-llm-response.json. Screenshots are not copied into this JSON.
      const directory = join(process.cwd(), "benchmark-output", body.meshKey, "tooth-numbers");
      const filename = `${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
      try {
        await mkdir(directory, { recursive: true });
        const path = join(directory, filename);
        output.savedPath = path;
        await writeFile(path, JSON.stringify({
          meshKey: body.meshKey, meshName: body.meshName,
          jawHint: body.jawHint, rightSideHint: body.rightSideHint,
          referenceView: referenceView.name,
          targets, rawResponse: result.parsed, result: output,
        }, null, 2), "utf8");
      } catch {
        output.savedPath = null;
        output.saveError = "Tooth numbers were returned, but their separate JSON file could not be saved.";
      }
    }
    return NextResponse.json(output);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Tooth numbering failed." }, { status: 400 });
  }
}
