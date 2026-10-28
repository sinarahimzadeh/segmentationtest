import { mkdir, readdir, readFile, unlink, writeFile } from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";
import { NextResponse } from "next/server";
import { z } from "zod";
import { OPENAI_MODEL_IDS } from "@/lib/openai-models";

const meshKeySchema = z.string().regex(/^[a-f0-9]{64}$/);
const pointSchema = z.tuple([z.number(), z.number()]);
const analysisSchema = z.object({
  views: z.array(
    z.object({
      view: z.string(),
      detections: z.array(
        z.object({
          label: z.literal("scanbody"),
          outline: z.array(pointSchema),
          confidence: z.number(),
        }),
      ),
    }),
  ),
  meta: z.object({
    model: z.string(),
    modelLabel: z.string(),
    reasoningEffort: z.enum(["low", "medium"]),
    elapsedMs: z.number(),
    inputTokens: z.number().nullable(),
    cachedInputTokens: z.number().nullable(),
    outputTokens: z.number().nullable(),
    estimatedCostUsd: z.number().nullable(),
  }),
});
const metricsSchema = z.object({
  sameTrianglePercent: z.number(),
  baselineFoundPercent: z.number(),
  missedBaselinePercent: z.number(),
  extraCandidatePercent: z.number(),
  sharedTriangleCount: z.number().int().nonnegative(),
  missedTriangleCount: z.number().int().nonnegative(),
  extraTriangleCount: z.number().int().nonnegative(),
  objectCountDifference: z.number().int(),
});
const savedRunSchema = z.object({
  id: z.string().min(1),
  storageVersion: z.number().int().positive(),
  meshKey: meshKeySchema,
  meshName: z.string().min(1).max(500),
  kind: z.enum(["baseline", "comparison"]),
  model: z.enum(OPENAI_MODEL_IDS),
  reasoningEffort: z.enum(["low", "medium"]),
  createdAt: z.string(),
  analysis: analysisSchema,
  selectedTriangleIds: z.array(z.number().int().nonnegative()),
  objectCount: z.number().int().nonnegative(),
  metrics: metricsSchema.nullable(),
});

const outputRoot = resolve(process.cwd(), "benchmark-output");

function getMeshDirectory(meshKey: string) {
  const directory = resolve(outputRoot, meshKey);
  // meshKey is already restricted to 64 hexadecimal characters. Keep this
  // second check so future edits cannot accidentally write outside the folder.
  if (!directory.startsWith(`${outputRoot}${sep}`)) {
    throw new Error("Invalid benchmark output path.");
  }
  return directory;
}

function comparisonFileName(run: z.infer<typeof savedRunSchema>) {
  const timestamp = run.createdAt.replace(/[:.]/g, "-");
  return `${timestamp}__${run.model}__${run.reasoningEffort}.json`;
}

async function removeOldComparisonFiles(comparisonsDirectory: string) {
  await mkdir(comparisonsDirectory, { recursive: true });
  const entries = await readdir(comparisonsDirectory, { withFileTypes: true });
  await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
      .map((entry) => unlink(join(comparisonsDirectory, entry.name))),
  );
}

export async function POST(request: Request) {
  try {
    const run = savedRunSchema.parse(await request.json());
    const meshDirectory = getMeshDirectory(run.meshKey);
    const comparisonsDirectory = join(meshDirectory, "comparisons");
    await mkdir(comparisonsDirectory, { recursive: true });

    const outputPath =
      run.kind === "baseline"
        ? join(meshDirectory, "baseline-sol-medium.json")
        : join(comparisonsDirectory, comparisonFileName(run));

    // A new baseline makes all old comparison scores invalid. Save the new
    // baseline first, then remove only the old comparison JSON files.
    await writeFile(outputPath, JSON.stringify(run, null, 2), "utf8");
    if (run.kind === "baseline") {
      await removeOldComparisonFiles(comparisonsDirectory);
    }

    return NextResponse.json({
      savedPath: relative(process.cwd(), outputPath).replaceAll("\\", "/"),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Save failed.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const meshKey = meshKeySchema.parse(url.searchParams.get("meshKey"));
    const meshDirectory = getMeshDirectory(meshKey);
    const comparisonsDirectory = join(meshDirectory, "comparisons");
    const runs: unknown[] = [];

    try {
      runs.push(
        JSON.parse(
          await readFile(
            join(meshDirectory, "baseline-sol-medium.json"),
            "utf8",
          ),
        ),
      );
    } catch {
      // No saved baseline for this STL yet.
    }

    try {
      const files = (await readdir(comparisonsDirectory))
        .filter((name) => name.endsWith(".json"))
        .sort()
        .reverse();
      const comparisons = await Promise.all(
        files.map(async (name) =>
          JSON.parse(await readFile(join(comparisonsDirectory, name), "utf8")),
        ),
      );
      runs.push(...comparisons);
    } catch {
      // The comparisons folder is created after the first saved run.
    }

    return NextResponse.json({
      runs: runs.flatMap((run) => {
        const parsed = savedRunSchema.safeParse(run);
        return parsed.success ? [parsed.data] : [];
      }),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Load failed.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
