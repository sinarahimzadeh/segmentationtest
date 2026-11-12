# Scanbody ROI Demo

A small local Next.js prototype that finds likely dental scanbodies in an STL scan.

## How it works

1. The STL is loaded in the browser.
2. PCA centers and consistently turns the model.
3. Three.js takes eight JPEG pictures from fixed angles.
4. LangChain sends those pictures to the selected AI model.
5. The API returns strict JSON outlines around likely scanbodies.
6. The app projects those outlines back onto the 3D mesh and highlights the matching areas.

The STL file itself stays in the browser. Only the rendered JPEG views are sent to the AI service.

## Setup

Create `.env.local` in this folder:

```env
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_google_ai_studio_key_here
XAI_API_KEY=your_xai_api_key_here
SCANBODY_OUTLINE_POINTS=40
SCANBODY_OUTLINE_MIN_POINTS=16
```

The AI chooses the number of outline points separately for each scanbody in
each image: fewer for simple boundaries, more for detailed ones. The default
range is **16–40 points**. `SCANBODY_OUTLINE_POINTS` sets the maximum (`3` to
`10000`), and `SCANBODY_OUTLINE_MIN_POINTS` sets the minimum (`3` to the maximum).
Set both to `40` to restore a fixed count. Restart the app after editing them.
These settings apply to every provider, in grouped and parallel mode. Saved
results record the requested minimum and maximum; each outline array contains
the actual returned points. This may reduce output and time, but the quality
and speed difference must be measured against the saved baseline.

Then run:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Model runs and benchmark

**Image settings**, below the model-runs table and above Highlight, controls
all subsequent baseline, selected-model, and tooth-numbering requests.
Dimensions: **1024×1024** (default), 896×896, 768×768, 640×640, 512×512,
or 384×384. JPEG quality: **90%** (default), 80%, 70%, 60%, 50%, 40%, or 30%.
Changing either dropdown updates the camera
previews locally; it does not make an API call. All eight segmentation views,
camera angles, lighting, model and thinking settings stay the same.

Every option is encoded directly from original render pixels, so repeated
setting changes do not accumulate JPEG compression. The 3D triangle-ID maps
remain at 1024×1024, and returned outlines are scaled to match them. Saved
results record `analysis.meta.imageSize` and `analysis.meta.jpegQuality`;
outlines in those files use that run's image dimensions. Older results without
these fields use 1024×1024 and 90%. Changing settings does not rerun or alter a
displayed result; click the relevant run button to measure the new settings.

Choose an STL, then select any model and thinking level under **Run another
model**. **Compare with Sol Medium** and **Save selected model results** both
default to off. Click **Run segmentation** to use the model without creating
a baseline or writing result files. Highlighting, exports, and tooth numbering
still work with this result.

The two switches are independent:

| Compare | Save | Behavior |
| --- | --- | --- |
| Off | Off | Normal segmentation, kept only for this session; no baseline needed |
| Off | On | Normal segmentation saved under `benchmark-output/<STL SHA-256>/runs/` |
| On | Off | Compare against an existing baseline, kept only for this session |
| On | On | Compare against an existing baseline and save the comparison |

The bottom camera-panel footer shows **PCA**, **Screenshots**, the existing
**Model time**, and **3D mapping** separately. Each browser timer starts
immediately before its function call and stops immediately after it returns:
`orientGeometryWithPca`, `captureModelViews`, and `mapAnalysisToTriangles`.
Capture includes all eight JPEG screenshots and hidden triangle-ID renders,
renderer setup, pixel readback, and resource cleanup. Mapping includes local
outline-to-triangle conversion, voting, and cleanup, but excludes API waiting,
highlight drawing, export, and saving. File loading and STL parsing are also
outside these timers.

With saving enabled, `analysis.meta` in baseline, comparison, or standalone-run
JSON records `pcaElapsedMs`, `captureElapsedMs`, and `mappingElapsedMs`, all in
milliseconds. `elapsedMs` remains the existing LLM request time; the legacy
`preparationElapsedMs` remains PCA + capture time. With saving off, measurements
stay in the current session. Older saved runs use current-load PCA/capture
values if separate measurements were not recorded, and display a dash for
missing mapping measurements. Reload the STL to measure the new separate stages.

Parallel runs also show an **AI request time under each camera image** and in
the enlarged preview. Each clock starts immediately before that image's request
and stops when its parsed response returns. It includes client setup, provider
and network waiting, and any configured retries; it is not pure inference time.
The measurements are kept in `analysis.meta.viewTimings` and saved with results
when saving is enabled. The existing overall Model time remains unchanged and
is not the sum of overlapping requests. Non-parallel runs use one shared request
and show **AI: shared request**, without invented per-image times. Older saved
results without measurements show a dash. Timing requires no extra AI requests.

To compare models, first run the fixed **GPT-6.1 Sol / Medium** baseline and
enable **Compare with Sol Medium**. The comparison uses final 3D triangle IDs,
not just 2D outlines. **Use OpenRouter for Sol Medium** defaults to on; turn it
off to use OpenAI directly. **Save baseline and tooth results** controls those
two separate tasks, independently of the selected model's save switch.

**Parallel baseline views** and **Parallel model views** are independent
toggles, both on by default. The latter applies to normal and compared model runs.
Parallel mode sends all eight views as eight concurrent requests using the selected model,
thinking level, speed, image quality, and outline settings. Turn it off to use
the original single request with all eight images. Parallel mode repeats the
prompt in every request, so cost and results can differ; an eightfold speedup
is not guaranteed. Time is the elapsed time of the complete run, while token
usage and cost are summed across all requests. Saved labels and comparison
filenames include `Parallel`, and metadata records `parallelViews` and
`viewRequestCount`. A failed view prevents a partial result from being saved;
completed calls can still be charged. No automatic grouped retry is made.

The app saves baseline and comparison runs as normal JSON files under
`benchmark-output/<STL SHA-256>/`. Each file contains the returned outline JSON,
final triangle IDs, detected object count, model time, estimated API cost, and
quality metrics when compared. Reopening the same STL loads its saved files,
including normal runs when no baseline exists. Standalone runs store no
comparison metrics (`metrics: null`) and display a dash for comparison percentages. Replacing the baseline
clears comparison files made against the old baseline and preserves standalone
files in `runs/`. With a save switch off, neither its result file nor its raw
LLM-response JSON is written.

For OpenAI models selected through OpenRouter, **Speed** is separate from the
thinking level. Fast is available for all five listed OpenAI models. Ultrafast
is currently available for GPT-6.1 Sol. Unsupported speed options are disabled.
Fast token prices are 2x Standard (2.5x for GPT-5.5); Sol Ultrafast is 6x.

Fast/Ultrafast requests pin `openai/fast` or `openai/ultrafast`, disable provider
fallbacks, and verify the returned `service_tier`. An unconfirmed or unavailable
tier produces an error. The saved run label and filename include the speed;
JSON also records `processingSpeed`, `serviceTier`, and `reportedCostUsd`.
The app uses OpenRouter's reported cost when provided, with a tier-specific
estimate as a fallback. Existing saved runs continue to load.

## Code map

### Separate tooth-number estimate

After segmentation, **Estimate tooth numbers** makes a separate Sol Medium API
request with **one teeth-facing image** and its existing scanbody outlines. It uses the
OpenRouter/OpenAI baseline-provider toggle, independently of the comparison
model, and displays the estimated permanent-tooth FDI positions in its own
panel below Highlight. Numbers are not drawn on camera images. The segmentation
prompt, mesh selection, exports, and benchmark scores are unchanged. Only
scanbodies with outlines in the selected view are numbered. **Auto** chooses
between Top and Bottom using the number of existing scanbody detections, with
confidence breaking ties. This is a heuristic, not an anatomical orientation
guarantee. Preview the chosen image and override it with Top or Bottom if needed.
PCA may put an upper jaw's teeth-facing surface in the Bottom image. The endpoint
also filters out other views if an older browser sends them.

Optional jaw and patient-right hints help with orientation: PCA camera names
do not establish the patient's anatomical left/right. Unknown positions remain
unassigned with an explanation. FDI uses the two-digit tooth designation system
described by [ISO 3950](https://www.iso.org/standard/68292.html).

The button calls `/api/tooth-numbers` only when clicked. Its cost and time are
shown separately. **Save baseline and tooth results** controls these results, saved under
`benchmark-output/<STL SHA-256>/tooth-numbers/<timestamp>.json`, including the
actual returned JSON. It never replaces baseline/comparison files or the latest
segmentation debug JSON. No screenshot copies are stored in these files.

### Segmentation files

- `src/lib/capture-views.ts` renders the eight normal images and the hidden
  per-pixel triangle-ID images.
- `src/lib/triangle-selection.ts` fills an AI outline and reads the exact STL
  triangle IDs beneath it.
- `src/lib/stl-export.ts` writes selected or remaining triangles as binary STL
  while preserving the uploaded file's original coordinate system.
- `src/lib/ai-models.ts` is the shared model, reasoning, and
  paid-price list used by both the UI and API route.
- `src/lib/benchmark.ts` calculates triangle-level similarity to the saved Sol
  Medium baseline.
- `src/app/api/benchmarks/route.ts` saves and reloads benchmark JSON files from
  the project's `benchmark-output` folder.
- `src/components/ScanbodyWorkspace.tsx` loads the STL, combines selected
  triangle IDs, and colors those faces orange in the 3D viewer.
- `src/app/api/analyze/route.ts` sends the camera images to the vision model and
  validates its JSON outlines.

## What this proves

- A normal web app can inspect STL geometry and create its own camera views.
- The AI can return image-space regions instead of editing the STL directly.
- Those regions can be mapped back onto the real 3D mesh with raycasting.
- The model name is configuration, so the AI provider layer can be changed later.

## Important limitation

This is an ROI prototype, not clinically validated segmentation. Its orange highlight is an approximate surface region. Production scanbody replacement should add precise geometry fitting, such as RANSAC plus ICP, and require user confirmation before replacing or cutting mesh data.
