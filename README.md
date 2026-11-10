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

## Model benchmark

1. Choose an STL and run the fixed **GPT-6.1 Sol / Medium** baseline. The
   **Use OpenRouter for Sol Medium** toggle is on by default; turn it off to
   use OpenAI directly. Changing the toggle applies to the next baseline run.
2. Select a model and thinking level. Grok 4.7, 4.6, and 4.3 use your xAI API key.
3. Run the comparison. The app compares the final 3D triangle IDs, not just
   the 2D outlines.

**Parallel baseline views** and **Parallel comparison views** are independent
toggles, both on by default. The comparison toggle is beside **Run and compare**.
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
quality metrics. Reopening the same STL loads its saved files. Replacing the
baseline clears comparison files made against the old baseline.

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
shown separately. **Save result files** also controls these results, saved under
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
