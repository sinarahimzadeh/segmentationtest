# Scanbody ROI Demo

A small local Next.js prototype that finds likely dental scanbodies in an STL scan.

## How it works

1. The STL is loaded in the browser.
2. PCA centers and consistently turns the model.
3. Three.js takes eight JPEG pictures from fixed angles.
4. LangChain sends those pictures to the selected OpenAI or Google Gemini model.
5. The API returns strict JSON outlines around likely scanbodies.
6. The app projects those outlines back onto the 3D mesh and highlights the matching areas.

The STL file itself stays in the browser. Only the rendered JPEG views are sent to the AI service.

## Setup

Create `.env.local` in this folder:

```env
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_google_ai_studio_key_here
SCANBODY_OUTLINE_POINTS=40
```

`SCANBODY_OUTLINE_POINTS` controls how many ordered edge points the model must
return for each detected scanbody. It accepts whole numbers from `3` to
`10000`; `40` is the default used for the first detection pass.

Then run:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Model benchmark

1. Choose an STL and run the fixed **GPT-6.1 Sol / Medium** baseline.
2. Select an OpenAI or Google Gemini model and thinking level.
3. Run the comparison. The app compares the final 3D triangle IDs, not just
   the 2D outlines.

The app saves baseline and comparison runs as normal JSON files under
`benchmark-output/<STL SHA-256>/`. Each file contains the returned outline JSON,
final triangle IDs, detected object count, model time, estimated API cost, and
quality metrics. Reopening the same STL loads its saved files. Replacing the
baseline clears comparison files made against the old baseline.

## Code map

- `src/lib/capture-views.ts` renders the eight normal images and the hidden
  per-pixel triangle-ID images.
- `src/lib/triangle-selection.ts` fills an AI outline and reads the exact STL
  triangle IDs beneath it.
- `src/lib/stl-export.ts` writes selected or remaining triangles as binary STL
  while preserving the uploaded file's original coordinate system.
- `src/lib/ai-models.ts` is the shared OpenAI and Gemini model, reasoning, and
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
