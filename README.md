# Scanbody ROI Demo

A small local Next.js prototype that finds likely dental scanbodies in an STL scan.

## How it works

1. The STL is loaded in the browser.
2. PCA centers and consistently turns the model.
3. Three.js takes eight JPEG pictures from fixed angles.
4. LangChain sends those pictures to an OpenAI vision model.
5. The API returns strict JSON outlines around likely scanbodies.
6. The app projects those outlines back onto the 3D mesh and highlights the matching areas.

The STL file itself stays in the browser. Only the rendered JPEG views are sent to the AI service.

## Setup

Create `.env.local` in this folder:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_VISION_MODEL=gpt-6.1-sol
SCANBODY_OUTLINE_POINTS=500
```

`SCANBODY_OUTLINE_POINTS` controls how many ordered edge points the model must
return for each detected scanbody. It accepts whole numbers from `3` to
`10000`; `500` is the default.

Then run:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## What this proves

- A normal web app can inspect STL geometry and create its own camera views.
- The AI can return image-space regions instead of editing the STL directly.
- Those regions can be mapped back onto the real 3D mesh with raycasting.
- The model name is configuration, so the AI provider layer can be changed later.

## Important limitation

This is an ROI prototype, not clinically validated segmentation. Its orange highlight is an approximate surface region. Production scanbody replacement should add precise geometry fitting, such as RANSAC plus ICP, and require user confirmation before replacing or cutting mesh data.
