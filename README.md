# Visualization — Experiment Sandbox

Experiment sandbox for the Hearmi spiral galaxy visualization. Mirrors the
component interfaces of `Hearmi-Frontend` so code can be migrated directly once
experiments are complete.

- **Input visualization** → `VisualizationWaitingView` (2D canvas spiral, synced with production)
- **Output visualization** → `StemVisualizationView` (concentric orbit rings per stem, replaces Three.js `VisualizationView`)
- **Backend** → FastAPI + Demucs for 4-stem separation

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10 or 3.12 (arm64) | Demucs requires ≥ 3.10 |
| Node.js | 18+ | |
| ffmpeg | any | `brew install ffmpeg` |

---

## Backend

### 1 — Create and activate a virtual environment

```bash
cd visualization/backend
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Core libraries used by the `/process` endpoint:

| Library | Purpose |
|---------|---------|
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `python-multipart` | multipart/form-data file uploads |
| `librosa` | audio feature extraction (onset, RMS, spectral centroid) |
| `soundfile` | encode stem audio to WAV for data URIs |
| `numpy` | array operations |
| `scikit-learn` | KMeans clustering (used by `/process-audio` endpoint) |
| `demucs` | 4-stem source separation (drums / bass / vocals / other) |

### 3 — Install Demucs

Demucs is not in `requirements.txt` because it pulls in large ML dependencies
(torch, torchaudio) separately from the rest. Install it once:

```bash
pip install demucs
```

> **GPU note:** Demucs runs on CPU locally — no GPU required, but a 3-minute
> song takes ~3–8 minutes on CPU depending on your machine.
> The production server runs the same command on GPU, which takes ~15 seconds.

### 4 — Install system dependency

```bash
brew install ffmpeg          # macOS
# sudo apt install ffmpeg    # Ubuntu/Debian
```

### 5 — Start the backend

```bash
# from visualization/backend, with venv activated
uvicorn audio_analysis:app --reload
```

The server runs on `http://localhost:8000`. You can verify it's up:

```
http://localhost:8000/docs
```

---

## Frontend

### 1 — Install dependencies

```bash
cd visualization
npm install
```

Key JS libraries:

| Library | Purpose |
|---------|---------|
| `react` + `react-dom` | UI framework |
| `meyda` | frame-wise spectral feature extraction (RMS, centroid) |
| `pitchy` | monophonic pitch detection |
| `vite` | build tool / dev server |

### 2 — Start the dev server

```bash
npm run dev
```

Opens at `http://localhost:5173`.

---

## Using the app

1. **Upload Input** — the audio you recorded or composed (simulates what the
   user provides to the AI model in production).
2. **Upload Output** — the audio you want to visualize as stems (simulates
   the AI model's generated output in production).
3. **Click Generate** — the input audio plays with the spiral visualization
   while the backend separates the output into 4 stems in the background.
4. **Auto-transition** — once stem analysis is complete, a cross-fade transitions
   to the output visualization showing 4 concentric orbit rings.
5. **Interact with stems** — click a ring or its label to solo it; click again
   to add another stem to the mix; click empty space to hear all stems.
   Drag a stem label to reorder the rings.

If you see **"Backend not reachable"** — the FastAPI server is not running.
Start it with `uvicorn audio_analysis:app --reload` as described above.

---

## Migration to production

| Experiment file | Replaces in Hearmi-Frontend |
|---|---|
| `src/components/VisualizationWaitingView.tsx` | `app/studio/components/VisualizationWaitingView.tsx` |
| `src/components/StemVisualizationView.tsx` | `app/studio/components/VisualizationView.tsx` |
| `src/utils/visDrawHelpers.ts` | inline helpers in `VisualizationWaitingView.tsx` |
| `src/utils/visMidiHelpers.ts` | `app/studio/utils/visMidiHelpers.ts` |
| `src/utils/api.ts` | already exists at `app/utils/api.ts` (change URL only) |

Steps:
1. Copy component files into `Hearmi-Frontend/app/studio/components/`.
2. Change imports: `../utils/api` → `../../utils/api`, `../utils/visDrawHelpers` → `../utils/visDrawHelpers`, etc.
3. Remove the `onVisualizationError` prop from `StemVisualizationView` (experiment-only).
4. The transition animation lives in the parent component (`OutputPanel` in production) — the individual view components have no transition logic of their own.
