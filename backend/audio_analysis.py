from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import asyncio
import dataclasses
import json
import os
import sys
import threading

import numpy as np
import librosa
import soundfile as sf
import io
import base64
import tempfile
import subprocess

from sklearn.cluster import KMeans

app = FastAPI()

# Allow requests from frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...), n_clusters: int = 4):
    try:

        print(f"n_clusters: {n_clusters}")

        # --- Load audio ---
        contents = await file.read()
        audio, sr = librosa.load(io.BytesIO(contents), sr=None, mono=True)
        hop_length = 512

        # --- Tempo ---
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(tempo)
        print(f"detected tempo: {tempo}")

        # --- Onset strength & detection ---
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=hop_length
        )
        print(f"detected onset strength: {onset_env}")

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )

        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        print(f"detected onset times: {onset_times}")

        if len(onset_frames) == 0:
            return {"tempo": tempo, "onsets": []}

        # --- Timbre features ---
        S = np.abs(librosa.stft(audio, hop_length=hop_length))

        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(S**2),
            sr=sr,
            n_mfcc=13,
            n_mels=128
        )

        centroid = librosa.feature.spectral_centroid(
            S=S,
            sr=sr
        )

        # --- Collect features at each onset ---
        feature_vectors = []
        for frame in onset_frames:
            # feature_vectors.append(mfcc[:, frame])
            feature_vectors.append(
                np.concatenate([
                    mfcc[:, frame],          # (13,)
                    [centroid[0, frame]]     # (1,)
                ]
            ))

        feature_vectors = np.array(feature_vectors)

        # --- Timbre clustering ---
        n_clusters = min(n_clusters, len(feature_vectors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(feature_vectors)

        # --- Final event list ---
        onsets = []
        for i, frame in enumerate(onset_frames):
            onsets.append({
                "time": float(onset_times[i]),
                "cluster": int(labels[i]),
                "centroid": float(centroid[0, frame])
            })

        return {
            "tempo": tempo,
            "onsets": onsets
        }

    except Exception as e:
        print(f"Failed to process audio: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


async def webm_to_wav(file) -> str:
    # Save uploaded webm
    tmp_webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tmp_webm.write(await file.read())
    tmp_webm.flush()

    # Create temporary wav file
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    # Convert webm to wav using ffmpeg
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_webm.name, tmp_wav.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    return tmp_wav.name

def normalize_array(arr, p_min=20, p_max=80):
    # RMS is roughly perceptual loudness, which is logarithmic. 
    arr = np.log1p(arr)  # log(1 + x) to avoid log(0)
    min_val = np.percentile(arr, p_min)
    max_val = np.percentile(arr, p_max)
    norm = (arr - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    return norm


def extract_voice_features(y, sr):
    """
    Extract musical features for spiral visualization:
    - Onsets
    - Pitch (midi)
    - RMS
    - Spectral centroid
    - Bandwidth
    - Flatness
    - Interval
    Normalized features are also included: rmsNorm, centroidNorm, bandwidthNorm, flatnessNorm
    """
    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Pitch tracking
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
    )
    midi = librosa.hz_to_midi(f0)
    midi = np.where(np.isnan(midi), 0, midi)

    # Spectral features
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # Normalize some features across the entire audio
    rms = normalize_array(rms)
    centroid = normalize_array(centroid)
    # bandwidth_norm = normalize_array(bandwidth)
    # flatness_norm = normalize_array(flatness)

    # Collect per-onset events
    events = []
    prev_pitch = None
    i = 0
    while i < len(onset_times):
        t = onset_times[i]
        idx = np.argmin(np.abs(frame_times - t))
        pitch = float(midi[idx])

        # Find duration until pitch changes or becomes unvoiced
        dur_idx = idx
        while dur_idx < len(midi) and abs(midi[dur_idx] - pitch) < 1 and midi[dur_idx] > 0:
            dur_idx += 1
        duration = float(frame_times[dur_idx - 1] - frame_times[idx]) if dur_idx > idx else 0.0

        interval = 0 if prev_pitch is None else pitch - prev_pitch
        prev_pitch = pitch

        events.append({
            "time": float(t),
            "pitch": pitch,
            "duration": duration,   
            "rms": float(rms[idx]),
            "centroid": float(centroid[idx]),
            "bandwidth": float(bandwidth[idx]),
            "flatness": float(flatness[idx]),
            "interval": float(interval),
        })

        # Skip any onsets that fall within this sustained pitch
        while i + 1 < len(onset_times) and onset_times[i + 1] <= frame_times[dur_idx - 1]:
            i += 1
        i += 1

    duration = len(y) / sr
    return {"duration": duration, "events": events}


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    wav_path = await webm_to_wav(file)
    y, sr = librosa.load(wav_path, sr=None)
    data = extract_voice_features(y, sr)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# /process  —  Stem separation + feature extraction for StemVisualizationView.
#
# Uses Demucs (python -m demucs) for 4-stem separation, matching production.
# Install: pip install demucs
#
# Response shape matches VisualizationData in Hearmi-Frontend/app/utils/api.ts
# exactly so the frontend can be migrated without changes.
# The `pos` field is a lightweight placeholder [time*0.5, rms*5, centroid*3];
# StemVisualizationView ignores pos and uses stemAudioUris for pitch analysis.
# ─────────────────────────────────────────────────────────────────────────────



def _extract_stem_frames(y: np.ndarray, sr: int, stem_name: str) -> list:
    """Extract per-onset VizFrames for one stem signal."""
    if y is None or len(y) == 0:
        return []
    hop = 512
    rms      = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    times    = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    rms_n = normalize_array(rms)
    cen_n = normalize_array(centroid)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    frames = []
    for fi in onset_frames:
        idx = min(int(fi), len(times) - 1)
        t, r, c = float(times[idx]), float(rms_n[idx]), float(cen_n[idx])
        frames.append({
            "pos":      [round(t * 0.5, 4), round(r * 5, 4), round(c * 3, 4)],
            "rms":      round(r, 4),
            "centroid": round(c, 4),
            "time":     round(t, 4),
            "stem":     stem_name,
        })
    return frames


@app.post("/process")
async def process_visualization(audio: UploadFile = File(...)):
    """
    4-stem separation with Demucs + per-stem feature extraction.
    Requires: pip install demucs
    """
    try:
        # Save upload to a temp WAV file
        contents = await audio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
            tmp_in.write(contents)
            in_path = tmp_in.name

        y_mix, sr = librosa.load(in_path, sr=None, mono=True)
        duration = float(len(y_mix) / sr)

        # --mp3 tells Demucs to encode output stems with ffmpeg instead of torchaudio,
        # avoiding the TorchCodec dependency in recent torchaudio versions.
        # The input format doesn't matter — we always pass a temp WAV to Demucs.
        import pathlib
        with tempfile.TemporaryDirectory() as out_dir:
            result = subprocess.run(
                ["python", "-m", "demucs", "-n", "htdemucs", "--mp3", "--out", out_dir, in_path],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                raise RuntimeError(f"Demucs failed: {result.stderr[-500:]}")

            # Demucs nests output: <out_dir>/<model>/<track_name>/<stem>.mp3
            stem_files = {p.stem: str(p) for p in pathlib.Path(out_dir).rglob("*.mp3")}
            if not stem_files:
                raise RuntimeError("Demucs produced no output files")

            all_frames: list = []
            stem_audio_uris: dict = {}

            for stem_name in ["drums", "bass", "vocals", "other"]:
                path = stem_files.get(stem_name)
                if not path:
                    continue
                y_stem, _ = librosa.load(path, sr=sr, mono=True)
                all_frames.extend(_extract_stem_frames(y_stem, sr, stem_name))
                # Encode the mp3 bytes directly — no WAV re-encode needed
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                stem_audio_uris[stem_name] = f"data:audio/mpeg;base64,{b64}"

        all_frames.sort(key=lambda f: f["time"])

        return {
            "frames":             all_frames,
            "duration":           round(duration, 3),
            "chords":             [],
            "key":                {"key_name": "Unknown", "key_root": 0, "key_mode": "major"},
            "songStructure":      [],
            "flamingoChords":     [],
            "vocalMelodyContour": [],
            "bassRootContour":    [],
            "audioDataUri":       "",
            "stemAudioUris":      stem_audio_uris,
        }

    except Exception as e:
        print(f"[/process] error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})


# ── /similarity ───────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time as _time
from melody_similarity import (  # noqa: E402
    find_similarity_matches, extract_notes, _flatten_polyphony, dump_debug_json,
)

_DEBUG_DIR = os.path.join(_HERE, "debug")
os.makedirs(_DEBUG_DIR, exist_ok=True)

@app.post("/similarity")
async def similarity_detection(
    input_audio:  UploadFile = File(...),
    output_audio: UploadFile = File(...),
):
    """
    Accepts two audio files, runs find_similarity_matches, and streams progress
    back as Server-Sent Events.

    Event shapes:
      {"type": "progress", "msg": "..."}   — one log line
      {"type": "done",     "matches": [...]}  — final result
      {"type": "error",    "msg": "..."}   — on exception
    """
    in_filename  = input_audio.filename  or "input.wav"
    out_filename = output_audio.filename or "output.wav"

    in_bytes  = await input_audio.read()
    out_bytes = await output_audio.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(in_bytes);  in_path = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(out_bytes); out_path = f.name

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _cb(msg: str) -> None:
        asyncio.run_coroutine_threadsafe(queue.put({"type": "progress", "msg": msg}), loop)

    def _run() -> None:
        try:
            # Extract notes once — reused by find_similarity_matches and debug JSON
            t0 = _time.perf_counter()
            raw_in  = extract_notes(in_path)
            raw_out = extract_notes(out_path)
            flat_in  = _flatten_polyphony(raw_in)
            flat_out = _flatten_polyphony(raw_out)
            t_extract = _time.perf_counter() - t0

            t1 = _time.perf_counter()
            matches = find_similarity_matches(
                in_path, out_path,
                K=5, min_similarity=0.6, polyphony="weighted",
                debug=False, progress_callback=_cb,
                raw_notes_in=raw_in, raw_notes_out=raw_out,
            )
            t_match = _time.perf_counter() - t1

            # Save debug JSON
            from datetime import datetime as _dt
            stamp      = _dt.now().strftime("%Y%m%d_%H%M%S")
            debug_path = os.path.join(_DEBUG_DIR, f"debug_{stamp}.json")
            dump_debug_json(
                debug_path,
                raw_in=raw_in,   raw_out=raw_out,
                flat_in=flat_in, flat_out=flat_out,
                matches=matches,
                input_audio=in_filename,
                output_audio=out_filename,
                latency={
                    "extract_notes_s":           t_extract,
                    "find_similarity_matches_s":  t_match,
                    "total_s":                   t_extract + t_match,
                },
            )

            payload = [dataclasses.asdict(m) for m in matches]
            asyncio.run_coroutine_threadsafe(queue.put({"type": "done", "matches": payload}), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put({"type": "error", "msg": str(exc)}), loop)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    threading.Thread(target=_run, daemon=True).start()

    async def _generate():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
