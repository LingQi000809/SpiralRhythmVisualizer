from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import librosa
import io
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
