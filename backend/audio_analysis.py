from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import librosa
import io

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
async def process_audio(file: UploadFile = File(...)):
    try:
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
            feature_vectors.append(mfcc[:, frame])

        feature_vectors = np.array(feature_vectors)

        # --- Timbre clustering ---
        n_clusters = min(4, len(feature_vectors))
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
