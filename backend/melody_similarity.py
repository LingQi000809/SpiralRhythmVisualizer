"""
Melody similarity detection between two audio files using MelodySim.

Finds all time segments where a snippet of the input audio is melodically
reproduced in the output audio, even under tempo or timbre variation.

Algorithm overview
------------------
1. Both audios are chunked into overlapping windows (default 5s / 2.5s hop).
2. Each window is encoded by MERT (a music-domain transformer) → 4-layer
   embedding of shape [3072, T].
3. The SiameseNet compares all (input_window, output_window) pairs and produces
   an N1×N2 similarity matrix where high values (→ 1) mean "same melody".
4. Matching segments appear as *diagonal stripes* in the matrix: a stripe at
   diagonal offset d = j - i means input windows starting at i align with
   output windows starting at j (same relative position within the shared
   excerpt, possibly at a different absolute time).
5. We scan every diagonal, threshold, find consecutive runs ≥ min_run_len
   windows, and convert each run to a SimilarityMatch with absolute timestamps.
"""

import os
import sys
import json
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from melodysim.config import config as _melodysim_config
from melodysim.model.module import LightningSiameseNet
from melodysim.data.extract_mert import run_mert_model_and_get_features

NATIVE_SR = 44100  # MelodySim's expected sample rate

_MATCH_COLORS: List[Tuple[int, int, int]] = [
    (255, 87,  87),   # red
    (87,  178, 255),  # blue
    (87,  255, 178),  # green
    (255, 200, 87),   # yellow
    (200, 87,  255),  # purple
    (255, 140, 87),   # orange
    (87,  255, 255),  # cyan
    (255, 87,  200),  # pink
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[LightningSiameseNet, AutoModel, Wav2Vec2FeatureExtractor]:
    """Load the SiameseNet checkpoint.

    The checkpoint stores both the siamese/classifier weights AND the MERT
    audio_model (frozen during training), so we load everything from it.
    audio_processor is reconstructed from the HuggingFace hub name (it is
    not an nn.Module so it was not saved in the state_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = LightningSiameseNet(_melodysim_config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"])
    module.to(device)
    module.eval()
    module.audio_model.eval()

    return module, module.audio_model, module.audio_processor


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """Load and mono-mix an audio file, resampling to NATIVE_SR."""
    wav, sr = torchaudio.load(path, normalize=True, channels_first=True)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != NATIVE_SR:
        wav = T.Resample(orig_freq=sr, new_freq=NATIVE_SR)(wav)
    return wav, NATIVE_SR


# ---------------------------------------------------------------------------
# Core: similarity matrix
# ---------------------------------------------------------------------------

def _window_slide(seq: torch.Tensor, window_len: int, hop_len: int) -> torch.Tensor:
    assert seq.dim() == 1
    if seq.shape[-1] >= window_len:
        return seq.unfold(-1, window_len, hop_len)
    return torch.nn.functional.pad(seq, (0, window_len - seq.shape[-1])).unsqueeze(0)


def _mert_encode_chunked(
    windows: torch.Tensor,
    audio_model: AutoModel,
    time_reduce: torch.nn.Module,
    mert_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run MERT on `windows` in chunks to bound peak memory on CPU."""
    chunks = []
    for start in range(0, windows.shape[0], mert_batch_size):
        batch = windows[start : start + mert_batch_size].to(device)
        chunks.append(run_mert_model_and_get_features(batch, audio_model, time_reduce).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def compute_similarity_matrix(
    module: LightningSiameseNet,
    audio_model: AutoModel,
    audio_processor: Wav2Vec2FeatureExtractor,
    waveform1: torch.Tensor,
    waveform2: torch.Tensor,
    window_len_sec: float = 10.0,
    hop_len_sec: float = 10.0,
    mert_batch_size: int = 4,
) -> Tuple[np.ndarray, float, float]:
    """
    Build the N1×N2 similarity matrix between all window pairs of two audio signals.

    Parameters
    ----------
    window_len_sec   Chunk length fed to MERT.  Matches original inference.py default
                     of 10s.  Shorter windows give finer timestamps but more N²
                     SiameseNet calls — use hop < window for overlap on GPU.
    hop_len_sec      Stride between windows.  Equal to window = no overlap (fast,
                     matches inference.py default).  Half-window = 50% overlap.
    mert_batch_size  Max windows per MERT forward pass.  Keeps memory bounded on CPU.
                     Increase on GPU (e.g. 16–32).

    Returns
    -------
    matrix   — np.ndarray shape (N1, N2), values in [0, 1] (1 = very similar)
    hop_sec  — hop length in seconds
    win_sec  — window length in seconds
    """
    mert_sr = audio_processor.sampling_rate
    device = module.device

    def _preprocess(wav: torch.Tensor) -> torch.Tensor:
        resampled = F_audio.resample(wav, NATIVE_SR, mert_sr)
        processed = audio_processor(resampled, sampling_rate=mert_sr, return_tensors="pt")
        return processed["input_values"].squeeze()

    w1 = _preprocess(waveform1)
    w2 = _preprocess(waveform2)

    window_samples = int(window_len_sec * mert_sr)
    hop_samples    = int(hop_len_sec    * mert_sr)

    p1 = _window_slide(torch.as_tensor(w1), window_samples, hop_samples)
    p2 = _window_slide(torch.as_tensor(w2), window_samples, hop_samples)
    N1, N2 = p1.shape[0], p2.shape[0]
    print(f"  [{N1} input windows × {N2} output windows = {N1*N2} comparisons]")

    time_reduce = torch.nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False).to(device)
    audio_model.to(device)

    # MERT encoding in chunks: (N, 4, frames, 768)
    feats1 = _mert_encode_chunked(p1, audio_model, time_reduce, mert_batch_size, device)
    feats2 = _mert_encode_chunked(p2, audio_model, time_reduce, mert_batch_size, device)

    # Reshape to SiameseNet input: (N, 4, 768, T) → (N, 3072, T)
    feats1 = feats1.permute(0, 1, 3, 2).flatten(start_dim=1, end_dim=2)
    feats2 = feats2.permute(0, 1, 3, 2).flatten(start_dim=1, end_dim=2)

    L1, L2 = feats1.shape[-1], feats2.shape[-1]
    max_len = max(83, L1, L2)
    if L1 < max_len:
        feats1 = torch.nn.functional.pad(feats1, (0, max_len - L1))
    if L2 < max_len:
        feats2 = torch.nn.functional.pad(feats2, (0, max_len - L2))

    # SiameseNet: score every (input_window, output_window) pair.
    # _inference_step returns a distance-like score (0=same, 1=different)
    # → 1 - score = similarity.
    # We process row-by-row (each row = one input window vs all N2 output windows)
    # to keep batch size at N2 rather than N1×N2.
    feats2_dev = feats2.to(device)
    sim = torch.zeros(N1, N2)
    for i in range(N1):
        sim[i] = 1.0 - module._inference_step(
            feats1[i : i + 1].repeat(N2, 1, 1).to(device),
            feats2_dev,
        )
        if (i + 1) % 5 == 0 or i == N1 - 1:
            print(f"  SiameseNet: {i+1}/{N1} rows done")

    return sim.cpu().numpy(), hop_len_sec, window_len_sec


# ---------------------------------------------------------------------------
# Core: diagonal stripe detection → SimilarityMatch list
# ---------------------------------------------------------------------------

def detect_matches(
    sim_matrix: np.ndarray,
    hop_len_sec: float,
    window_len_sec: float,
    threshold: float = 0.5,
    min_run_len: int = 2,
    max_run_len: Optional[int] = None,
) -> List[dict]:
    """
    Convert the similarity matrix into a list of SimilarityMatch dicts by
    scanning every diagonal for runs of above-threshold values.

    Why diagonals?
    A shared musical excerpt of length L appears as a diagonal stripe of L
    consecutive high-similarity cells.  The stripe's position on the i-axis
    (input) and j-axis (output) gives the absolute timestamps in each track.
    Different diagonals (offsets d = j - i) correspond to the excerpt appearing
    at different relative positions in the two tracks.

    Each returned dict matches the TypeScript interface:
        { inputStart, inputEnd, outputStart, outputEnd, label, rgb }
    where times are in seconds and rgb is an integer 3-tuple.
    """
    N1, N2 = sim_matrix.shape
    B = sim_matrix >= threshold

    matches: List[dict] = []
    color_idx = 0

    for d in range(-(N1 - 1), N2):
        # np.diagonal(B, offset=d) gives the diagonal at column-offset d
        diag_vals = np.diagonal(B, offset=d)
        if len(diag_vals) < min_run_len:
            continue

        # Row index of the first cell on this diagonal
        i_offset = max(0, -d)

        # Find runs of True
        runs: List[Tuple[int, int]] = []
        in_run = False
        run_start = 0
        for k, val in enumerate(diag_vals):
            if val and not in_run:
                in_run, run_start = True, k
            elif not val and in_run:
                in_run = False
                if k - run_start >= min_run_len:
                    runs.append((run_start, k - 1))
        if in_run and len(diag_vals) - run_start >= min_run_len:
            runs.append((run_start, len(diag_vals) - 1))

        for rs, re in runs:
            run_len = re - rs + 1
            if max_run_len is not None and run_len > max_run_len:
                continue

            i_start = i_offset + rs
            i_end   = i_offset + re
            j_start = i_start + d
            j_end   = i_end   + d

            avg_sim = float(
                np.array([sim_matrix[i_start + k, j_start + k] for k in range(run_len)]).mean()
            )

            matches.append({
                "inputStart":  round(i_start * hop_len_sec, 3),
                "inputEnd":    round(i_end   * hop_len_sec + window_len_sec, 3),
                "outputStart": round(j_start * hop_len_sec, 3),
                "outputEnd":   round(j_end   * hop_len_sec + window_len_sec, 3),
                "label":       f"match_{len(matches)+1} (sim={avg_sim:.2f}, windows={run_len})",
                "rgb":         list(_MATCH_COLORS[color_idx % len(_MATCH_COLORS)]),
                "similarity":  round(avg_sim, 4),
            })
            color_idx += 1

    return matches


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_similarity_detection(
    input_audio_path: str,
    output_audio_path: str,
    ckpt_path: str = "./model/siamese_net_20250328.ckpt",
    window_len_sec: float = 5.0,
    hop_len_sec: float = 2.5,
    threshold: float = 0.5,
    min_run_len: int = 2,
    device: Optional[torch.device] = None,
) -> List[dict]:
    """
    End-to-end: load models, compute similarity matrix, return SimilarityMatch list.

    Parameters
    ----------
    window_len_sec  Window size for each audio chunk fed to MERT (seconds).
                    Shorter windows give finer time resolution but less context
                    per window for the model. 5s works well for most music.
    hop_len_sec     Stride between windows.  Use hop < window for overlap (50%
                    overlap = hop = window/2) which improves boundary precision.
    threshold       Similarity score above which two windows are considered a
                    match.  0.5 is a good default; lower = more sensitive.
    min_run_len     Minimum number of consecutive matching window pairs required
                    to report a segment.  Filters short/noisy hits.
    """
    module, audio_model, audio_processor = load_models(ckpt_path, device)
    wav1, _ = load_audio(input_audio_path)
    wav2, _ = load_audio(output_audio_path)
    sim_matrix, hop, win = compute_similarity_matrix(
        module, audio_model, audio_processor,
        wav1, wav2, window_len_sec, hop_len_sec
    )
    return detect_matches(sim_matrix, hop, win, threshold, min_run_len)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect matching segments between two audio files")
    parser.add_argument("--input-audio",    required=True)
    parser.add_argument("--output-audio",   required=True)
    parser.add_argument("--ckpt-path",      required=True)
    parser.add_argument("--window-len-sec", type=float, default=5.0)
    parser.add_argument("--hop-len-sec",    type=float, default=2.5)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--min-run-len",    type=int,   default=2)
    args = parser.parse_args()

    matches = run_similarity_detection(
        args.input_audio, args.output_audio, args.ckpt_path,
        args.window_len_sec, args.hop_len_sec, args.threshold, args.min_run_len,
    )
    print(json.dumps(matches, indent=2))
