"""
Melody similarity analysis script.

Runs the full pipeline on any two audio files and logs:
  - latency for each step
  - every pairwise window comparison, sorted by similarity
  - detected matches above threshold
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from melody_similarity import load_models, load_audio, compute_similarity_matrix, detect_matches

# ── Config ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

CKPT_PATH    = os.path.join(_HERE, "melodysim", "model", "siamese_net_20250328.ckpt")
# INPUT_AUDIO  = os.path.join(_HERE, "..", "public", "data", "dragonBoyLofi.wav")
# INPUT_AUDIO  = os.path.join(_HERE, "..", "public", "data", "mono_poly.mp3")
INPUT_AUDIO = os.path.join(_HERE, "..", "public", "data", "piano1.wav")
OUTPUT_AUDIO = os.path.join(_HERE, "..", "public", "data", "piano2.wav")

WINDOW_LEN_SEC = 1.0
HOP_LEN_SEC    = 0.5
THRESHOLD      = 0.5
MIN_RUN_LEN    = 2
MAX_WIN_LEN    = 3

# Trim both files to this length for faster CPU runs (set None to use full audio)
TRIM_SEC = 180.0


def _trim(wav, trim_sec):
    max_samples = int(trim_sec * 44100)
    return wav[:, :max_samples] if wav.shape[-1] > max_samples else wav


def _ts(sec: float) -> str:
    """Format seconds as mm:ss for easy scrubbing in an audio player."""
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


if __name__ == "__main__":
    latency: dict[str, float] = {}

    print("=" * 60)
    print(f"Input  : {os.path.basename(INPUT_AUDIO)}")
    print(f"Output : {os.path.basename(OUTPUT_AUDIO)}")
    print(f"Window : {WINDOW_LEN_SEC}s / hop {HOP_LEN_SEC}s  |  threshold={THRESHOLD}  min_run={MIN_RUN_LEN}  max_run={MAX_WIN_LEN}")
    print("=" * 60)

    # ── Step 1: load models ────────────────────────────────────────────────
    t0 = time.perf_counter()
    module, audio_model, audio_processor = load_models(CKPT_PATH)
    latency["load_models"] = time.perf_counter() - t0
    print(f"\n[1] load_models          {latency['load_models']:6.2f}s")

    # ── Step 2: load audio ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    wav1, _ = load_audio(INPUT_AUDIO)
    wav2, _ = load_audio(OUTPUT_AUDIO)
    latency["load_audio"] = time.perf_counter() - t0
    if TRIM_SEC:
        wav1 = _trim(wav1, TRIM_SEC)
        wav2 = _trim(wav2, TRIM_SEC)
    dur1 = wav1.shape[-1] / 44100
    dur2 = wav2.shape[-1] / 44100
    print(f"[2] load_audio           {latency['load_audio']:6.2f}s  ({dur1:.1f}s input, {dur2:.1f}s output)")

    # ── Step 3: similarity matrix ──────────────────────────────────────────
    t0 = time.perf_counter()
    sim_matrix, hop, win = compute_similarity_matrix(
        module, audio_model, audio_processor,
        wav1, wav2, WINDOW_LEN_SEC, HOP_LEN_SEC,
    )
    latency["compute_similarity_matrix"] = time.perf_counter() - t0
    N1, N2 = sim_matrix.shape
    print(f"[3] compute_similarity   {latency['compute_similarity_matrix']:6.2f}s  ({N1}×{N2} = {N1*N2} pairs)")
    print(f"    range [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]  mean {sim_matrix.mean():.3f}")

    # ── Step 4: detect matches ─────────────────────────────────────────────
    t0 = time.perf_counter()
    matches = detect_matches(sim_matrix, hop, win,
                             threshold=THRESHOLD, min_run_len=MIN_RUN_LEN, max_run_len=MAX_WIN_LEN)
    latency["detect_matches"] = time.perf_counter() - t0
    print(f"[4] detect_matches       {latency['detect_matches']:6.2f}s  ({len(matches)} matches)")

    # ── Latency summary ────────────────────────────────────────────────────
    total = sum(latency.values())
    print(f"\n── Latency ──")
    for op, secs in latency.items():
        print(f"  {op:<28} {secs:6.2f}s")
    print(f"  {'TOTAL':<28} {total:6.2f}s")

    # ── All pairwise comparisons ───────────────────────────────────────────
    print(f"\n── All {N1*N2} window pairs (sorted by similarity, highest first) ──")
    pairs = []
    for i in range(N1):
        for j in range(N2):
            i_start = i * hop
            i_end   = i * hop + win
            j_start = j * hop
            j_end   = j * hop + win
            pairs.append((sim_matrix[i, j], i, j, i_start, i_end, j_start, j_end))
    pairs.sort(reverse=True)

    for sim, i, j, i_s, i_e, j_s, j_e in pairs:
        print(
            f"  [sim={sim:.4f}]"
            f"  in  [{_ts(i_s)}–{_ts(i_e)}] (win {i:2d})"
            f"  →  out [{_ts(j_s)}–{_ts(j_e)}] (win {j:2d})"
        )

    # ── Detected matches ───────────────────────────────────────────────────
    print(f"\n── Detected matches (threshold={THRESHOLD}, min_run={MIN_RUN_LEN}, max_run={MAX_WIN_LEN}) ──")
    if not matches:
        print("  (none)")
    for m in sorted(matches, key=lambda x: x.get("similarity", 0), reverse=True):
        i_s, i_e = m["inputStart"], m["inputEnd"]
        o_s, o_e = m["outputStart"], m["outputEnd"]
        print(
            f"  [sim={m.get('similarity', 0):.4f}]"
            f"  in  [{_ts(i_s)}–{_ts(i_e)}]"
            f"  →  out [{_ts(o_s)}–{_ts(o_e)}]"
            f"    {m['label']}"
        )

    # ── Visualisation ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    in_name  = os.path.basename(INPUT_AUDIO)
    out_name = os.path.basename(OUTPUT_AUDIO)
    fig.suptitle(f"MelodySim — {in_name} vs {out_name}", fontsize=13)

    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto",
                   vmin=0, vmax=1, origin="lower", extent=[0, N2, 0, N1])
    ax.set_title("Similarity matrix (row=input win, col=output win)")
    ax.set_xlabel("Output window index")
    ax.set_ylabel("Input window index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for m in matches:
        i0 = m["inputStart"]  / hop
        j0 = m["outputStart"] / hop
        length = min(
            round((m["inputEnd"]  - m["inputStart"]) / hop),
            round((m["outputEnd"] - m["outputStart"]) / hop),
        )
        rgb_norm = [c / 255.0 for c in m["rgb"]]
        ax.scatter([j0 + k + 0.5 for k in range(length)],
                   [i0 + k + 0.5 for k in range(length)],
                   color=rgb_norm, s=30, alpha=0.85, zorder=5)

    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    ax2.set_xlim(0, (N2 - 1) * hop + win)
    ax2.set_ylim(0, (N1 - 1) * hop + win)
    ax2.set_title("Matches in time (x=output, y=input)")
    ax2.set_xlabel("Output (s)")
    ax2.set_ylabel("Input (s)")
    ax2.grid(color="#374151", linewidth=0.5)
    for m in matches:
        rgb_norm = [c / 255.0 for c in m["rgb"]]
        rect = mpatches.FancyBboxPatch(
            (m["outputStart"], m["inputStart"]),
            m["outputEnd"] - m["outputStart"],
            m["inputEnd"]  - m["inputStart"],
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=rgb_norm, facecolor=(*rgb_norm, 0.25),
        )
        ax2.add_patch(rect)

    plt.tight_layout()
    out_path = os.path.join(_HERE, "test_melody_similarity_output.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")
