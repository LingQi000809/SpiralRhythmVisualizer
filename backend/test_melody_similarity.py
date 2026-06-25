"""
Diagnostic test for melody_similarity.

Prints detected pitches, explains the cost profile,
and saves a visualization to dtw_debug.png.
Also renders four WAV files so you can listen to raw vs. flattened notes.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle as _Rect

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(_HERE, "..", "public", "data")

INPUT_AUDIO  = os.path.join(DATA, "compare1_hp.wav")
OUTPUT_AUDIO = os.path.join(DATA, "compare2_hp.wav")

K             = 10
MIN_SIM       = 0.3
NOTE_NAMES    = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

MATCH_COLORS = [
    "#e07b39",  # orange
    "#2ecc71",  # green
    "#9b59b6",  # purple
    "#e74c3c",  # red
    "#1abc9c",  # teal
    "#f39c12",  # amber
    "#e91e63",  # pink
]

def midi_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def _render_notes_to_wav(notes, path, sr=22050):
    """
    Synthesise note events as piano-like tones and write a 16-bit WAV.

    Uses four harmonics (fundamental + three overtones) for timbre and a
    piano-style ADSR envelope so each note sustains at full volume for its
    written duration before a short release tail — no artificial early decay.

    ADSR shape
    ----------
    Attack  10 ms  → ramp 0 → peak
    Decay   50 ms  → drop to SUSTAIN (80 %)
    Sustain        → hold at 80 % until note ends
    Release 120 ms → fade to 0 after note ends
    """
    import scipy.io.wavfile as wav_io
    if not notes:
        print(f"  (skipped — 0 notes): {os.path.basename(path)}")
        return
    end_t = max(off for _, off, _, _ in notes) + 0.5
    audio = np.zeros(int(end_t * sr), dtype=np.float32)

    ATK     = 0.010   # 10 ms attack
    DEC     = 0.050   # 50 ms decay to sustain
    SUS_LVL = 0.80    # sustain at 80 % of peak
    REL     = 0.120   # 120 ms release after note end

    for onset, offset, midi, amp in notes:
        freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        dur  = max(offset - onset, 0.05)
        n    = int((dur + REL) * sr)
        t    = np.arange(n) / sr

        # Piano timbre: fundamental + three overtones
        wave = (
            np.sin(2 * np.pi * 1 * freq * t) * 1.00 +
            np.sin(2 * np.pi * 2 * freq * t) * 0.50 +
            np.sin(2 * np.pi * 3 * freq * t) * 0.25 +
            np.sin(2 * np.pi * 4 * freq * t) * 0.10
        ) / 1.85  # normalise harmonic mix

        # ADSR envelope
        env      = np.zeros(n, dtype=np.float32)
        atk_n    = int(ATK * sr)
        dec_n    = int(DEC * sr)
        sus_end  = int(dur * sr)      # sustain runs until note-off
        rel_end  = min(n, sus_end + int(REL * sr))

        a_end = min(atk_n, sus_end)
        env[:a_end] = np.linspace(0.0, 1.0, a_end)
        if atk_n < sus_end:
            d_end = min(atk_n + dec_n, sus_end)
            env[atk_n:d_end] = np.linspace(1.0, SUS_LVL, d_end - atk_n)
            env[d_end:sus_end] = SUS_LVL
        env[sus_end:rel_end] = np.linspace(
            SUS_LVL if sus_end > 0 else 0.0, 0.0, rel_end - sus_end)

        sample = float(amp) * 0.35 * wave * env
        i0 = int(onset * sr)
        i1 = i0 + n
        if i1 > len(audio):
            audio = np.pad(audio, (0, i1 - len(audio)))
        audio[i0:i1] += sample

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.85
    wav_io.write(path, sr, (audio * 32767).astype(np.int16))
    print(f"  saved → {os.path.basename(path)}")


def _match_range(notes, t_start, t_end):
    """Return (start_idx, end_idx) of the notes covered by a match.

    t_start = onset of first matched note (seconds)
    t_end   = offset of last matched note (strictly > all matched onsets)
    """
    start_idx = None
    end_idx   = None
    for k, (o, _, _, _) in enumerate(notes):
        if start_idx is None and o >= t_start - 0.05:
            start_idx = k
        if o < t_end:
            end_idx = k
    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx < start_idx:
        end_idx = start_idx
    return start_idx, end_idx


def main() -> None:
    from melody_similarity import (
        extract_notes, _flatten_polyphony, _to_intervals,
        _dtw_with_path, find_similarity_matches, dump_debug_json,
    )
    from datetime import datetime as _dt

    _DEBUG_DIR = os.path.join(_HERE, "debug")
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    _stamp = _dt.now().strftime("%Y%m%d_%H%M%S")

    def _dbg(name: str) -> str:
        return os.path.join(_DEBUG_DIR, f"{_stamp}_{name}")

    # ── 1. Extract notes ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    in_notes_raw  = extract_notes(INPUT_AUDIO)
    out_notes_raw = extract_notes(OUTPUT_AUDIO)
    in_notes  = _flatten_polyphony(in_notes_raw)
    out_notes = _flatten_polyphony(out_notes_raw)
    t_extract = time.perf_counter() - t0
    print(f"extract_notes: {t_extract:.2f}s")
    print(f"  input:  {len(in_notes_raw)} raw notes → {len(in_notes)} after polyphony flattening")
    print(f"  output: {len(out_notes_raw)} raw notes → {len(out_notes)} after polyphony flattening")

    # ── 1b. Render WAV files for listening ────────────────────────────────────
    print("\nRendering WAV files for listening…")
    _render_notes_to_wav(in_notes_raw,  _dbg("input_raw.wav"))
    _render_notes_to_wav(in_notes,      _dbg("input_flat.wav"))
    _render_notes_to_wav(out_notes_raw, _dbg("output_raw.wav"))
    _render_notes_to_wav(out_notes,     _dbg("output_flat.wav"))

    in_pitches  = [p for _, _, p, _ in in_notes]
    out_pitches = [p for _, _, p, _ in out_notes]

    # print(f"\nInput  ({len(in_notes)} notes): {[midi_name(p) for p in in_pitches]}")
    # print(f"         MIDI: {in_pitches}")
    # print(f"Output ({len(out_notes)} notes): {[midi_name(p) for p in out_pitches]}")
    # print(f"         MIDI: {out_pitches}")

    # ── 2. Intervals ──────────────────────────────────────────────────────────
    in_intervals  = _to_intervals(in_pitches)
    out_intervals = _to_intervals(out_pitches)
    # print(f"\nInput  intervals ({len(in_intervals)}): {in_intervals.astype(int).tolist()}")
    # print(f"Output intervals ({len(out_intervals)}): {out_intervals.astype(int).tolist()}")

    # ── 3. Full match run ─────────────────────────────────────────────────────
    t2 = time.perf_counter()
    matches = find_similarity_matches(
        INPUT_AUDIO, OUTPUT_AUDIO, K=K,
        min_similarity=MIN_SIM, polyphony="weighted",
        debug=True,
        raw_notes_in=in_notes_raw,
        raw_notes_out=out_notes_raw,
    )
    t_match = time.perf_counter() - t2
    print(f"\nfind_similarity_matches (min_sim={MIN_SIM}): {t_match:.2f}s  →  {len(matches)} match(es)")
    for idx, m in enumerate(matches):
        print(f"  [{idx}] similarity={m.similarity:.4f}  "
              f"input=[{m.input_start:.2f}s, {m.input_end:.2f}s]  "
              f"output=[{m.output_start:.2f}s, {m.output_end:.2f}s]")

    # ── 3b. Debug JSON dump ───────────────────────────────────────────────────
    dump_debug_json(
        _dbg("similarity.json"),
        raw_in=in_notes_raw,  raw_out=out_notes_raw,
        flat_in=in_notes,     flat_out=out_notes,
        matches=matches,
        input_audio=INPUT_AUDIO, output_audio=OUTPUT_AUDIO,
        latency={
            "extract_notes_s":        t_extract,
            "find_similarity_matches_s": t_match,
            "total_s":                t_extract + t_match,
        },
    )

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    pitch_fmt = ticker.FuncFormatter(
        lambda v, _: midi_name(int(v)) if 0 <= int(v) < 128 else "")

    # ── (A) Input: raw detections vs. flattened melody ────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    raw_in_t = [n[0] for n in in_notes_raw]
    raw_in_p = [n[2] for n in in_notes_raw]
    ax.scatter(raw_in_t, raw_in_p, color="gray", s=18, alpha=0.45, zorder=1,
               label=f"Raw ({len(in_notes_raw)} notes)")
    in_times = [n[0] for n in in_notes]
    ax.step(in_times, in_pitches, where="post", color="#e07b39", linewidth=1.8, zorder=2)
    ax.scatter(in_times, in_pitches, color="#e07b39", s=45, zorder=3,
               label=f"Melody ({len(in_notes)} notes)")
    ax.legend(fontsize=8)
    ax.set_ylabel("MIDI pitch"); ax.set_xlabel("Time (s)")
    ax.set_title("Input: raw detections (gray) + melody after flattening (orange)")
    ax.yaxis.set_major_formatter(pitch_fmt); ax.grid(alpha=0.3)

    # ── (B) Output: raw detections vs. flattened melody ───────────────────────
    ax = fig.add_subplot(gs[0, 1])
    raw_out_t = [n[0] for n in out_notes_raw]
    raw_out_p = [n[2] for n in out_notes_raw]
    ax.scatter(raw_out_t, raw_out_p, color="gray", s=18, alpha=0.45, zorder=1,
               label=f"Raw ({len(out_notes_raw)} notes)")
    out_times = [n[0] for n in out_notes]
    ax.step(out_times, out_pitches, where="post", color="#4a90d9", linewidth=1.8, zorder=2)
    ax.scatter(out_times, out_pitches, color="#4a90d9", s=45, zorder=3,
               label=f"Melody ({len(out_notes)} notes)")
    ax.legend(fontsize=8)
    ax.set_ylabel("MIDI pitch"); ax.set_xlabel("Time (s)")
    ax.set_title("Output: raw detections (gray) + melody after flattening (blue)")
    ax.yaxis.set_major_formatter(pitch_fmt); ax.grid(alpha=0.3)

    # ── (C) DTW warp mapping — full sequences with ALL matches ────────────────
    ax = fig.add_subplot(gs[1, 0])
    if len(matches) > 0:
        all_p  = np.array(in_pitches + out_pitches, dtype=float)
        p_lo, p_hi = all_p.min(), all_p.max()
        def _scale(p): return (np.asarray(p, float) - p_lo) / (p_hi - p_lo + 1e-9)

        OFFSET = 1.4
        in_y  = _scale(in_pitches);  out_y = _scale(out_pitches) - OFFSET
        in_x  = np.linspace(0, 1, len(in_pitches))
        out_x = np.linspace(0, 1, len(out_pitches))

        ax.plot(in_x,  in_y,  color="#e07b39", alpha=0.18, linewidth=1.0,
                marker="o", markersize=3, zorder=1)
        ax.plot(out_x, out_y, color="#4a90d9", alpha=0.18, linewidth=1.0,
                marker="s", markersize=3, zorder=1)

        for m_idx, m in enumerate(matches):
            color   = MATCH_COLORS[m_idx % len(MATCH_COLORS)]
            is_best = (m_idx == 0)
            lw, ms, alpha = (2.4, 6, 1.0) if is_best else (1.4, 4, 0.65)

            i0, i1 = _match_range(in_notes,  m.input_start,  m.input_end)
            j0, j1 = _match_range(out_notes, m.output_start, m.output_end)

            ax.plot(in_x[i0:i1+1], in_y[i0:i1+1], color=color, linewidth=lw,
                    marker="o", markersize=ms, alpha=alpha, zorder=2,
                    label=f"Match {m_idx+1}  sim={m.similarity:.0%}")
            ax.plot(out_x[j0:j1+1], out_y[j0:j1+1], color=color, linewidth=lw,
                    marker="s", markersize=ms, alpha=alpha, zorder=2)

            n_in_intervals  = i1 - i0
            n_out_intervals = j1 - j0
            if n_in_intervals > 0 and n_out_intervals > 0:
                band_r = max(2, int(min(n_in_intervals, n_out_intervals) * 0.25))
                _, path = _dtw_with_path(in_intervals[i0:i0 + n_in_intervals],
                                         out_intervals[j0:j0 + n_out_intervals], band_r)
                la, llw = (0.55, 0.9) if is_best else (0.2, 0.5)
                for pi, pj in path:
                    ax.plot([in_x[i0 + pi], out_x[j0 + pj]],
                            [in_y[i0 + pi], out_y[j0 + pj]],
                            color=color, alpha=la, linewidth=llw, zorder=3)

        ax.text(0.01, 0.97, "Input",  transform=ax.transAxes, fontsize=8,
                color="#e07b39", va="top")
        ax.text(0.01, 0.03, "Output", transform=ax.transAxes, fontsize=8,
                color="#4a90d9", va="bottom")
        ax.set_title(f"DTW warp mapping — {len(matches)} match(es)\n"
                     "Faded = full sequence · Color = window · Lines = alignment")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No matches found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("DTW warp mapping — no match"); ax.axis("off")

    # ── (D) Full-sequence interval distance matrix + top-N match paths ──────────
    #
    # Background: D[i,j] = |in_intervals[i] − out_intervals[j]| (semitones)
    #   Yellow = 0 semitones difference between that pair of intervals
    #   Dark red = large difference (capped at 12 st for display)
    #
    # Colored rectangles = selected match windows in the full sequence space.
    # The line inside each rectangle = the DTW warp path for that match.
    #   Diagonal path → 1:1 note rate (same local tempo)
    #   More horizontal → output slower / more notes than input in this window
    #   More vertical   → input slower / more notes than output
    # ─────────────────────────────────────────────────────────────────────────
    N_SHOW = 3

    ax_D = fig.add_subplot(gs[1, 1])
    if len(in_intervals) > 0 and len(out_intervals) > 0:
        # Pairwise interval distance (n_in × n_out)
        D = np.abs(in_intervals[:, None].astype(float) - out_intervals[None, :].astype(float))

        cmap_D = plt.cm.YlOrRd.copy()
        cmap_D.set_bad(color="#e8e8e8")
        ax_D.imshow(D, origin="lower", aspect="auto",
                    cmap=cmap_D, vmin=0, vmax=12,
                    extent=[-0.5, len(out_intervals) - 0.5,
                            -0.5, len(in_intervals)  - 0.5])

        # Draw top-N match rectangles + paths
        for k in range(min(N_SHOW, len(matches))):
            m   = matches[k]
            col = MATCH_COLORS[k % len(MATCH_COLORS)]

            i0, i1 = _match_range(in_notes,  m.input_start,  m.input_end)
            j0, j1 = _match_range(out_notes, m.output_start, m.output_end)
            w = min(i1 - i0, j1 - j0)
            if w < 1:
                continue

            # Colored rectangle around the matched window
            ax_D.add_patch(_Rect(
                (j0 - 0.5, i0 - 0.5), w, w,
                linewidth=2.2, edgecolor=col, facecolor="none",
                zorder=3, label=f"Match {k+1}  sim={m.similarity:.0%}"))

            # DTW warp path for this window (global coordinates)
            band_r = max(2, int(w * 0.25))
            _, path = _dtw_with_path(in_intervals[i0:i0 + w], out_intervals[j0:j0 + w], band_r)
            px_g = [j0 + pj for _, pj in path]
            py_g = [i0 + pi for pi, _ in path]
            ax_D.plot(px_g, py_g, color=col, linewidth=2.0, zorder=4)

        ax_D.set_xlabel("Output interval index  →", fontsize=8)
        ax_D.set_ylabel("Input interval index  →", fontsize=8)
        ax_D.set_title(
            f"Full interval distance matrix + top-{min(N_SHOW, len(matches))} match paths\n"
            "Yellow = 0 st diff · Dark red = large diff · Box + line = match window + path",
            fontsize=8)
        if matches:
            ax_D.legend(fontsize=7, loc="upper right")
    else:
        ax_D.text(0.5, 0.5, "Not enough notes", ha="center", va="center",
                  transform=ax_D.transAxes)
        ax_D.axis("off")

    fig.suptitle("Melody Similarity — Debug View", fontsize=13, fontweight="bold")
    out_path = _dbg("dtw_debug.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nVisualization saved → {out_path}")


if __name__ == "__main__":
    main()
