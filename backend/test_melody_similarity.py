"""
Diagnostic test for melody_similarity.

Prints detected pitches, explains the cost profile,
and saves a visualization to dtw_debug.png.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(_HERE, "..", "public", "data")

INPUT_AUDIO  = os.path.join(DATA, "piano1.wav")
OUTPUT_AUDIO = os.path.join(DATA, "piano2.wav")

K             = 5
MIN_SIM       = 0.7
NOTE_NAMES    = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def midi_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def main() -> None:
    from melody_similarity import (
        extract_notes, _flatten_polyphony, _to_intervals,
        _cost_profile, _dtw_with_path, find_similarity_matches,
    )

    # ── 1. Extract notes ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    in_notes_raw  = extract_notes(INPUT_AUDIO)
    out_notes_raw = extract_notes(OUTPUT_AUDIO)
    in_notes  = _flatten_polyphony(in_notes_raw)
    out_notes = _flatten_polyphony(out_notes_raw)
    print(f"extract_notes: {time.perf_counter() - t0:.2f}s")
    print(f"  input:  {len(in_notes_raw)} raw notes → {len(in_notes)} after polyphony flattening")
    print(f"  output: {len(out_notes_raw)} raw notes → {len(out_notes)} after polyphony flattening")

    in_pitches  = [p for _, _, p in in_notes]
    out_pitches = [p for _, _, p in out_notes]

    print(f"\nInput  ({len(in_notes)} notes): {[midi_name(p) for p in in_pitches]}")
    print(f"         MIDI: {in_pitches}")
    print(f"Output ({len(out_notes)} notes): {[midi_name(p) for p in out_pitches]}")
    print(f"         MIDI: {out_pitches}")

    # ── 2. Intervals ──────────────────────────────────────────────────────────
    in_iv  = _to_intervals(in_pitches)
    out_iv = _to_intervals(out_pitches)
    print(f"\nInput  intervals ({len(in_iv)}): {in_iv.astype(int).tolist()}")
    print(f"Output intervals ({len(out_iv)}): {out_iv.astype(int).tolist()}")

    min_w = 3
    max_w = min(len(in_iv), len(out_iv))

    # ── 3. Cost profile ───────────────────────────────────────────────────────
    t1 = time.perf_counter()
    best_cost, best_in_pos, best_win = _cost_profile(in_iv, out_iv, min_w, max_w, 0.25)
    best_sim = 1.0 / (1.0 + best_cost)
    print(f"\n_cost_profile: {time.perf_counter() - t1:.3f}s")
    print(f"  j  | similarity | dtw_cost | in_pos | window | pass?")
    for j in range(len(out_iv)):
        s    = best_sim[j] if np.isfinite(best_cost[j]) else 0.0
        flag = "✓" if s >= MIN_SIM else " "
        best = " <--" if np.isfinite(best_cost[j]) and best_cost[j] == best_cost[best_cost < np.inf].min() else ""
        print(f"  {j:2d} | {s:.4f}     | {best_cost[j]:.4f}   | {best_in_pos[j]:2d}     | {best_win[j]:2d}     | {flag}{best}")

    # ── 4. Full match run ─────────────────────────────────────────────────────
    t2 = time.perf_counter()
    matches = find_similarity_matches(
        INPUT_AUDIO, OUTPUT_AUDIO, K=K,
        min_similarity=MIN_SIM, flatten_polyphony=True,
    )
    print(f"\nfind_similarity_matches (min_sim={MIN_SIM}): {time.perf_counter()-t2:.2f}s  →  {len(matches)} match(es)")
    for idx, m in enumerate(matches):
        print(f"  [{idx}] similarity={m.similarity:.4f}  "
              f"input=[{m.input_start:.2f}s, {m.input_end:.2f}s]  "
              f"output=[{m.output_start:.2f}s, {m.output_end:.2f}s]")

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Melody Similarity — Debug View", fontsize=13, fontweight="bold")

    # ── (A) Pitch sequences ───────────────────────────────────────────────────
    ax = axes[0, 0]
    in_times  = [o for o, _, _ in in_notes]
    out_times = [o for o, _, _ in out_notes]
    ax.step(in_times,  in_pitches,  where="post", color="#e07b39",
            label=f"Input ({len(in_notes)} notes)", linewidth=1.5)
    ax.scatter(in_times,  in_pitches,  color="#e07b39", s=40, zorder=3)
    ax.step(out_times, out_pitches, where="post", color="#4a90d9",
            label=f"Output ({len(out_notes)} notes)", linewidth=1.5, linestyle="--")
    ax.scatter(out_times, out_pitches, color="#4a90d9", s=40, zorder=3)
    ax.set_ylabel("MIDI pitch")
    ax.set_xlabel("Time (s)")
    ax.set_title("Detected pitch sequences")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: midi_name(int(v)) if 0 <= int(v) < 128 else ""))
    ax.legend(); ax.grid(alpha=0.3)

    # ── (B) Interval sequences ────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(in_iv,  color="#e07b39", marker="o", label="Input intervals",
            linewidth=1.5, markersize=5)
    ax.plot(out_iv, color="#4a90d9", marker="s", label="Output intervals",
            linewidth=1.5, markersize=5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    if len(matches) > 0:
        m0 = matches[0]
        # find note indices for best match
        ii = next(k for k,(o,_,_) in enumerate(in_notes)  if abs(o - m0.input_start)  < 1e-3)
        jj = next(k for k,(o,_,_) in enumerate(out_notes) if abs(o - m0.output_start) < 1e-3)
        w0 = int(best_win[jj])
        ax.axvspan(ii - 0.4, ii + w0 - 0.6, alpha=0.15, color="#e07b39", label="Best match (input)")
        ax.axvspan(jj - 0.4, jj + w0 - 0.6, alpha=0.15, color="#4a90d9", label="Best match (output)")
    ax.set_ylabel("Semitone interval")
    ax.set_xlabel("Note index")
    ax.set_title("Semitone interval sequences (transposition-invariant)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── (C) DTW warp mapping — full sequences, matched window highlighted ────
    # Top curve  = full input pitch sequence  (orange, faded outside match)
    # Bottom curve = full output pitch sequence (blue,   faded outside match)
    # Black lines connect aligned notes inside the matched window.
    ax = axes[1, 0]
    if len(matches) > 0:
        m0 = matches[0]
        ii = next(k for k,(o,_,_) in enumerate(in_notes)  if abs(o - m0.input_start)  < 1e-3)
        jj = next(k for k,(o,_,_) in enumerate(out_notes) if abs(o - m0.output_start) < 1e-3)
        w0 = int(best_win[jj])

        # Normalize both pitch sequences to the same [0,1] scale
        all_p  = np.array(in_pitches + out_pitches, dtype=float)
        p_lo, p_hi = all_p.min(), all_p.max()
        def _scale(p): return (np.asarray(p, float) - p_lo) / (p_hi - p_lo + 1e-9)

        OFFSET = 1.4   # vertical gap between the two curves

        in_y  = _scale(in_pitches)
        out_y = _scale(out_pitches) - OFFSET

        # x positions: spread each sequence uniformly across [0,1]
        in_x  = np.linspace(0, 1, len(in_pitches))
        out_x = np.linspace(0, 1, len(out_pitches))

        # Draw full sequences (faded)
        ax.plot(in_x,  in_y,  color="#e07b39", alpha=0.25, linewidth=1.2,
                marker="o", markersize=4, zorder=1)
        ax.plot(out_x, out_y, color="#4a90d9", alpha=0.25, linewidth=1.2,
                marker="s", markersize=4, zorder=1)

        # Highlighted matched windows (full opacity)
        in_match_x  = in_x[ii : ii + w0 + 1]
        out_match_x = out_x[jj : jj + w0 + 1]
        ax.plot(in_match_x,  in_y[ii : ii + w0 + 1],  color="#e07b39", linewidth=2.5,
                marker="o", markersize=6, zorder=2,
                label=f"Input match  (notes {ii}–{ii+w0})")
        ax.plot(out_match_x, out_y[jj : jj + w0 + 1], color="#4a90d9", linewidth=2.5,
                marker="s", markersize=6, zorder=2,
                label=f"Output match (notes {jj}–{jj+w0})")

        # Warp lines: DTW path is on intervals [ii:ii+w0] / [jj:jj+w0]
        # Path pair (pi, pj) means interval pi aligns to interval pj,
        # which corresponds to the note AT that interval's start.
        band_r = max(2, int(w0 * 0.25))
        _, path = _dtw_with_path(in_iv[ii:ii+w0], out_iv[jj:jj+w0], band_r)
        for (pi, pj) in path:
            ax.plot([in_x[ii + pi], out_x[jj + pj]],
                    [in_y[ii + pi], out_y[jj + pj]],
                    color="black", alpha=0.4, linewidth=0.9, zorder=3)

        ax.set_title(f"DTW warp mapping  (similarity={m0.similarity:.2%})\n"
                     f"Faded = full sequence · Solid = matched window · Lines = alignment")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No matches found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("DTW warp mapping — no match")
        ax.axis("off")

    # ── (D) Similarity score per output position ──────────────────────────────
    # Each bar = best similarity achievable starting at that output note,
    # optimised over all input positions and window sizes.
    ax = axes[1, 1]
    finite = np.isfinite(best_cost)
    sim_vals = np.where(finite, best_sim, 0.0)
    colors = ["#5cb85c" if s >= MIN_SIM else "#aac4e8" for s in sim_vals]
    ax.bar(np.arange(len(sim_vals)), sim_vals, color=colors, edgecolor="none")
    ax.plot(np.arange(len(sim_vals)), sim_vals, color="#333", marker="o",
            markersize=5, linewidth=1.2)
    ax.axhline(MIN_SIM, color="red", linestyle="--", linewidth=1.2,
               label=f"Threshold ({MIN_SIM:.0%})")
    for m in matches:
        jj = next(k for k,(o,_,_) in enumerate(out_notes) if abs(o - m.output_start) < 1e-3)
        ax.axvline(jj, color="#e07b39", linestyle=":", linewidth=1.5, alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Output start position (note index)")
    ax.set_ylabel("Best similarity  [1/(1+dtw_cost)]")
    ax.set_title("Best similarity per output position\n"
                 "(green = above threshold, orange dotted = selected match)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(_HERE, "dtw_debug.png")
    plt.savefig(out_path, dpi=130)
    print(f"\nVisualization saved → {out_path}")


if __name__ == "__main__":
    main()
