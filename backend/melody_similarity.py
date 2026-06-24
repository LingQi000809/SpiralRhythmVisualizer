"""
Melody similarity: find K pairs of (input window, output window) that are
melodically similar — free start and end in both recordings.

Workflow
--------
  extract_notes          — audio → raw (onset, offset, pitch) events via basic-pitch
       │
  _flatten_polyphony     — collapse simultaneous notes to one melody note per moment
       │
  _to_intervals          — absolute pitches → semitone differences (key-invariant)
       │
  _cost_profile          — 2-D slide: for every output position × every input position
       │                   × every window size, compute DTW cost
  _dtw / _dtw_with_path  — inner kernel: equal-length DTW with Sakoe-Chiba band
       │
  find_similarity_matches — peak-pick K non-overlapping matches above min_similarity
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

# ── compatibility shim ─────────────────────────────────────────────────────────
# scipy.signal.gaussian was removed in scipy 1.12; basic-pitch 0.3.0 still
# calls it at import time, so patch it before importing basic-pitch.
import scipy.signal as _scipy_signal
if not hasattr(_scipy_signal, "gaussian"):
    _scipy_signal.gaussian = _scipy_signal.windows.gaussian

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import basic_pitch as _bp_pkg
    from basic_pitch.inference import predict as _bp_predict

import os as _os
# Use the TFLite model — the SavedModel format fails to load on TF 2.16+ / Python 3.12
# because optimizer slot variables changed; TFLite is self-contained.
_BP_MODEL_PATH = _os.path.join(
    _os.path.dirname(_bp_pkg.__file__),
    "saved_models", "icassp_2022", "nmp.tflite",
)

AudioPath  = Union[str, Path]
_NoteEvent = Tuple[float, float, int]  # (onset_sec, offset_sec, midi_pitch)


@dataclass(frozen=True)
class SimilarityMatch:
    input_start:  float   # onset of first matched input note, seconds
    input_end:    float   # offset of last matched input note, seconds
    output_start: float   # onset of first matched output note, seconds
    output_end:   float   # offset of last matched output note, seconds
    similarity:   float   # 1 / (1 + dtw_cost)  →  (0, 1],  1.0 = perfect match


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — pitch extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_notes(audio_path: AudioPath) -> List[_NoteEvent]:
    """
    Run Spotify basic-pitch on *audio_path* and return every detected note as
    (onset_sec, offset_sec, midi_pitch), sorted by onset time.

    basic-pitch runs a neural network on the raw audio to estimate a
    piano-roll; its output includes all voiced notes (including polyphonic
    ones).  Downstream steps may further filter/flatten these notes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # predict() returns (model_output_dict, PrettyMIDI, note_event_list).
        # Each element of note_event_list is
        #   (onset_sec, offset_sec, midi_pitch, amplitude, [pitch_bend]).
        _, _, note_events = _bp_predict(str(audio_path), _BP_MODEL_PATH)

    notes: List[_NoteEvent] = [
        (float(onset), float(offset), int(pitch))
        for onset, offset, pitch, *_ in note_events   # drop amplitude / bend
    ]
    notes.sort(key=lambda n: n[0])  # chronological order
    return notes


# ══════════════════════════════════════════════════════════════════════════════
# Step 1b — polyphony flattening  (optional but recommended)
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_polyphony(
    notes:     List[_NoteEvent],
    onset_tol: float = 0.05,   # seconds — notes closer than this count as simultaneous
) -> List[_NoteEvent]:
    """
    Reduce a polyphonic note list to a single melody line by keeping only the
    *highest-pitched* note from each group of simultaneously sounding notes.

    Why highest pitch?  In piano music the melody almost always sits in the
    topmost voice (soprano position).  This heuristic is simple, interpretable,
    and correct for the vast majority of piano repertoire.

    Alternatives (not implemented here):
      • Loudest note — requires amplitude data; useful when the melody is an
        inner voice (e.g. Alberti bass + melody in left hand).
      • Continuity tracking — prefer the note closest in pitch to the previous
        chosen note; avoids sudden jumps when the melody passes through a chord.
      • Weighted / soft — assign exponentially decreasing weights by pitch rank,
        use expected interval distance Σᵢ Σⱼ wᵢ wⱼ |ivᵢ − ivⱼ| in DTW.
        Theoretically cleanest but requires rewriting the DTW cost function.

    Parameters
    ----------
    onset_tol
        Two notes are considered simultaneous when their onsets are within
        `onset_tol` seconds.  50 ms is a good default for piano recordings
        (accounts for timing imprecision without merging legato passages).
    """
    if not notes:
        return notes

    melody: List[_NoteEvent] = []
    group: List[_NoteEvent] = [notes[0]]
    group_onset = notes[0][0]

    for note in notes[1:]:
        if note[0] - group_onset <= onset_tol:
            # Still within the same simultaneous group — collect.
            group.append(note)
        else:
            # Group closed: emit the highest-pitched note and start a new group.
            melody.append(max(group, key=lambda n: n[2]))
            group = [note]
            group_onset = note[0]

    melody.append(max(group, key=lambda n: n[2]))  # flush the final group
    return melody


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — semitone intervals  (transposition invariance)
# ══════════════════════════════════════════════════════════════════════════════

def _to_intervals(pitches: Sequence[int]) -> np.ndarray:
    """
    Convert a sequence of absolute MIDI pitches to semitone *differences*
    between consecutive notes.

    Example: [60, 64, 62]  →  [+4, -2]

    Why intervals?  A melody played in C major and the same melody in G major
    have different absolute pitches but identical interval sequences.  Working
    in interval space makes the matching transposition-invariant — a motif
    is recognised regardless of the key it was played in.
    """
    return np.diff(np.asarray(pitches, dtype=np.float32))
    # np.diff computes arr[i+1] - arr[i], i.e. the signed semitone step
    # between each consecutive pair.


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — local DTW on equal-length windows
# ══════════════════════════════════════════════════════════════════════════════

def _dtw(a: np.ndarray, b: np.ndarray, band_r: int) -> float:
    """
    Dynamic Time Warping between two *equal-length* interval sequences.

    DTW finds the monotone alignment path through an n×n grid that minimises
    the total element-wise distance, allowing each sequence to locally
    speed up or slow down to match the other.

    Parameters
    ----------
    a, b
        Interval sequences of the same length n.
    band_r
        Sakoe-Chiba band radius.  The alignment path is constrained to cells
        within `band_r` steps of the main diagonal.  This prevents absurd
        mappings (e.g. first note ↔ last note) and keeps computation O(n·r).

        band_r is computed from band_ratio as:
            band_r = max(2, int(window_length * band_ratio))
        band_ratio = 0.25 means the path may deviate at most ¼ of the window
        length away from the diagonal.  A tighter band (smaller ratio) demands
        near-uniform tempo; a looser band allows more time-stretching.

    Returns
    -------
    Normalised cost: total_accumulated_cost / (2 * n).
    Dividing by 2n makes scores comparable across windows of different sizes
    (the shortest valid path visits n steps → cost per step ≈ result).
    Returns inf if no valid path exists within the band.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return np.inf

    # dp[i, j] = minimum accumulated cost of aligning a[0..i] with b[0..j].
    # Initialised to inf; cells outside the band stay inf and are never used.
    dp = np.full((n, n), np.inf, dtype=np.float64)

    for i in range(n):
        # Only iterate over the Sakoe-Chiba band around the diagonal.
        for j in range(max(0, i - band_r), min(n, i + band_r + 1)):

            # Local cost: how different are these two interval values?
            d = abs(float(a[i]) - float(b[j]))

            if i == 0 and j == 0:
                # Base case: first cell has no predecessor.
                dp[0, 0] = d
            else:
                # Accumulate from the cheapest valid predecessor.
                # Three allowed moves that keep the path monotone:
                prev = np.inf
                if i > 0 and j > 0: prev = min(prev, dp[i-1, j-1])  # diagonal  — both advance
                if i > 0:           prev = min(prev, dp[i-1, j])     # vertical  — a advances, b repeats
                if j > 0:           prev = min(prev, dp[i,   j-1])   # horizontal— b advances, a repeats

                # If prev is still inf, we're at a band edge with no valid
                # predecessor — treat as a fresh start (cost = just d).
                dp[i, j] = d + (prev if np.isfinite(prev) else 0.0)

    total = dp[n-1, n-1]
    return float(total / (2 * n)) if np.isfinite(total) else np.inf


def _dtw_with_path(
    a: np.ndarray,
    b: np.ndarray,
    band_r: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Same as _dtw but also traces back and returns the optimal alignment path.

    The path is a list of (i, j) pairs from (0,0) to (n-1,n-1) describing
    which element of `a` aligns to which element of `b` at each step.
    Used for the warp-mapping visualisation.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return np.inf, []

    # --- forward pass: identical to _dtw ---
    dp = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        for j in range(max(0, i - band_r), min(n, i + band_r + 1)):
            d = abs(float(a[i]) - float(b[j]))
            if i == 0 and j == 0:
                dp[0, 0] = d
            else:
                prev = np.inf
                if i > 0 and j > 0: prev = min(prev, dp[i-1, j-1])
                if i > 0:           prev = min(prev, dp[i-1, j])
                if j > 0:           prev = min(prev, dp[i,   j-1])
                dp[i, j] = d + (prev if np.isfinite(prev) else 0.0)

    total = dp[n-1, n-1]
    cost = float(total / (2 * n)) if np.isfinite(total) else np.inf

    # --- traceback: walk back from (n-1, n-1) choosing the cheapest predecessor ---
    i, j = n - 1, n - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:   j -= 1                                         # at top edge: go left
        elif j == 0: i -= 1                                         # at left edge: go up
        else:
            # Pick the move that came from the lowest-cost predecessor.
            k = int(np.argmin([dp[i-1, j-1], dp[i-1, j], dp[i, j-1]]))
            if k == 0:   i -= 1; j -= 1   # diagonal
            elif k == 1: i -= 1            # vertical
            else:        j -= 1            # horizontal
        path.append((i, j))

    return cost, list(reversed(path))   # reverse so path runs (0,0)→(n-1,n-1)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3b — 2-D sliding window cost profile
# ══════════════════════════════════════════════════════════════════════════════

def _cost_profile(
    in_iv:      np.ndarray,
    out_iv:     np.ndarray,
    min_w:      int,
    max_w:      int,
    band_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every possible output start position j, try all window sizes w in
    [min_w, max_w] and all input start positions i, and record the (i, w) pair
    that gives the lowest DTW cost.

    This implements the "free ends" design: neither the input nor the output
    is required to be consumed fully.  We're looking for whatever sub-sequence
    of the output sounds most like some sub-sequence of the input.

    Parameters
    ----------
    in_iv, out_iv   Interval sequences from input and output recordings.
    min_w           Minimum window size in intervals (= min_notes − 1).
    max_w           Maximum window size (= min(len(in_iv), len(out_iv))).
    band_ratio      Passed through to _dtw to set the Sakoe-Chiba band width.
                    See _dtw docstring for details.

    Returns
    -------
    best_cost  float[n_out]   lowest DTW cost found at each output position
    best_in    int[n_out]     input start index for that best cost
    best_w     int[n_out]     window size for that best cost
    """
    n_in  = len(in_iv)
    n_out = len(out_iv)

    best_cost = np.full(n_out, np.inf, dtype=np.float64)
    best_in   = np.zeros(n_out, dtype=np.int32)
    best_w    = np.full(n_out, min_w, dtype=np.int32)

    for j in range(n_out):
        # Don't try windows that would overflow the output sequence.
        for w in range(min_w, min(max_w, n_out - j) + 1):
            out_win = out_iv[j : j + w]
            band_r  = max(2, int(w * band_ratio))
            # Slide input window of the same size over the input sequence.
            for i in range(n_in - w + 1):
                cost = _dtw(in_iv[i : i + w], out_win, band_r)
                if cost < best_cost[j]:
                    best_cost[j] = cost
                    best_in[j]   = i
                    best_w[j]    = w

    return best_cost, best_in, best_w


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — peak-pick → K matches
# ══════════════════════════════════════════════════════════════════════════════

def find_similarity_matches(
    input_audio:        AudioPath,
    output_audio:       AudioPath,
    *,
    K:                  int   = 5,
    min_notes:          int   = 4,     # minimum match length in notes
    min_similarity:     float = 0.0,   # 0–1; 0 = keep everything, 0.7 = keep only strong matches
    band_ratio:         float = 0.25,  # Sakoe-Chiba band = this fraction of the window length
    min_sep_ratio:      float = 0.5,   # after picking a match of width w, exclude w*ratio neighbours
    flatten_polyphony:  bool  = True,  # collapse simultaneous notes to highest pitch
    onset_tol:          float = 0.05,  # seconds — onset window for "simultaneous" notes
) -> List[SimilarityMatch]:
    """
    Full pipeline: audio → K melodically similar (input, output) window pairs.

    Steps
    -----
    1. Extract notes from both audio files via basic-pitch.
    2. Optionally flatten polyphony: keep only the highest-pitched note at
       each moment so the interval sequence represents the melody line.
    3. Convert pitch sequences to semitone interval sequences.
    4. Run _cost_profile to compute, for every output note position, the best
       matching input position and window size.
    5. Greedily pick K non-overlapping output positions above min_similarity,
       breaking ties by preferring the longest real-time span (longest-
       deterministic).

    Parameters
    ----------
    K               Maximum number of matches to return.
    min_notes       Minimum number of notes a match must span.
    min_similarity  Reject candidates with similarity < this value.
                    similarity = 1 / (1 + dtw_cost), so 0.7 corresponds to
                    an average interval mismatch of ~0.43 semitones per step.
    band_ratio      Controls the Sakoe-Chiba band in DTW.  0.25 = the
                    alignment path may deviate at most 25 % of the window
                    length from the diagonal.  Larger = allows more tempo
                    variation but slower and noisier.
    min_sep_ratio   After selecting a match of window size w, suppresses
                    output positions within w * min_sep_ratio steps.
                    Prevents the same musical moment being returned twice.
    flatten_polyphony
                    When True (default), run _flatten_polyphony before
                    computing intervals so simultaneous notes don't corrupt
                    the interval sequence.  Set False only if the input is
                    guaranteed monophonic.
    onset_tol       Passed to _flatten_polyphony; see its docstring.

    Returns
    -------
    List[SimilarityMatch] sorted by similarity descending (best match first).
    """
    # --- 1. Extract notes ---
    in_notes  = extract_notes(input_audio)
    out_notes = extract_notes(output_audio)

    # --- 2. Flatten polyphony ---
    if flatten_polyphony:
        in_notes  = _flatten_polyphony(in_notes,  onset_tol)
        out_notes = _flatten_polyphony(out_notes, onset_tol)

    if len(in_notes) < min_notes or len(out_notes) < min_notes:
        return []

    # --- 3. Semitone intervals ---
    in_iv  = _to_intervals([p for _, _, p in in_notes])
    out_iv = _to_intervals([p for _, _, p in out_notes])

    n_in  = len(in_iv)
    n_out = len(out_iv)
    min_w = min_notes - 1   # intervals = notes - 1
    max_w = min(n_in, n_out)

    if min_w > max_w:
        return []

    # --- 4. 2-D cost profile ---
    best_cost, best_in_pos, best_win = _cost_profile(
        in_iv, out_iv, min_w, max_w, band_ratio
    )

    # Convert DTW cost to similarity in [0, 1].
    def _sim(c: float) -> float:
        return 1.0 / (1.0 + c)

    # Helper: real-time duration of the output window at position j.
    def _out_duration(j: int) -> float:
        w = int(best_win[j])
        return out_notes[j + w][0] - out_notes[j][0]

    # --- 5. Greedy peak-pick ---
    # Sort candidates: descending similarity, then descending duration
    # (longest-deterministic: equal-score ties go to the longer match).
    candidates = sorted(
        [j for j in range(n_out)
         if np.isfinite(best_cost[j]) and _sim(best_cost[j]) >= min_similarity],
        key=lambda j: (-_sim(best_cost[j]), -_out_duration(j)),
    )

    used    = np.zeros(n_out, dtype=bool)
    matches: List[SimilarityMatch] = []

    for j in candidates:
        # Skip if this output position is already covered by a prior match.
        if used[max(0, j-1) : min(n_out, j+2)].any():
            continue

        i = int(best_in_pos[j])
        w = int(best_win[j])

        matches.append(SimilarityMatch(
            input_start=in_notes[i][0],
            input_end=in_notes[i + w][1],
            output_start=out_notes[j][0],
            output_end=out_notes[j + w][1],
            similarity=_sim(best_cost[j]),
        ))

        # Suppress nearby positions proportional to the selected window size.
        sep = max(1, int(w * min_sep_ratio))
        used[max(0, j - sep) : min(n_out, j + sep + 1)] = True

        if len(matches) >= K:
            break

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches
