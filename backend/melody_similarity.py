"""
Melody similarity: find K pairs of (input window, output window) that are
melodically similar — free start and end in both recordings.

Workflow
--------
  extract_notes              audio → (onset, offset, pitch, amplitude) events
       │
  ┌────┴──────────────────────────────────────────────────────────┐
  │ polyphony="top"           │ polyphony="weighted"              │
  │ _flatten_polyphony        │ _group_notes                      │
  │  keep highest pitch       │  _assign_weights                  │
  │  per simultaneous group   │   pitch + loudness + continuity   │
  │ _to_intervals             │  _chord_interval_dists            │
  │  scalar semitone diffs    │   distribution over interval steps│
  │ _dtw  (scalar cost)       │  _dtw_weighted (expected distance)│
  └───────────────────────────┘───────────────────────────────────┘ 
       │                                                           
  _cost_profile / _cost_profile_weighted
       │
  find_similarity_matches  —  peak-pick K non-overlapping matches
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import time as _time

import numpy as np

# ── compatibility shim ────────────────────────────────────────────────────────
# scipy.signal.gaussian was removed in scipy 1.12; basic-pitch 0.3.0 still
# calls it at import time, so patch before importing.
import scipy.signal as _scipy_signal
if not hasattr(_scipy_signal, "gaussian"):
    _scipy_signal.gaussian = _scipy_signal.windows.gaussian

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import basic_pitch as _bp_pkg
    from basic_pitch.inference import predict as _bp_predict

import os as _os
_BP_MODEL_PATH = _os.path.join(
    _os.path.dirname(_bp_pkg.__file__),
    "saved_models", "icassp_2022", "nmp.tflite",
)

AudioPath  = Union[str, Path]
# Full note event — includes amplitude (0–1) so the weighted path can use it.
_NoteEvent = Tuple[float, float, int, float]  # onset_sec, offset_sec, midi_pitch, amplitude

# Weighted chord: [(midi_pitch, probability), ...], probabilities sum to 1.
_Chord   = List[Tuple[int, float]]
# Interval distribution: {semitone_interval: probability}.
_IntervalDist  = Dict[int, float]


_VIZ_CTX_S = 2.0  # seconds of context to show before/after each match in the panel

@dataclass(frozen=True)
class SimilarityMatch:
    input_start:  float   # onset of first matched input note, seconds
    input_end:    float   # offset of last matched input note, seconds
    output_start: float   # onset of first matched output note, seconds
    output_end:   float   # offset of last matched output note, seconds
    similarity:   float   # 1 / (1 + dtw_cost) — (0, 1], 1.0 = perfect match
    # Visualization data — top-voice note sequences with context for the panel
    in_note_times:    tuple = field(default=())  # note onsets relative to match start (s)
    in_note_pitches:  tuple = field(default=())  # MIDI pitch per note
    out_note_times:   tuple = field(default=())
    out_note_pitches: tuple = field(default=())
    in_match_slice:   tuple = field(default=(0, 0))   # (start_idx, end_idx) into in_note_* for match window
    out_match_slice:  tuple = field(default=(0, 0))   # same for out_note_*
    dtw_path:         tuple = field(default=())        # ((in_win_idx, out_win_idx), ...) within match window


def _ctx_notes_for_viz(
    notes: "List[_NoteEvent]",
    match_t0: float,
    match_t1: float,
) -> "Tuple[tuple, tuple, tuple]":
    """
    Extract the top-voice note sequence around a match window for visualization.

    Returns (times_rel, pitches, (match_start_idx, match_end_idx)) where times
    are in seconds relative to match_t0 (so context-before has negative times)
    and the slice indices mark which entries fall inside [match_t0, match_t1].
    """
    ctx = sorted(
        (on - match_t0, pitch)
        for on, off, pitch, _ in notes
        if on <= match_t1 + _VIZ_CTX_S and off >= match_t0 - _VIZ_CTX_S
    )
    if not ctx:
        return (), (), (0, 0)
    times   = tuple(t for t, _ in ctx)
    pitches = tuple(p for _, p in ctx)
    dur     = match_t1 - match_t0
    ms = next((k for k, (t, _) in enumerate(ctx) if t >= -0.01), 0)
    me = next((k for k, (t, _) in enumerate(ctx) if t >  dur + 0.01), len(ctx))
    return times, pitches, (ms, me)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — pitch extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_notes(audio_path: AudioPath) -> List[_NoteEvent]:
    """
    Run Spotify basic-pitch on *audio_path* and return every detected note as
    (onset_sec, offset_sec, midi_pitch, amplitude), sorted by onset.

    basic-pitch returns the amplitude of each note (0–1); we keep it so the
    weighted polyphony path can use loudness as one of its three scoring criteria.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, note_events = _bp_predict(str(audio_path), _BP_MODEL_PATH)
    # note_events: (onset, offset, pitch, amplitude, [pitch_bend])
    notes: List[_NoteEvent] = [
        (float(onset), float(offset), int(pitch), float(amp))
        for onset, offset, pitch, amp, *_ in note_events
    ]
    notes.sort(key=lambda n: n[0])
    return notes


# ══════════════════════════════════════════════════════════════════════════════
# Step 1b — pre-processing (applied before flattening / grouping)
# ══════════════════════════════════════════════════════════════════════════════

def _filter_short_notes(
    notes:   List[_NoteEvent],
    min_dur: float,
) -> List[_NoteEvent]:
    """
    Drop notes whose duration (offset − onset) is shorter than *min_dur* seconds.

    Short notes (typically < 80–100 ms) are often ornaments, grace notes, trills,
    or basic-pitch artefacts.  They add extra intervals to the sequence without
    contributing to the perceived melodic shape, which hurts DTW alignment.
    Setting min_dur=0.0 (the default) keeps all notes.
    """
    return [(on, off, p, a) for on, off, p, a in notes if off - on >= min_dur]


def _merge_unisons(notes: List[_NoteEvent]) -> List[_NoteEvent]:
    """
    Collapse consecutive same-pitch notes into a single longer note.

    This handles *note splitting*: a sustained pitch performed as two (or more)
    short attacks (common in MIDI quantisation, legato re-articulation, or
    basic-pitch's tendency to split long notes at low-confidence frames) becomes
    a single event.  The resulting interval sequence then matches an unsplit
    version, improving alignment robustness.

    Keeps the first note's onset, the last note's offset, and the maximum
    amplitude across the merged run.

    Only meaningful after polyphony has been flattened to a single melody line;
    for the weighted path this is a no-op because consecutive groups with the
    same dominant pitch are rare.
    """
    if not notes:
        return notes
    merged: List[_NoteEvent] = [notes[0]]
    for on, off, p, a in notes[1:]:
        prev_on, prev_off, prev_p, prev_a = merged[-1]
        if p == prev_p:
            merged[-1] = (prev_on, max(prev_off, off), prev_p, max(prev_a, a))
        else:
            merged.append((on, off, p, a))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Step 1c-top — polyphony flattening  (polyphony="top")
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_polyphony(
    notes:     List[_NoteEvent],
    onset_tol: float = 0.05,
) -> List[_NoteEvent]:
    """
    Group notes whose onsets are within *onset_tol* seconds (i.e. played
    simultaneously) and keep only the highest-pitched note from each group.

    This is the fast default: O(N), no extra bookkeeping.
    Works well for piano music where the melody sits in the top voice.
    """
    if not notes:
        return notes
    melody: List[_NoteEvent] = []
    group = [notes[0]]
    for note in notes[1:]:
        if note[0] - group[0][0] <= onset_tol:
            group.append(note)
        else:
            melody.append(max(group, key=lambda n: n[2]))  # highest pitch
            group = [note]
    melody.append(max(group, key=lambda n: n[2]))
    return melody


# ══════════════════════════════════════════════════════════════════════════════
# Step 1b-weighted — polyphony weighting  (polyphony="weighted")
# ══════════════════════════════════════════════════════════════════════════════

def _group_notes(
    notes:     List[_NoteEvent],
    onset_tol: float = 0.05,
) -> List[List[_NoteEvent]]:
    """
    Group simultaneous notes into chord groups without discarding any.
    Returns a list of groups; each group is a list of _NoteEvent tuples.
    """
    if not notes:
        return []
    groups: List[List[_NoteEvent]] = []
    current = [notes[0]]
    for note in notes[1:]:
        if note[0] - current[0][0] <= onset_tol:
            current.append(note)
        else:
            groups.append(current)
            current = [note]
    groups.append(current)
    return groups


def _assign_weights(
    groups:             List[List[_NoteEvent]],
    w_pitch:            float = 0.5,   # weight of pitch-height criterion
    w_loudness:         float = 0.3,   # weight of loudness (amplitude) criterion
    w_cont:             float = 0.2,   # weight of melodic-continuity criterion
    cont_sigma:         float = 12.0,  # semitones — how fast continuity decays with distance
    min_weight:         float = 0.1,   # drop notes whose normalised weight falls below this
    max_notes_per_chord: int  = 3,     # keep at most this many notes per group (0 = no cap)
) -> List[_Chord]:
    """
    Assign a probability distribution over notes within each chord group,
    then prune low-weight notes to bound the O(k²) cost in _chord_interval_dists.

    Three criteria are combined via a weighted sum, then normalised:

    1. Pitch height  — highest note scores 1.0, lowest scores 0.0 (rank-based).
    2. Loudness      — basic-pitch amplitude (0–1). Louder = more prominent.
    3. Continuity    — exp(−|pitch − prev_expected| / cont_sigma).
       "prev_expected" is the probability-weighted mean pitch of the previous chord.
       On the first chord, continuity is uniform (no prior context).

    After computing weights, two pruning steps are applied **in order**:

    1. min_weight threshold — any note whose normalised weight < min_weight is
       dropped.  E.g. 0.1 removes notes that contribute less than 10 % of the
       probability mass.  This eliminates clear non-melody notes while keeping
       genuinely ambiguous cases.

    2. max_notes_per_chord cap — if more than this many notes survive the
       threshold, keep only the top-weighted ones.  Hard-bounds the DTW cell
       cost at O(max_notes²).  Set to 0 to disable.

    After pruning, weights are re-normalised so they still sum to 1.
    The expected pitch used for the next chord's continuity score is computed
    from the pruned, re-normalised weights.

    Parameters
    ----------
    min_weight
        Floor for normalised note weight.  0.1 is a good default: a solo note
        always passes; in a 3-note chord the weakest note needs at least 10 %
        of the mass to survive.  Set to 0.0 to disable threshold pruning.
    max_notes_per_chord
        Hard cap on chord size after threshold pruning.  3 keeps the overhead
        at ≤9× vs the scalar path.  Set to 0 to disable cap pruning.
    """
    chords: List[_Chord] = []
    prev_exp: Optional[float] = None

    for group in groups:
        pitches = np.array([n[2] for n in group], dtype=float)
        amps    = np.array([n[3] for n in group], dtype=float)
        k       = len(group)

        # 1. Pitch height: rank-normalised to [0, 1]
        ranks   = np.argsort(pitches).argsort().astype(float)
        pitch_s = ranks / max(ranks.max(), 1.0)

        # 2. Loudness: min-max normalise amplitudes to [0, 1]
        a_lo, a_hi = amps.min(), amps.max()
        loud_s     = (amps - a_lo) / (a_hi - a_lo + 1e-9)

        # 3. Continuity: Gaussian proximity to previous expected pitch
        if prev_exp is not None:
            cont_s = np.exp(-np.abs(pitches - prev_exp) / cont_sigma)
        else:
            cont_s = np.ones(k)

        # Combine and first normalisation
        raw     = w_pitch * pitch_s + w_loudness * loud_s + w_cont * cont_s
        weights = raw / raw.sum()

        # ── Pruning step 1: drop notes below the weight floor ─────────────────
        if min_weight > 0.0 and k > 1:
            keep    = weights >= min_weight
            if keep.any():                   # always keep at least one note
                pitches = pitches[keep]
                weights = weights[keep]

        # ── Pruning step 2: cap at max_notes_per_chord ────────────────────────
        if max_notes_per_chord > 0 and len(weights) > max_notes_per_chord:
            top_idx = np.argpartition(weights, -max_notes_per_chord)[-max_notes_per_chord:]
            top_idx = top_idx[np.argsort(weights[top_idx])[::-1]]  # sort desc
            pitches = pitches[top_idx]
            weights = weights[top_idx]

        # Re-normalise after pruning
        weights = weights / weights.sum()

        chord: _Chord = [(int(pitches[i]), float(weights[i])) for i in range(len(pitches))]
        chords.append(chord)

        # Expected pitch for next group's continuity score
        prev_exp = float(sum(p * w for p, w in chord))

    return chords


def _chord_interval_dists(chords: List[_Chord]) -> List[_IntervalDist]:
    """
    Pre-compute an interval distribution for every consecutive chord pair.
    Returns len(chords) − 1 distributions.

    For chords A = [(p₁,w₁), (p₂,w₂)] and B = [(q₁,v₁), (q₂,v₂)]:
      interval q₁−p₁ gets weight w₁·v₁,
      interval q₁−p₂ gets weight w₂·v₁,  etc.
    All weights are re-normalised to sum to 1.

    Complexity: O(N · k²)  — fast for k ≤ 4.
    """
    dists: List[_IntervalDist] = []
    for a, b in zip(chords[:-1], chords[1:]):
        d: _IntervalDist = {}
        for pa, wa in a:
            for pb, wb in b:
                interval = pb - pa
                d[interval] = d.get(interval, 0.0) + wa * wb
        total = sum(d.values()) or 1.0
        dists.append({interval: w / total for interval, w in d.items()})
    return dists


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — semitone intervals  (top path only)
# ══════════════════════════════════════════════════════════════════════════════

def _to_intervals(pitches: Sequence[int]) -> np.ndarray:
    """
    Convert absolute MIDI pitches to semitone differences between consecutive
    notes.  Example: [60, 64, 62] → [+4, −2].

    Makes matching transposition-invariant: a motif in C and the same motif
    in G have different pitches but identical interval sequences.
    """
    return np.diff(np.asarray(pitches, dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — DTW kernels
# ══════════════════════════════════════════════════════════════════════════════

def _dtw(a: np.ndarray, b: np.ndarray, band_r: int) -> float:
    """
    DTW between two equal-length scalar interval sequences.

    dp[i, j] = minimum accumulated cost to align a[0..i] with b[0..j].
    Three allowed predecessors keep the path monotone:
      (i−1, j−1) diagonal   — both sequences advance one step
      (i−1, j  ) vertical   — a advances, b repeats (local slowdown in b)
      (i,   j−1) horizontal — b advances, a repeats (local slowdown in a)

    The Sakoe-Chiba band (|i−j| ≤ band_r) prevents the path from drifting far
    from the diagonal, ruling out absurd alignments like "first note ↔ last note".
    band_r = max(2, int(window_length × band_ratio)).

    Returns cost / (2n) so scores are comparable across window sizes.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return np.inf

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
    return float(total / (2 * n)) if np.isfinite(total) else np.inf


def _dtw_with_path(
    a: np.ndarray, b: np.ndarray, band_r: int
) -> Tuple[float, List[Tuple[int, int]]]:
    """Like _dtw but also traces back the optimal warping path for visualisation."""
    n = len(a)
    if n == 0 or n != len(b):
        return np.inf, []

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
    cost  = float(total / (2 * n)) if np.isfinite(total) else np.inf

    i, j  = n - 1, n - 1
    path  = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:   j -= 1
        elif j == 0: i -= 1
        else:
            k = int(np.argmin([dp[i-1, j-1], dp[i-1, j], dp[i, j-1]]))
            if k == 0:   i -= 1; j -= 1
            elif k == 1: i -= 1
            else:        j -= 1
        path.append((i, j))
    return cost, list(reversed(path))


def _expected_interval_distance(da: _IntervalDist, db: _IntervalDist) -> float:
    """
    Expected absolute interval difference between two distributions:
      Σ_{a,b} P(a) · P(b) · |a − b|

    This is the 1-D Wasserstein-1 (Earth Mover's) distance.
    Implemented as a numpy outer product — much faster than nested Python loops.

    Complexity per call: O(|da| · |db|) ≈ O(k²) where k = chord size.
    """
    ivs_a = np.fromiter(da.keys(),   dtype=np.float32)
    wts_a = np.fromiter(da.values(), dtype=np.float32)
    ivs_b = np.fromiter(db.keys(),   dtype=np.float32)
    wts_b = np.fromiter(db.values(), dtype=np.float32)
    # Outer product: each (a, b) pair contributes w_a · w_b · |a − b|
    return float(np.sum(wts_a[:, None] * wts_b[None, :] *
                        np.abs(ivs_a[:, None] - ivs_b[None, :])))


def _dtw_weighted(
    a_dists: List[_IntervalDist],
    b_dists: List[_IntervalDist],
    band_r:  int,
) -> float:
    """
    DTW where local cost = expected absolute interval difference (Wasserstein-1).

    Structurally identical to _dtw; only the cost function changes.
    This lets the matching consider ALL simultaneous notes, weighted by their
    probability of carrying the melody.

    Overhead vs _dtw: O(k²) per cell instead of O(1), where k = chord size.
    For k=3 this is ≈9× per cell; with numpy vectorisation the real factor is
    closer to 3–5×.
    """
    n = len(a_dists)
    if n == 0 or n != len(b_dists):
        return np.inf

    dp = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        for j in range(max(0, i - band_r), min(n, i + band_r + 1)):
            d = _expected_interval_distance(a_dists[i], b_dists[j])
            if i == 0 and j == 0:
                dp[0, 0] = d
            else:
                prev = np.inf
                if i > 0 and j > 0: prev = min(prev, dp[i-1, j-1])
                if i > 0:           prev = min(prev, dp[i-1, j])
                if j > 0:           prev = min(prev, dp[i,   j-1])
                dp[i, j] = d + (prev if np.isfinite(prev) else 0.0)

    total = dp[n-1, n-1]
    return float(total / (2 * n)) if np.isfinite(total) else np.inf


# ══════════════════════════════════════════════════════════════════════════════
# Step 3b — 2-D sliding window cost profile  (both paths)
# ══════════════════════════════════════════════════════════════════════════════

def _cost_profile(
    in_intervals:      np.ndarray,
    out_intervals:     np.ndarray,
    min_w:      int,
    max_w:      int,
    band_ratio: float,
    _log:       Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every output start position j, try all window sizes w ∈ [min_w, max_w]
    and all input start positions i; record the (i, w) with the lowest scalar DTW cost.

    "Free ends": neither the full input nor the full output needs to be consumed —
    we find whatever sub-sequence of the output best matches some sub-sequence of
    the input.

    Returns (best_cost[n_out], best_in_pos[n_out], best_win[n_out]).
    """
    n_in, n_out = len(in_intervals), len(out_intervals)
    best_cost = np.full(n_out, np.inf, dtype=np.float64)
    best_in   = np.zeros(n_out, dtype=np.int32)
    best_w    = np.full(n_out, min_w, dtype=np.int32)

    for j in range(n_out):
        if _log and j % 20 == 0:
            _log(f"  [progress] DTW (top):      {j:4d}/{n_out} output positions")
        for w in range(min_w, min(max_w, n_out - j) + 1):
            out_win = out_intervals[j : j + w]
            band_r  = max(2, int(w * band_ratio))
            for i in range(n_in - w + 1):
                cost = _dtw(in_intervals[i : i + w], out_win, band_r)
                if cost < best_cost[j]:
                    best_cost[j] = cost; best_in[j] = i; best_w[j] = w
    if _log:
        _log(f"  [progress] DTW (top):      {n_out:4d}/{n_out} output positions — done")

    return best_cost, best_in, best_w


def _cost_profile_weighted(
    in_dists:   List[_IntervalDist],
    out_dists:  List[_IntervalDist],
    min_w:      int,
    max_w:      int,
    band_ratio: float,
    _log:       Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as _cost_profile but operates on interval distributions (_IntervalDist)
    and uses _dtw_weighted as the inner kernel.

    Overhead vs _cost_profile: O(k²) per DTW cell (k = avg chord size).
    """
    n_in, n_out = len(in_dists), len(out_dists)
    best_cost = np.full(n_out, np.inf, dtype=np.float64)
    best_in   = np.zeros(n_out, dtype=np.int32)
    best_w    = np.full(n_out, min_w, dtype=np.int32)

    for j in range(n_out):
        if _log and j % 20 == 0:
            _log(f"  [progress] DTW (weighted): {j:4d}/{n_out} output positions")
        for w in range(min_w, min(max_w, n_out - j) + 1):
            out_win = out_dists[j : j + w]
            band_r  = max(2, int(w * band_ratio))
            for i in range(n_in - w + 1):
                cost = _dtw_weighted(in_dists[i : i + w], out_win, band_r)
                if cost < best_cost[j]:
                    best_cost[j] = cost; best_in[j] = i; best_w[j] = w
    if _log:
        _log(f"  [progress] DTW (weighted): {n_out:4d}/{n_out} output positions — done")

    return best_cost, best_in, best_w


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — peak-pick → K matches
# ══════════════════════════════════════════════════════════════════════════════

def find_similarity_matches(
    input_audio:   AudioPath,
    output_audio:  AudioPath,
    *,
    K:              int   = 5,
    min_notes:      int   = 4,
    min_similarity: float = 0.0,
    band_ratio:     float = 0.25,
    min_sep_ratio:  float = 0.5,
    polyphony:      str   = "top",   # "top" | "weighted"
    onset_tol:      float = 0.05,
    # weighted-path tuning — ignored when polyphony="top"
    w_pitch:             float = 0.5,
    w_loudness:          float = 0.3,
    w_continuity:        float = 0.2,
    cont_sigma:          float = 12.0,
    min_weight:          float = 0.3,   # drop notes whose weight < this after normalisation
    max_notes_per_chord: int   = 5,     # hard cap on chord size (0 = no cap)
    # pre-processing — applied to raw notes before flattening / grouping
    min_note_dur:   float = 0.0,   # drop notes shorter than this (seconds); 0 = keep all
    merge_unisons:  bool  = True,  # collapse split notes (consecutive same-pitch → one note)
    debug:          bool  = True,  # print cost-profile table + progress to stdout
    progress_callback: Optional[Callable[[str], None]] = None,  # called with each log line (overrides debug prints when set)
    # pass pre-extracted notes to skip re-running basic-pitch
    raw_notes_in:   Optional[List[_NoteEvent]] = None,
    raw_notes_out:  Optional[List[_NoteEvent]] = None,
) -> List[SimilarityMatch]:
    """
    Full pipeline: audio → K melodically similar (input, output) window pairs.

    Parameters
    ----------
    polyphony
        "top"      Fast path.  Collapses simultaneous notes to the highest-
                   pitched one before computing intervals.  O(1) per DTW cell.
        "weighted" Soft path.  All simultaneous notes are kept, weighted by
                   pitch height + loudness + melodic continuity.  The DTW cost
                   is the expected interval distance (Wasserstein-1) under the
                   joint distribution.  O(k²) per DTW cell (k = chord size).
    w_pitch, w_loudness, w_continuity
        Criterion weights for the weighted path (need not sum to 1).
    cont_sigma
        Continuity decay in semitones: a note cont_sigma semitones away from
        the previous expected pitch gets weight exp(−1) ≈ 0.37.
    band_ratio
        Sakoe-Chiba band = band_ratio × window_length.  Controls how much
        tempo variation is allowed.  0.25 = ±25 % of the window can be
        time-stretched in either direction.
    min_sep_ratio
        After selecting a match of width w, suppress output positions within
        w × min_sep_ratio steps to prevent double-counting.
    """
    _log: Optional[Callable[[str], None]] = (
        progress_callback if progress_callback is not None
        else (print if debug else None)
    )

    if raw_notes_in is not None:
        raw_in = raw_notes_in
        if _log: _log(f"  [timing] basic-pitch (input):  skipped (pre-extracted, {len(raw_in)} notes)")
    else:
        _t0 = _time.perf_counter()
        if _log: _log("Extracting pitches from input audio...")
        raw_in = extract_notes(input_audio)
        if _log: _log(f"  [timing] basic-pitch (input):  {_time.perf_counter() - _t0:.2f}s  ({len(raw_in)} notes)")

    if raw_notes_out is not None:
        raw_out = raw_notes_out
        if _log: _log(f"  [timing] basic-pitch (output): skipped (pre-extracted, {len(raw_out)} notes)")
    else:
        _t0 = _time.perf_counter()
        if _log: _log("Extracting pitches from output audio...")
        raw_out = extract_notes(output_audio)
        if _log: _log(f"  [timing] basic-pitch (output): {_time.perf_counter() - _t0:.2f}s  ({len(raw_out)} notes)")
    _t_total = _time.perf_counter()

    # ── pre-processing ────────────────────────────────────────────────────────
    if _log: _log("Flattening / weighting polyphony...")
    if min_note_dur > 0:
        raw_in  = _filter_short_notes(raw_in,  min_note_dur)
        raw_out = _filter_short_notes(raw_out, min_note_dur)

    # ── prepare sequences depending on polyphony mode ─────────────────────────
    if polyphony == "top":
        in_notes  = _flatten_polyphony(raw_in,  onset_tol)
        out_notes = _flatten_polyphony(raw_out, onset_tol)
        if merge_unisons:
            in_notes  = _merge_unisons(in_notes)
            out_notes = _merge_unisons(out_notes)
        if len(in_notes) < min_notes or len(out_notes) < min_notes:
            return []

        in_intervals  = _to_intervals([p for _, _, p, _ in in_notes])
        out_intervals = _to_intervals([p for _, _, p, _ in out_notes])
        n_in, n_out = len(in_intervals), len(out_intervals)
        min_w, max_w = min_notes - 1, min(n_in, n_out)
        if min_w > max_w:
            return []

        _t0 = _time.perf_counter()
        if _log: _log(f"Computing DTW cost profile (top-note, {n_out} output positions)...")
        best_cost, best_in_pos, best_win = _cost_profile(
            in_intervals, out_intervals, min_w, max_w, band_ratio, _log=_log)
        if _log: _log(f"  [timing] DTW cost profile (top, {n_out} positions): {_time.perf_counter() - _t0:.2f}s")

        # timing arrays — onset/offset of the flattened notes
        in_times  = [(n[0], n[1]) for n in in_notes]
        out_times = [(n[0], n[1]) for n in out_notes]
        _viz_in   = in_notes
        _viz_out  = out_notes

    elif polyphony == "weighted":
        in_groups  = _group_notes(raw_in,  onset_tol)
        out_groups = _group_notes(raw_out, onset_tol)
        if len(in_groups) < min_notes or len(out_groups) < min_notes:
            return []

        in_chords  = _assign_weights(in_groups,  w_pitch, w_loudness, w_continuity, cont_sigma, min_weight, max_notes_per_chord)
        out_chords = _assign_weights(out_groups, w_pitch, w_loudness, w_continuity, cont_sigma, min_weight, max_notes_per_chord)

        # Interval distributions between consecutive chord groups
        in_dists   = _chord_interval_dists(in_chords)
        out_dists  = _chord_interval_dists(out_chords)
        n_in, n_out = len(in_dists), len(out_dists)
        min_w, max_w = min_notes - 1, min(n_in, n_out)
        if min_w > max_w:
            return []

        _t0 = _time.perf_counter()
        if _log: _log(f"Computing DTW cost profile (weighted, {n_out} output positions)...")
        best_cost, best_in_pos, best_win = _cost_profile_weighted(
            in_dists, out_dists, min_w, max_w, band_ratio, _log=_log)
        if _log: _log(f"  [timing] DTW cost profile (weighted, {n_out} positions): {_time.perf_counter() - _t0:.2f}s")

        # timing: onset = earliest note in group, offset = latest offset in group
        in_times  = [(min(n[0] for n in g), max(n[1] for n in g)) for g in in_groups]
        out_times = [(min(n[0] for n in g), max(n[1] for n in g)) for g in out_groups]
        # top voice per group for visualization
        _viz_in  = [(min(n[0] for n in g), max(n[1] for n in g), max(n[2] for n in g), 0.0) for g in in_groups]
        _viz_out = [(min(n[0] for n in g), max(n[1] for n in g), max(n[2] for n in g), 0.0) for g in out_groups]

    else:
        raise ValueError(f"polyphony must be 'top' or 'weighted', got {polyphony!r}")

    # ── similarity conversion and threshold ───────────────────────────────────
    def _sim(c: float) -> float:
        return 1.0 / (1.0 + c)

    def _out_duration(j: int) -> float:
        return out_times[j + int(best_win[j])][0] - out_times[j][0]

    if debug and _log:
        algo = "weighted" if polyphony == "weighted" else "scalar/top-note"
        finite_costs = best_cost[np.isfinite(best_cost)]
        best_j = int(np.where(best_cost == finite_costs.min())[0][0]) \
            if len(finite_costs) else -1
        _log(f"\n  Cost profile ({algo}, {n_out} output positions):")
        _log(f"  {'output_pos':>10} | {'out_time_s':>10} | {'similarity':>10} | "
             f"{'dtw_cost':>8} | {'input_pos':>9} | {'win_notes':>9} | {'≥thresh':>7}")
        _log(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")
        for output_pos in range(n_out):
            s        = _sim(best_cost[output_pos]) if np.isfinite(best_cost[output_pos]) else 0.0
            flag     = "✓" if s >= min_similarity else " "
            out_time = out_times[output_pos][0]
            if np.isfinite(best_cost[output_pos]):
                cost_str   = f"{best_cost[output_pos]:.4f}"
                in_pos_str = f"{best_in_pos[output_pos]}"
                win_str    = f"{best_win[output_pos]}"
            else:
                cost_str = in_pos_str = win_str = "—"
            win_notes_str = str(int(win_str) + 1) if win_str != "—" else "—"
            note = "  ← lowest cost" if output_pos == best_j else ""
            _log(f"  {output_pos:>10} | {out_time:>10.2f} | {s:>10.4f} | "
                 f"{cost_str:>8} | {in_pos_str:>9} | {win_notes_str:>9} | {flag:>7}{note}")

    # ── greedy peak-pick ──────────────────────────────────────────────────────
    # Sort: descending similarity, then descending output duration
    # (longest-deterministic: equal-score ties resolve to the longer match).

    if _log: _log("Searching for similarity matches...")
    candidates = sorted(
        [j for j in range(n_out)
         if np.isfinite(best_cost[j]) and _sim(best_cost[j]) >= min_similarity],
        key=lambda j: (-_sim(best_cost[j]), -_out_duration(j)),
    )

    # `occupied[k]` is True if output note k is already inside an accepted match window.
    # Using occupied notes (not just start positions) guarantees zero output overlap.
    occupied = np.zeros(n_out, dtype=bool)
    matches: List[SimilarityMatch] = []

    for j in candidates:
        w = int(best_win[j])
        # Reject if any output note in this window is already claimed
        if occupied[j : min(n_out, j + w + 1)].any():
            continue

        i = int(best_in_pos[j])

        in_t0, in_t1   = in_times[i][0],  in_times[i + w][1]
        out_t0, out_t1 = out_times[j][0], out_times[j + w][1]

        in_times_rel, in_pitches, in_slice   = _ctx_notes_for_viz(_viz_in,  in_t0,  in_t1)
        out_times_rel, out_pitches, out_slice = _ctx_notes_for_viz(_viz_out, out_t0, out_t1)

        # DTW path on the match-window pitch intervals
        in_ms,  in_me  = in_slice
        out_ms, out_me = out_slice
        in_mp  = list(in_pitches[in_ms:in_me])
        out_mp = list(out_pitches[out_ms:out_me])
        if len(in_mp) >= 2 and len(out_mp) >= 2:
            n_dtw  = min(len(in_mp) - 1, len(out_mp) - 1)
            band_r = max(2, int(n_dtw * band_ratio))
            _, raw_path = _dtw_with_path(
                np.diff(in_mp[:n_dtw + 1]).astype(float),
                np.diff(out_mp[:n_dtw + 1]).astype(float),
                band_r,
            )
            viz_path = tuple(raw_path)
        else:
            viz_path = ()

        matches.append(SimilarityMatch(
            input_start=in_t0,   input_end=in_t1,
            output_start=out_t0, output_end=out_t1,
            similarity=_sim(best_cost[j]),
            in_note_times=in_times_rel,   in_note_pitches=in_pitches,
            out_note_times=out_times_rel, out_note_pitches=out_pitches,
            in_match_slice=in_slice,      out_match_slice=out_slice,
            dtw_path=viz_path,
        ))

        gap = max(1, int(w * min_sep_ratio))
        occupied[max(0, j - gap) : min(n_out, j + w + 1 + gap)] = True
        if len(matches) >= K:
            break

    matches.sort(key=lambda m: m.similarity, reverse=True)
    if _log: _log(f"  [timing] match search + peak-pick: {_time.perf_counter() - _t_total:.2f}s  →  {len(matches)} match(es)")
    return matches


# ── Debug JSON dump ───────────────────────────────────────────────────────────

def dump_debug_json(
    path:         str,
    raw_in:       "List[_NoteEvent]",
    raw_out:      "List[_NoteEvent]",
    flat_in:      "List[_NoteEvent]",
    flat_out:     "List[_NoteEvent]",
    matches:      "List[SimilarityMatch]",
    input_audio:  str = "",
    output_audio: str = "",
    latency:      "Dict[str, float] | None" = None,
) -> None:
    """Serialise note sequences + all matches to a JSON file for debugging."""
    import json as _json
    import dataclasses as _dc
    from datetime import datetime as _dt

    def _notes(events):
        return [[round(float(o), 4), round(float(f), 4), int(p), round(float(a), 4)]
                for o, f, p, a in events]

    payload = {
        "timestamp":     _dt.now().isoformat(timespec="seconds"),
        "input_audio":   input_audio,
        "output_audio":  output_audio,
        "latency_s":     {k: round(v, 3) for k, v in (latency or {}).items()},
        "raw_notes_in":   _notes(raw_in),
        "flat_notes_in":  _notes(flat_in),
        "raw_notes_out":  _notes(raw_out),
        "flat_notes_out": _notes(flat_out),
        "matches": [_dc.asdict(m) for m in matches],
    }
    with open(path, "w") as fh:
        _json.dump(payload, fh, indent=2)
    print(f"Debug JSON → {path}  ({len(matches)} matches)")
