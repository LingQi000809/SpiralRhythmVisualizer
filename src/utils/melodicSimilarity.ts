// Melodic motif similarity detection via sliding-window subsequence DTW.
// Key-invariant (median pitch normalization per window).
// Tempo-invariant (window size varies 50–150% of input length).
// One-to-many: finds ALL matching output segments for the input melody.

import type { FrameFeatures } from './visAudioHelpers';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface MotifMatchResult {
  inputStart:  number;
  inputEnd:    number;
  outputStart: number;
  outputEnd:   number;
  score:       number;   // normalized DTW cost — lower = more similar
  label:       string;
  rgb:         [number, number, number];
}

// ─── Internals ────────────────────────────────────────────────────────────────

const MOTIF_COLORS: [number, number, number][] = [
  [255, 140,  80],
  [ 80, 220, 170],
  [140, 160, 255],
  [255, 100, 180],
  [100, 220, 255],
];

function melodyNotes(feats: FrameFeatures[]): FrameFeatures[] {
  return feats
    .filter(f => f.isMelody !== false && f.pitch > 0)
    .sort((a, b) => a.time - b.time);
}

// Center pitches around their median — removes key/transposition differences.
function medianCenter(notes: FrameFeatures[]): number[] {
  const pitches = notes.map(n => n.pitch);
  const sorted  = [...pitches].sort((a, b) => a - b);
  const median  = sorted[Math.floor(sorted.length / 2)] ?? 60;
  return pitches.map(p => p - median);
}

/**
 * Subsequence DTW with Sakoe-Chiba band.
 * Returns average cost per warping step so scores are comparable across
 * different sequence lengths.
 */
function dtwNorm(a: number[], b: number[], bandR: number): number {
  const n = a.length, m = b.length;
  if (n === 0 || m === 0) return Infinity;
  const INF = 1e8;
  const dp  = Array.from({ length: n }, () => new Float32Array(m).fill(INF));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      // Skew band so it tracks the diagonal even for non-square matrices
      if (Math.abs(i - j * n / m) > bandR) continue;
      const d    = Math.abs(a[i] - b[j]);
      let   prev = INF;
      if (i > 0 && j > 0) prev = Math.min(prev, dp[i-1][j-1]);
      if (i > 0)           prev = Math.min(prev, dp[i-1][j]);
      if (j > 0)           prev = Math.min(prev, dp[i][j-1]);
      dp[i][j] = d + (prev < INF ? prev : 0);
    }
  }
  return dp[n-1][m-1] / (n + m);
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Find all output time-ranges that melodically resemble the input melody.
 *
 * Algorithm:
 *  1. Extract melody notes from both feature arrays (isMelody === true).
 *  2. Median-center pitches per sequence/window for key invariance.
 *  3. Slide a variable-length window across the output (50–150% of input
 *     length) and compute DTW against the full input melody.
 *  4. Pick local minima in the cost profile as match sites.
 *  5. Return all sites below `threshold` as MotifMatchResult[].
 */
export function findMelodicMotifMatches_dtw(
  inputFeats:  FrameFeatures[],
  outputFeats: FrameFeatures[],
  opts: {
    threshold?:  number;  // max normalized DTW cost to count as a match (default 3.0)
    minNotes?:   number;  // min melody notes needed (default 4)
  } = {},
): MotifMatchResult[] {
  const { threshold = 3.0, minNotes = 4 } = opts;

  const inNotes  = melodyNotes(inputFeats);
  const outNotes = melodyNotes(outputFeats);
  if (inNotes.length < minNotes || outNotes.length < minNotes) return [];

  const inPitches = medianCenter(inNotes);
  const n         = inNotes.length;
  const bandR     = Math.max(2, Math.ceil(n * 0.3));   // 30% Sakoe-Chiba band

  // Window sizes to try: 4 steps from 50% to 150% of input note count
  const minW    = Math.max(minNotes, Math.floor(n * 0.5));
  const maxW    = Math.min(outNotes.length, Math.ceil(n * 1.5));
  const wStep   = Math.max(1, Math.floor((maxW - minW) / 4));
  const windows = Array.from(
    { length: Math.ceil((maxW - minW + 1) / wStep) },
    (_, k) => Math.min(minW + k * wStep, maxW),
  );

  // Best DTW cost and winning window size for each output start position
  const scores   = new Float32Array(outNotes.length).fill(Infinity);
  const bestWins = new Uint16Array(outNotes.length).fill(n);

  for (const w of windows) {
    for (let j = 0; j + w <= outNotes.length; j++) {
      const winPitches = medianCenter(outNotes.slice(j, j + w));
      const cost       = dtwNorm(inPitches, winPitches, bandR);
      if (cost < scores[j]) { scores[j] = cost; bestWins[j] = w; }
    }
  }

  // Local-minimum peak picking — separation ≥ 40% of input length
  const minSep  = Math.max(2, Math.floor(n * 0.4));
  const results: MotifMatchResult[] = [];

  for (let j = 0; j < outNotes.length; j++) {
    if (scores[j] >= threshold) continue;

    let isMin = true;
    const lo  = Math.max(0, j - minSep);
    const hi  = Math.min(scores.length, j + minSep + 1);
    for (let k = lo; k < hi; k++) {
      if (k !== j && scores[k] <= scores[j]) { isMin = false; break; }
    }
    if (!isMin) continue;

    const w    = bestWins[j];
    const oEnd = Math.min(j + w - 1, outNotes.length - 1);

    results.push({
      inputStart:  inNotes[0].time,
      inputEnd:    inNotes[n - 1].time + inNotes[n - 1].duration,
      outputStart: outNotes[j].time,
      outputEnd:   outNotes[oEnd].time + outNotes[oEnd].duration,
      score:       scores[j],
      label:       results.length === 0 ? 'Motif' : `Motif ${results.length + 1}`,
      rgb:         MOTIF_COLORS[results.length % MOTIF_COLORS.length],
    });

    j += minSep;  // advance past this match to prevent double-counting
  }

  return results;
}
