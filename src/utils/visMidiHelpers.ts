// ============================================================
// SYNCED WITH: Hearmi-Frontend/app/studio/utils/visMidiHelpers.ts
// Keep this file in sync with the Hearmi repo — do not add
// local-only code here. Local dev utilities belong in pages/.
// ============================================================

export interface MidiFeatures {
  startTime: number;
  duration: number;
  chord: string;

  notes: (MidiNote & {
    isChordTone: boolean;
  })[];
}

export interface MidiNote {
  pitch: number; // MIDI note number (0-127)
  startTime: number; // in sec
  duration: number; // in sec
  velocity: number; // 0-127
}

export interface InputData {
  inputStartTime: number // seconds; start time of the input in the concatenated audio stream
  inputEndTime: number // seconds; end time of the input in the concatenated audio stream
  audioUrl: string // audio URL
  midiNotes?: MidiNote[] // if the input is MIDI, also include its midi notes
}

// ========
// ANALYSIS
// ========

export const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

const CHORD_TEMPLATES: Record<string, number[]> = {
  maj: [0, 4, 7],
  min: [0, 3, 7],
  dim: [0, 3, 6],
  aug: [0, 4, 8],
  maj7: [0, 4, 7, 11],
  min7: [0, 3, 7, 10],
  dom7: [0, 4, 7, 10],
  sus4: [0, 5, 7],
};

type ChordResult = {
  label: string;
  finalScore: number;
  root: number;
  type: keyof typeof CHORD_TEMPLATES | "maj";
};

// ==========================
// Pitch Scoring
// ==========================
// MEMORY_SEC: How far back should we remember notes?
// 1.0 to 1.5 seconds is usually the sweet spot for slow arpeggios.
const MEMORY_SEC = 2;

function detectChord(
  notes: MidiNote[],
  winStart: number,
  winEnd: number
): ChordResult {
  const profile = new Array(12).fill(0);
  let totalWeight = 0;

  const lookbackStart = winStart - MEMORY_SEC;

  for (const n of notes) {

    const overlapStart = Math.max(lookbackStart, n.startTime);
    const overlapEnd = Math.min(winEnd, n.startTime + n.duration);

    const weight = overlapEnd - overlapStart;
    if (weight <= 0) continue;

    const recency = 1 - (winEnd - n.startTime) / MEMORY_SEC;
    const effectiveWeight = weight * Math.max(0.2, recency);

    profile[n.pitch % 12] += effectiveWeight;
    totalWeight += effectiveWeight;
  }

  let best: ChordResult = {
    label: "N",
    finalScore: 0,
    root: 0,
    type: "maj",
  };

  for (let root = 0; root < 12; root++) {
    for (const [type, template] of Object.entries(CHORD_TEMPLATES)) {
      let chordWeight = 0;
      let matchCount = 0;

      for (const interval of template) {
        const pc = (root + interval) % 12;
        if (profile[pc] > 0) {
          chordWeight += profile[pc];
          matchCount++;
        }
      }

      const coverage = matchCount / template.length;
      const purity = chordWeight / (totalWeight || 1);
      const complexityPenalty = template.length === 4 ? 0.9 : 1.0;

      const score = chordWeight * coverage * purity * complexityPenalty;

      if (score > best.finalScore) {
        best = {
          label: NOTE_NAMES[root] + type,
          finalScore: score,
          root,
          type: type as keyof typeof CHORD_TEMPLATES,
        };
      }
    }
  }

  return best.finalScore > 0.3
    ? best
    : { label: "N", finalScore: 0, root: 0, type: "maj" };
}

export function analyzeMidi(input: InputData): MidiFeatures[] {
  const notes = input.midiNotes;
  if (!notes?.length) return [];

  // 1. Collect all unique time points (Note On / Note Off)
  const timePoints = new Set<number>();
  notes.forEach((n) => {
    timePoints.add(n.startTime);
    timePoints.add(n.startTime + n.duration);
  });

  const sortedTimes = Array.from(timePoints).sort((a, b) => a - b);
  const rawSegments: MidiFeatures[] = [];

  // 2. Initial Segment Creation
  for (let i = 0; i < sortedTimes.length - 1; i++) {
    const start = sortedTimes[i];
    const end = sortedTimes[i + 1];
    const duration = end - start;

    if (duration <= 0) continue;

    const activeNotes = notes.filter(
      (n) => n.startTime <= start && (n.startTime + n.duration) >= end
    );

    if (!activeNotes.length) continue;

    const bestChord = detectChord(activeNotes, start, end);
    const chordPcs = bestChord.label === "N"
      ? []
      : CHORD_TEMPLATES[bestChord.type].map((v) => (bestChord.root + v) % 12);

    rawSegments.push({
      startTime: start,
      duration: duration,
      chord: bestChord.label,
      notes: activeNotes.map((n) => ({
        ...n,
        isChordTone: chordPcs.includes(n.pitch % 12),
      })),
    });
  }

  // 3. Merging & Filtering
  const merged: MidiFeatures[] = [];
  const MIN_DURATION = 0.3;

  for (const seg of rawSegments) {
    const last = merged[merged.length - 1];

    // Merge if same chord OR if current segment is a "micro-segment" (< 0.1s)
    // Merging micro-segments into the previous one prevents "stuttering"
    if (last && (last.chord === seg.chord || seg.duration < MIN_DURATION)) {
      last.duration += seg.duration;

      // Update notes list
      const noteIds = new Set(last.notes.map(n => `${n.pitch}-${n.startTime}`));
      for (const n of seg.notes) {
        const id = `${n.pitch}-${n.startTime}`;
        if (!noteIds.has(id)) {
          last.notes.push(n);
          noteIds.add(id);
        }
      }
    } else {
      merged.push({ ...seg });
    }
  }

  // 4. Final Cleanup
  // Remove any remaining segments that are still too short (e.g., at the very start)
  // or segments where no chord was detected ("N") if you only want harmony.
  return merged.filter((s) => s.duration >= MIN_DURATION);
}

// =============
// VISUALIZATION
// =============

// Nebula puff: a cluster of expanding radial gradients painted on the background
// canvas whenever a new chord becomes active. Each puff lives for PUFF_LIFETIME ms,
// expanding outward and fading out. Multiple puffs from successive chords overlap
// via 'screen' blending to produce a layered, smoky look.
export interface NebulaPuff {
  hue: number;  // circle-of-fifths hue (0–360)
  sat: number;  // saturation — lower for minor/dim chords
  blobs: {
    ox: number; oy: number; // blob center (canvas px)
    r0: number;             // initial radius
    rMax: number;           // max radius at full expansion
    alpha: number;          // per-blob peak opacity
  }[];
  born: number; // performance.now() timestamp at spawn
}


// Circle-of-fifths hue per pitch class (index = MIDI pitch class 0..11, C..B).
// Adjacent entries on the circle of fifths are 30° apart so harmonically
// related chords (e.g. C–G–Am) produce visually neighboring colors.
const FIFTH_HUE = [
  212, // C   visible deep cyan-blue (lifted brightness)
  228, // C#  blue-indigo (brightened)
  246, // D   violet-blue
  268, // D#  violet
  292, // E   saturated purple
  312, // F   magenta
  328, // F#  pink
  345, // G   rose (high visibility)
  18,  // G#  orange-red (kept bright, not brown)
  34,  // A   amber (lifted for glow visibility)
  52,  // A#  yellow (soft but visible)
  190  // B   cyan-teal (bright return anchor)
];

const PUFF_LIFETIME = 6000;   // ms before a puff is removed
const PUFF_FADE_START = 0.45; // fraction of lifetime at which fade begins

export function chordRootHue(chord: string): number {
  if (!chord || chord === "N") return 200;
  // Match longest note name first to avoid "C#" being parsed as "C"
  const pc = NOTE_NAMES.findIndex(n => chord.startsWith(n));
  return pc >= 0 ? FIFTH_HUE[pc] : 200;
}

export function spawnNebulaPuff(
  chord: string,
  puffs: NebulaPuff[],
  w: number,
  h: number
) {
  const hue = chordRootHue(chord);
  const isMinor = /min|dim|m7/.test(chord);
  const baseAlpha = isMinor ? 0.055 : 0.07;
  const sat = isMinor ? 45 : 60;

  // Function to get a corner-weighted coordinate
  const edgeWeight = (size: number) => {
    const push = Math.pow(Math.random(), 1.5); // 1.5 is a "gentle" push to edges
    const pos = Math.random() > 0.5 ? push : 1 - push;
    return size * pos;
  };

  const cx = edgeWeight(w);
  const cy = edgeWeight(h);

  const numBlobs = 4 + Math.floor(Math.random() * 4);
  const blobs = Array.from({ length: numBlobs }, () => ({
    ox: cx + (Math.random() - 0.5) * w * 0.2, // Reduced spread slightly to keep clusters distinct
    oy: cy + (Math.random() - 0.5) * h * 0.2,
    r0: 40 + Math.random() * 60,
    rMax: 120 + Math.random() * 140,
    alpha: baseAlpha * (0.6 + Math.random() * 0.8),
  }));

  puffs.push({ hue, sat, blobs, born: performance.now() });
}

export function drawNebulaLayer(
  ctx: CanvasRenderingContext2D,
  puffs: NebulaPuff[],
  now: number
) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (let i = puffs.length - 1; i >= 0; i--) {
    const p = puffs[i];
    const age = now - p.born;
    if (age > PUFF_LIFETIME) { puffs.splice(i, 1); continue; }

    const progress = age / PUFF_LIFETIME;
    const alpha = progress > PUFF_FADE_START
      ? 1 - (progress - PUFF_FADE_START) / (1 - PUFF_FADE_START)
      : 1;

    for (const b of p.blobs) {
      // Radius expands quickly early (progress * 2 clamped to 1) then holds
      const r = b.r0 + (b.rMax - b.r0) * Math.min(progress * 2, 1);
      const grad = ctx.createRadialGradient(b.ox, b.oy, 0, b.ox, b.oy, r);
      const a = b.alpha * alpha;
      grad.addColorStop(0,   `hsla(${p.hue},${p.sat}%,55%,${a})`);
      grad.addColorStop(0.4, `hsla(${p.hue},${p.sat}%,45%,${a * 0.5})`);
      grad.addColorStop(1,   `hsla(${p.hue},${p.sat}%,35%,0)`);
      // 'screen' blend lets puffs stack additively without blowing out to white
      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(b.ox, b.oy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalCompositeOperation = 'source-over';
}

// ==========================
// MIDI Galaxy Color
// ==========================

function clamp01(x: number): number { return Math.max(0, Math.min(1, x)); }
function pitchHash(pitch: number): number {
  const x = Math.sin(pitch * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

function chordColor(drift: number, jitter: number, light: number, v: number): string {
  const baseHue = 235;
  const hue = baseHue + drift + jitter * 10;
  const sat = 55 + v * 20;
  const alpha = 0.3 + v * 0.2;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

function nonChordColor(drift: number, jitter: number, light: number, v: number): string {
  const baseHue = 350;
  const hue = baseHue + drift - jitter * 12;
  const sat = 50 + v * 15;
  const alpha = 0.3 + v * 0.2;
  return `hsla(${hue}, ${sat}%, ${light - 3}%, ${alpha})`;
}

export function getMidiGalaxyColor(
  pitch: number, velocity: number, time: number, isChordTone: boolean
): string {
  const v = clamp01(velocity / 127);
  const drift = Math.sin(time * 0.08 + pitch * 0.03) * 5;
  const jitter = pitchHash(pitch);
  const light = 38 + v * 32;
  return isChordTone
    ? chordColor(drift, jitter, light, v)
    : nonChordColor(drift, jitter, light, v);
}

// ==========================
// Circle of Fifths / Harmony
// ==========================

export const CIRCLE_OF_FIFTHS_ORDER = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F'];

export const COF_HUES: Record<string, number> = {};
CIRCLE_OF_FIFTHS_ORDER.forEach((note, i) => { COF_HUES[note] = i * 30; });

export const CHORD_INTERVALS: Record<string, number[]> = {
  'maj': [0, 4, 7], 'min': [0, 3, 7], 'dim': [0, 3, 6], 'aug': [0, 4, 8],
  'dom7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'min7': [0, 3, 7, 10],
  '7': [0, 4, 7, 10], 'sus4': [0, 5, 7], 'sus2': [0, 2, 7],
  'min7b5': [0, 3, 6, 10], 'dim7': [0, 3, 6, 9],
};

export const CHORD_DISSONANCE: Record<string, number> = {
  'maj': 0.1, 'min': 0.15, 'sus4': 0.2, 'sus2': 0.2,
  'maj7': 0.35, 'dom7': 0.4, '7': 0.4, 'min7': 0.45,
  'aug': 0.6, 'dim': 0.65, 'min7b5': 0.75, 'dim7': 0.85,
};

export interface ParsedChord {
  root: string;
  rootIdx: number;
  type: string;
  intervals: number[];
  pitchClasses: number[];
}

export function parseChordName(name: string): ParsedChord | null {
  if (!name || name === 'N') return null;
  let root: string, type: string;
  if (name.length > 1 && name[1] === '#') {
    root = name.substring(0, 2);
    type = name.substring(2);
  } else {
    root = name.substring(0, 1);
    type = name.substring(1);
  }
  if (!type) type = 'maj';
  const rootIdx = NOTE_NAMES.indexOf(root as typeof NOTE_NAMES[number]);
  if (rootIdx < 0) return null;
  const intervals = CHORD_INTERVALS[type] || CHORD_INTERVALS['maj'];
  const pitchClasses = intervals.map(iv => (rootIdx + iv) % 12);
  return { root, rootIdx, type, intervals, pitchClasses };
}

export interface ChordSegment {
  chord: string;
  start: number;
  end: number;
  confidence: number;
  dissonance: number;
  tension?: number;
}

export function lookupChord(chords: ChordSegment[], time: number): ChordSegment | null {
  if (!chords || chords.length === 0) return null;
  let lo = 0, hi = chords.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (time < chords[mid].start) hi = mid - 1;
    else if (time >= chords[mid].end) lo = mid + 1;
    else return chords[mid];
  }
  return null;
}
