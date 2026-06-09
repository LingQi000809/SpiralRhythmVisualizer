// ============================================================
// Shared drawing + audio-analysis helpers for the spiral galaxy visualizer.
// ============================================================

import Meyda from 'meyda';
import { PitchDetector } from 'pitchy';

export type PitchDetectorType = 'pitchy' | 'basic-pitch';

const BASIC_PITCH_MODEL_URL = '/model/basic-pitch/model.json';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _basicPitchInstance: any = null;
async function getBasicPitch() {
  if (!_basicPitchInstance) {
    const { BasicPitch } = await import('@spotify/basic-pitch');
    _basicPitchInstance = new BasicPitch(BASIC_PITCH_MODEL_URL);
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return _basicPitchInstance as any;
}

async function resampleTo22050(buf: AudioBuffer): Promise<Float32Array> {
  const TARGET_SR = 22050;
  if (buf.sampleRate === TARGET_SR) return buf.getChannelData(0).slice();
  const off = new OfflineAudioContext(1, Math.ceil(buf.duration * TARGET_SR), TARGET_SR);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start(0);
  return (await off.startRendering()).getChannelData(0).slice();
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface FrameFeatures {
  time: number;       // seconds
  duration: number;   // seconds (per note segment)
  pitch: number;      // MIDI
  pitchConf: number;  // confidence about pitch
  rms: number;        // 0–1 normalized
  centroid: number;   // 0–1 normalized
  // basic-pitch path only:
  isMelody?: boolean;   // true = salient melody voice, false = background harmony
  pitchLabel?: string;  // e.g. "A4" — set only for melody notes
}

// ── Audio analysis helpers ────────────────────────────────────────────────────

export function normalizeFeatureArr(arr: number[], log = true, eps = 1e-6): number[] {
  if (!arr.length) return [];
  const values = log ? arr.map(v => Math.log1p(v)) : [...arr];
  const sorted = [...values].sort((a, b) => a - b);

  const median = sorted[Math.floor(sorted.length * 0.5)];
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = Math.max(q3 - q1, eps);

  const normalized = values.map(v => (v - median) / (2 * iqr));

  const mn = Math.min(...normalized);
  const mx = Math.max(...normalized);
  if (Math.abs(mx - mn) < eps) return normalized.map(() => 0.5);
  const scaled = normalized.map(v => (v - mn) / (mx - mn));
  return scaled.map(v => Math.pow(v, 0.9));
}

export function medianPitch(pitches: number[]): number {
  const voiced = pitches.filter(p => p > 0);
  if (voiced.length === 0) return 0;
  const sorted = voiced.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

// ── Color helpers ─────────────────────────────────────────────────────────────

export function mapPitch(pitch: number): number {
  if (pitch <= 0) return 0;
  const minMidi = 20;
  const maxMidi = 90;
  let x = (pitch - minMidi) / (maxMidi - minMidi);
  x = Math.min(1, Math.max(0, x));
  const k = 5;
  const sigmoid = (t: number) => 1 / (1 + Math.exp(-k * (t - 0.5)));
  const s0 = sigmoid(0);
  const s1 = sigmoid(1);
  return (sigmoid(x) - s0) / (s1 - s0);
}

export function midiToNoteName(midi: number): string {
  if (!midi || midi <= 0) return '--';
  const noteNames = [
    'C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F',
    'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B',
  ];
  return noteNames[Math.round(midi) % 12];
}

export function getGalaxyColor(rms: number, centroid: number): string {
  const hue = 220 + centroid * 80;
  const sat = 40 + centroid * 40;
  const light = 50 + centroid * 20;
  const alpha = 0.3 + rms * 0.5;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

function pitchHash(pitch: number): number {
  const x = Math.sin(pitch * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
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

// ── Drawing primitives ────────────────────────────────────────────────────────

function drawNoteDot(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  size: number, glowSize: number,
  color: string, alpha: number
) {
  ctx.globalAlpha = alpha * 0.5;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, glowSize, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, size, 0, Math.PI * 2);
  ctx.fill();
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  text: string, life: number, angle: number, strength: number
) {
  if (life < 0.08) return;
  const offset = 14 + strength * 6;
  const lx = x + Math.cos(angle) * offset;
  const ly = y + Math.sin(angle) * offset;
  ctx.globalAlpha = life * 0.7;
  ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.font = '18px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, lx, ly);
}

export function drawChordLabel(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  text: string, hue: number, sat: number
) {
  const now = performance.now();
  const pulse = 1 + Math.sin(now * 0.0012) * 0.04;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(pulse, pulse);

  const bgGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, 100);
  bgGrad.addColorStop(0, `hsla(${hue}, ${sat}%, 50%, 0.18)`);
  bgGrad.addColorStop(1, `hsla(${hue}, ${sat}%, 50%, 0)`);
  ctx.fillStyle = bgGrad;
  ctx.beginPath();
  ctx.arc(0, 0, 100, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = 'screen';
  ctx.shadowBlur = 15;
  ctx.shadowColor = `hsla(${hue}, ${sat}%, 65%, 0.6)`;

  ctx.font = "bold 24px 'Outfit', 'Inter', sans-serif";
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const textGrad = ctx.createLinearGradient(0, -20, 0, 20);
  textGrad.addColorStop(0, '#ffffff');
  textGrad.addColorStop(0.5, '#ffffff');
  textGrad.addColorStop(1, `hsla(${hue}, ${sat}%, 90%, 1)`);
  ctx.fillStyle = textGrad;
  ctx.fillText(text, 0, 0);

  ctx.shadowBlur = 0;
  ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.fillText(text, 0, 0);

  ctx.restore();
}

// Main comet-trail renderer
// alphaScale is an optional multiplier for the transition fade
export function visualizeNote(
  ctx: CanvasRenderingContext2D,
  dt: number, audioTime: number,
  startTime: number, duration: number, pitch: number, strength: number,
  cx: number, cy: number, baseRadius: number,
  color: string, size: number, glowSize: number,
  orbitDuration: number,
  isMidi = false,
  alphaScale = 1,
  overrideR?: number
) {
  const minAlpha = 0.015;
  const fadeTime = 0.7;
  let life: number;
  if (dt < 0) {
    life = 0;
  } else if (dt <= duration) {
    life = 1;
  } else if (dt <= duration + fadeTime) {
    life = 1 - (dt - duration) / fadeTime;
  } else {
    life = minAlpha;
  }
  if (life === 0) return;
  life *= alphaScale;

  const pitchNorm = mapPitch(pitch);
  const pitchSpread = 2.5;
  const rBase = overrideR !== undefined
    ? overrideR
    : baseRadius + (pitchNorm - 0.5) * baseRadius * pitchSpread;

  const orbitIndex = Math.floor(startTime / orbitDuration);
  const orbitStart = orbitIndex * orbitDuration;
  const orbitProgress = (startTime - orbitStart) / orbitDuration;
  const angle = orbitProgress * Math.PI * 2 + orbitIndex * 0.3 + audioTime * 0.1;

  const progress = Math.min(dt / duration, 1);
  const maxTrailSteps = Math.max(Math.floor(duration * 100), 1);
  const trailSteps = Math.max(Math.floor(maxTrailSteps * progress), 1);

  const trailLengthMultiplier = 0.006;
  const trailR = rBase + (strength - 0.5) * 10;

  for (let j = 0; j < trailSteps; j++) {
    const trailAlpha = Math.max(life * (j / trailSteps), minAlpha * alphaScale);
    const trailSize = size * (j / trailSteps) * 0.7;
    const trailGlow = glowSize * (j / trailSteps) * 0.7;
    const trailAngle = angle + j * trailLengthMultiplier;
    const x = cx + Math.cos(trailAngle) * trailR;
    const y = cy + Math.sin(trailAngle) * trailR;
    drawNoteDot(ctx, x, y, trailSize, trailGlow, color, trailAlpha);
  }

  const longNoteDurationThreshold = 0.2;
  const lowNoteThreshold = 47;
  if (duration > longNoteDurationThreshold && pitch > lowNoteThreshold) {
    const baseAngle = angle + trailSteps * trailLengthMultiplier;
    const radialFactor = Math.max(0, 1 - (trailR / (baseRadius * 2)));
    const lead = (8 + strength * 14) * radialFactor;
    const futureAngle = baseAngle + lead * 0.05;
    const headX = cx + Math.cos(futureAngle) * trailR;
    const headY = cy + Math.sin(futureAngle) * trailR;
    drawLabel(ctx, headX, headY, `${midiToNoteName(pitch)}`, life, angle, strength);
  }
  ctx.globalAlpha = 1.0;
}

// Ergonomic wrapper for drawing a FrameFeatures event.
// Computes color/size/glow from the event and delegates to visualizeNote.
// alphaScale is used by ComparisonPage for the transition-fade effect.
export function drawFeatureNote(
  ctx: CanvasRenderingContext2D,
  evt: FrameFeatures,
  audioT: number,
  cx: number, cy: number,
  baseR: number, orbitDur: number,
  alphaScale = 1,
  overrideR?: number,
  colorOverride?: string,
  sizeScale = 1,
) {
  const color    = colorOverride ?? getGalaxyColor(evt.rms, evt.centroid);
  const baseSize = 10 + evt.rms * 10;
  const size     = baseSize * sizeScale;
  const glowSize = (baseSize * 2 + evt.rms * 10) * sizeScale;
  const dt = audioT - evt.time;
  visualizeNote(
    ctx, dt, audioT,
    evt.time, evt.duration, evt.pitch, evt.rms,
    cx, cy, baseR, color, size, glowSize,
    orbitDur, false, alphaScale, overrideR
  );
}

// ── Audio analysis ────────────────────────────────────────────────────────────

/**
 * Fetches an audio URL, runs Meyda + pitchy frame analysis, and returns
 * FrameFeatures[] via callback. Shared between ComparisonPage and StemVisualizationView.
 *
 * @param url         - Audio URL or base64 data URI to analyze
 * @param onDone      - Called with (features, duration) on completion
 * @param isCancelled - Polled periodically; analysis stops if it returns true
 */
export async function analyzeAudioUrl(
  url: string,
  onDone: (features: FrameFeatures[], duration: number) => void,
  isCancelled: () => boolean,
  minRms = 0.01,
  pitchDetector: PitchDetectorType = 'pitchy',
): Promise<void> {
  try {
    const buf = await (await fetch(url)).arrayBuffer();
    if (isCancelled()) return;
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const full = await audioCtx.decodeAudioData(buf);
    void audioCtx.close();
    if (isCancelled()) return;

    const sr       = full.sampleRate;
    const frameSize = 2048;
    const hopSize   = Math.max(512, Math.floor(sr / 20));

    Meyda.sampleRate = sr;
    Meyda.bufferSize = frameSize;
    const ch = full.getChannelData(0);

    // ── Meyda pass (always) — RMS + spectral centroid per frame ──────────────
    const rawRms: number[] = [];
    const rawC:   number[] = [];
    const times:  number[] = [];

    // pitchy detector: only instantiated when needed
    const det = pitchDetector === 'pitchy' ? PitchDetector.forFloat32Array(frameSize) : null;
    const rawP:  number[] = [];
    const confs: number[] = [];

    for (let i = 0; i < ch.length - frameSize; i += hopSize) {
      const frame = ch.slice(i, i + frameSize);
      const f = Meyda.extract(['spectralCentroid', 'rms'], frame);
      if (!f) continue;
      times.push(i / sr);
      rawRms.push(f.rms || 0);
      rawC.push(f.spectralCentroid || 0);
      if (det) {
        const [freq, cl] = det.findPitch(frame, sr);
        rawP.push(freq && cl > 0 ? 69 + 12 * Math.log2(freq / 440) : 0);
        confs.push(cl);
      }
      if (rawRms.length % 50 === 0) await new Promise(r => setTimeout(r, 0));
      if (isCancelled()) return;
    }

    const nRms = normalizeFeatureArr(rawRms);
    const nC   = normalizeFeatureArr(rawC);

    // ── basic-pitch branch ────────────────────────────────────────────────────
    if (pitchDetector === 'basic-pitch') {
      const mono22k = await resampleTo22050(full);
      if (isCancelled()) return;

      const bp = await getBasicPitch();
      if (isCancelled()) return;

      const allFrames: number[][] = [];
      const allOnsets: number[][] = [];
      await bp.evaluateModel(
        mono22k,
        (f: number[][], o: number[][], _c: number[][]) => { allFrames.push(...f); allOnsets.push(...o); },
        (_p: number) => {},
      );
      if (isCancelled()) return;

      const { outputToNotesPoly, noteFramesToTime } = await import('@spotify/basic-pitch');
      const notes = noteFramesToTime(outputToNotesPoly(allFrames, allOnsets));

      const { extractMelodyNotes_salience, midiNoteLabel } =
        await import('./melodicAnalysis');
      const { melody, harmony } = extractMelodyNotes_salience(notes);

      const hopSec = hopSize / sr;

      const toFeature = (note: typeof notes[0], isMelody: boolean): FrameFeatures | null => {
        const fi = Math.min(Math.round(note.startTimeSeconds / hopSec), rawRms.length - 1);
        if ((rawRms[fi] ?? 0) < minRms) return null;
        return {
          time:       note.startTimeSeconds,
          duration:   note.durationSeconds,
          pitch:      note.pitchMidi,
          pitchConf:  note.amplitude,
          rms:        nRms[fi] ?? 0,
          centroid:   nC[fi]  ?? 0,
          isMelody,
          pitchLabel: isMelody ? midiNoteLabel(note.pitchMidi) : undefined,
        };
      };

      const feats_bp: FrameFeatures[] = [
        ...melody.map(n => toFeature(n, true)),
        ...harmony.map(n => toFeature(n, false)),
      ].filter((f): f is FrameFeatures => f !== null);


      if (!isCancelled()) onDone(feats_bp, full.duration);
      return;
    }

    // ── pitchy branch (original) ──────────────────────────────────────────────
    const avgOf = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    const feats: FrameFeatures[] = [];
    let si = -1;
    for (let i = 1; i < rawP.length; i++) {
      const p = rawP[i];
      if (p > 0 && si === -1) si = i;
      const end = si !== -1 && (p === 0 || Math.abs(p - rawP[si]) > 0.8 || i === rawP.length - 1);
      if (end) {
        if (avgOf(rawRms.slice(si, i + 1)) >= minRms) {
          feats.push({
            time:      times[si],
            duration:  times[i] - times[si],
            pitch:     medianPitch(rawP.slice(si, i + 1)),
            pitchConf: confs[si],
            rms:       avgOf(nRms.slice(si, i + 1)),
            centroid:  avgOf(nC.slice(si, i + 1)),
          });
        }
        si = p > 0 ? i : -1;
      }
    }

    // Energy fallback for polyphonic / percussive content
    if (!feats.length && times.length) {
      const fd = hopSize / sr;
      for (let i = 0; i < times.length; i += 8) {
        if ((rawRms[i] ?? 0) < minRms) continue;
        const r = nRms[i] ?? 0, c = nC[i] ?? 0, p = rawP[i] ?? 0;
        feats.push({
          time:      times[i],
          duration:  Math.max(fd * 8, 0.06),
          pitch:     p > 0 ? p : 48 + c * 24,
          pitchConf: confs[i] ?? 0,
          rms:       r,
          centroid:  c,
        });
      }
    }

    if (!isCancelled()) onDone(feats, full.duration);
  } catch (e) {
    console.error('[analyzeAudioUrl]', e);
  }
}
