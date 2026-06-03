// ============================================================
// Shared drawing + audio-analysis helpers for the spiral galaxy visualizer.
// ============================================================

// ── Types ─────────────────────────────────────────────────────────────────────

export interface FrameFeatures {
  time: number;       // seconds
  duration: number;   // seconds (per note segment)
  pitch: number;      // MIDI
  pitchConf: number;  // confidence about pitch
  rms: number;        // 0–1 normalized
  centroid: number;   // 0–1 normalized
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
  alphaScale = 1
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
  const rBase = baseRadius + (pitchNorm - 0.5) * baseRadius * pitchSpread;

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
    ctx.globalAlpha = 1.0;
  }
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
  alphaScale = 1
) {
  const color = getGalaxyColor(evt.rms, evt.centroid);
  const size = 10 + evt.rms * 10;
  const glowSize = size * 2 + evt.rms * 10;
  const dt = audioT - evt.time;
  visualizeNote(
    ctx, dt, audioT,
    evt.time, evt.duration, evt.pitch, evt.rms,
    cx, cy, baseR, color, size, glowSize,
    orbitDur, false, alphaScale
  );
}
