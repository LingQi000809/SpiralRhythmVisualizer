// ============================================================
// SYNCED WITH: Hearmi-Frontend/app/studio/components/VisualizationWaitingView.tsx
// Only difference: import paths point to local utils/.
// Keep logic in sync with the Hearmi repo.
// ============================================================

import { useEffect, useRef, useState } from 'react';
import Meyda from "meyda";
import { PitchDetector } from "pitchy";
import type { InputData, MidiFeatures, NebulaPuff } from '../utils/visMidiHelpers';
import { analyzeMidi, chordRootHue, spawnNebulaPuff, drawNebulaLayer } from '../utils/visMidiHelpers';

export interface FrameFeatures {
  time: number;        // seconds
  duration: number;    // seconds (per note segment)
  pitch: number;       // MIDI
  pitchConf: number;   // confidence about pitch
  rms: number;         // 0-1 normalized
  centroid: number;    // 0-1 normalized
  flux?: number;
}

interface VisualizationWaitingViewProps {
  concatenatedAudioUrl: string | null; // URL to the concatenated audio
  audioRef: React.RefObject<HTMLAudioElement | null>;
  inputs: InputData[] | null; // list of InputData objects with optional MIDI notes per input
  isVisible?: boolean;
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export function VisualizationWaitingView({
  concatenatedAudioUrl,
  audioRef,
  inputs,
  isVisible = true,
}: VisualizationWaitingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const frameFeaturesRef = useRef<FrameFeatures[]>([]);
  const midiFeaturesRef = useRef<MidiFeatures[]>([]);
  const durationRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);

  const nebulaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const nebulaPuffsRef = useRef<NebulaPuff[]>([]);
  const lastActiveChordRef = useRef<string>("");

  const [statusText, setStatusText] = useState('Preparing visualization...');

  // --- EFFECT 1: OFFLINE ANALYSIS ---
  useEffect(() => {
    if (!concatenatedAudioUrl || !inputs) {
      frameFeaturesRef.current = [];
      midiFeaturesRef.current = [];
      durationRef.current = 0;
      setStatusText('Preparing MIDI preview...');
      return;
    }

    let cancelled = false;
    let localAudioContext: AudioContext | null = null;

    // We don't use a ref to prevent overlap, instead we rely on `cancelled`
    // If this effect re-runs, the previous promise's cancellation will
    // stop it from committing to state, and this new run will do the actual analysis.


    const runAudioAnalysis = async (input: InputData, fullBuffer: AudioBuffer) => {
      // Start total timer
      const t0 = performance.now();
      const sampleRate = fullBuffer.sampleRate;

      // Slice the relevant sample range out of the full buffer
      const startSample = Math.floor(input.inputStartTime * sampleRate);
      const endSample = Math.min(
        Math.ceil(input.inputEndTime * sampleRate),
        fullBuffer.length
      );
      const segmentLength = endSample - startSample;
      if (segmentLength <= 0) return;

      // Copy channel 0 samples for the segment only
      const fullChannelData = fullBuffer.getChannelData(0);
      const channelData = fullChannelData.subarray(startSample, endSample);
      const frameSize = 2048;
      const hopSize = Math.max(512, Math.floor(sampleRate / 20));

      if (cancelled) return;

      Meyda.sampleRate = sampleRate;
      Meyda.bufferSize = frameSize;
      console.log(`⏱ Decode Time: ${(performance.now() - t0).toFixed(2)}ms`);

      const detector = PitchDetector.forFloat32Array(frameSize);

      // ===================
      //     FRAME-WISE
      //  RMS, Centroid
      // ===================
      const rawRms: number[] = [];
      const rawCentroid: number[] = [];
      const rawPitchMidi: number[] = [];
      const pitchConfs: number[] = [];
      const frameTimes: number[] = [];

      console.log("Analyzing frame-wise spectral features and rms...");
      const framewiseStartTime = performance.now();
      for (let i = 0; i < channelData.length - frameSize; i += hopSize) {
        const frame = channelData.slice(i, i + frameSize);
        const f = Meyda.extract(['spectralCentroid', 'rms'], frame);
        if (!f) continue;

        // Local time within segment → shift to global timeline position
        const localTime = i / sampleRate;
        const globalTime = localTime + input.inputStartTime;
        frameTimes.push(globalTime);

        rawRms.push(f.rms || 0);
        rawCentroid.push(f.spectralCentroid || 0);
        const [frequency, clarity] = detector.findPitch(frame, sampleRate);

        if (frequency && clarity > 0) { // clarity threshold
          const midi = 69 + 12 * Math.log2(frequency / 440);
          rawPitchMidi.push(midi);
          pitchConfs.push(clarity);
        } else {
          rawPitchMidi.push(0);
          pitchConfs.push(clarity);
        }

        // Yield to main thread every 50 frames to prevent browser UI freeze
        if (rawRms.length % 50 === 0) {
          await new Promise(r => setTimeout(r, 0));
        }
      }
      const normRms = normalizeFeatureArr(rawRms);
      const normCentroid = normalizeFeatureArr(rawCentroid);
      if (cancelled) return;
      console.log(`⏱ Total Frame Analysis Time: ${(performance.now() - framewiseStartTime).toFixed(2)}ms;`);

      // ============================
      //    BUILD FEATURES TO DRAW
      // ============================
      console.log("Grouping features per note");
      const features: FrameFeatures[] = [];
      let startIdx = -1;
      const pitchThreshold = 0.8; // MIDI change tolerance
      for (let i = 1; i < rawPitchMidi.length; i++) {
        const pitch = rawPitchMidi[i];
        // start of a pitched segment
        if (pitch > 0 && startIdx === -1) startIdx = i;

        // end of segment
        const isEnd =
          startIdx !== -1 &&
          (pitch === 0 ||
            Math.abs(pitch - rawPitchMidi[startIdx]) > pitchThreshold ||
            i === rawPitchMidi.length - 1);

        if (isEnd) {
          const segmentRms = normRms.slice(startIdx, i + 1);
          const segmentCentroid = normCentroid.slice(startIdx, i + 1);
          const avgRms = segmentRms.reduce((a, b) => a + b, 0) / segmentRms.length;
          const avgCentroid = segmentCentroid.reduce((a, b) => a + b, 0) / segmentCentroid.length;
          const segmentPitch = rawPitchMidi.slice(startIdx, i + 1);
          const medPitch = medianPitch(segmentPitch);

          features.push({
            time: frameTimes[startIdx],
            duration: frameTimes[i] - frameTimes[startIdx],
            pitch: medPitch,
            pitchConf: pitchConfs[startIdx],
            rms: avgRms,
            centroid: avgCentroid,
          });

          startIdx = pitch > 0 ? i : -1;
        }
      }

      // Fallback: Some MIDI/polyphonic content can fail monophonic pitch segmentation,
      // resulting in zero features (blank visualization) even though audio exists.
      // Build lightweight energy-based events from framewise data so users always
      // get visual feedback while generation runs.
      if (features.length === 0 && frameTimes.length > 0) {
        const step = 8; // downsample to keep rendering lightweight
        const frameDuration = hopSize / sampleRate;

        for (let i = 0; i < frameTimes.length; i += step) {
          const rms = normRms[i] ?? 0;
          const centroid = normCentroid[i] ?? 0;
          const detectedPitch = rawPitchMidi[i] ?? 0;

          // Use detected pitch when available; otherwise derive a stable pseudo-pitch
          // from centroid so radial mapping still varies musically.
          const fallbackPitch = detectedPitch > 0
            ? detectedPitch
            : 48 + centroid * 24;

          features.push({
            time: frameTimes[i],
            duration: Math.max(frameDuration * step, 0.06),
            pitch: fallbackPitch,
            pitchConf: pitchConfs[i] ?? 0,
            rms,
            centroid,
          });
        }

        console.log('[VisualizationWaitingView] Using energy fallback visualization frames:', features.length);
      }

      if (cancelled) return;

      console.log(`🏁 TOTAL ANALYSIS TIME: ${(performance.now() - t0).toFixed(2)}ms`);
      console.log("Analysis complete.");
      frameFeaturesRef.current = frameFeaturesRef.current.concat(features);
    };

    const runMidiAnalysis = async (input: InputData) => {
      // MIDI note startTimes from Studio.tsx are relative to the segment (start = 0).
      // Shift them to the global concatenated timeline here.
      const shiftedInput: InputData = {
        ...input,
        midiNotes: input.midiNotes?.map(note => ({
          ...note,
          startTime: note.startTime + input.inputStartTime,
        })),
      };
      const midiFeatures = analyzeMidi(shiftedInput);
      midiFeaturesRef.current = midiFeaturesRef.current.concat(midiFeatures);
      if (cancelled) return;
    }

    const runAnalysis = async () => {
      try {
        setStatusText('Analyzing input audio and MIDI...');
        const arrayBuffer = await (async () => {
          const response = await fetch(concatenatedAudioUrl);
          if (!response.ok) {
            throw new Error(`Failed to fetch waiting audio (${response.status})`);
          }
          return response.arrayBuffer();
        })();
        if (cancelled) return;

        // Use offline context for faster decoding without playback overhead if supported
        // or just standard decoding block
        localAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = localAudioContext;

        let fullBuffer: AudioBuffer;
        try {
          fullBuffer = await localAudioContext.decodeAudioData(arrayBuffer);
        } catch (e) {
          throw new Error('Failed to decode audio data: ' + e);
        }
        if (cancelled) return;

        durationRef.current = fullBuffer.duration;

        let audioInputs: InputData[] = [];
        let midiInputs: InputData[] = [];
        inputs.forEach(input => {
          const isMidi =
            Array.isArray(input.midiNotes) && input.midiNotes.length > 0;
          if (isMidi) {
            midiInputs.push(input);
          } else {
            audioInputs.push(input);
          }
        });

        // Run audio segment analyses (each gets the already-decoded fullBuffer)
        await Promise.all(audioInputs.map(inp => runAudioAnalysis(inp, fullBuffer)));
        await Promise.all(midiInputs.map(runMidiAnalysis));

        setStatusText('');
      } catch (error) {
        if (!cancelled) {
          console.error('[VisualizationWaitingView] Analysis failed:', error);
          setStatusText('Unable to analyze preview audio');
        }
      }
    }

    void runAnalysis();

    return () => {
      cancelled = true;
      if (audioContextRef.current) {
        void audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
      if (localAudioContext && localAudioContext !== audioContextRef.current) {
        void localAudioContext.close().catch(() => {});
      }
    };
  }, [concatenatedAudioUrl, inputs]);

  // --- EFFECT 2: RENDER LOOP ---
  useEffect(() => {
    if (!isVisible || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // --- Window Resizing Observer ---
    const resize = () => {
      const canvas = canvasRef.current;
      const nebula = nebulaCanvasRef.current;
      if (!canvas || !nebula) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      // Update Main Canvas
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
      }

      // Update Nebula Canvas
      nebula.width = Math.round(rect.width * dpr);
      nebula.height = Math.round(rect.height * dpr);
      const nCtx = nebula.getContext("2d");
      if (nCtx) {
        nCtx.setTransform(1, 0, 0, 1, 0, 0);
        nCtx.scale(dpr, dpr);
      }
    };

    resize();
    const observer = new ResizeObserver(() => {
      resize();
    });
    observer.observe(canvas);

    // --- Animation Loop ---
    const draw = () => {
      const audio = audioRef.current;
      if (!audio || (!frameFeaturesRef.current.length && !midiFeaturesRef.current.length)) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const t = audio.currentTime;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const cx = w / 2;
      const cy = h / 2;
      const baseRadius = Math.min(w, h) * 0.2;
      const totalDuration = durationRef.current;
      const orbitDuration = Math.min(totalDuration, 10); // seconds per orbit

      // --- Nebula layer (drawn behind everything else) ---
      const nebulaCanvas = nebulaCanvasRef.current;
      if (nebulaCanvas) {
        const nCtx = nebulaCanvas.getContext("2d");
        if (nCtx) {
          drawNebulaLayer(nCtx, nebulaPuffsRef.current, performance.now());
        }
      }

      ctx.clearRect(0, 0, w, h);

      // Reset alpha for subsequent drawing
      ctx.globalAlpha = 1.0;

      // Draw Frame Features
      frameFeaturesRef.current.forEach((evt) => {
        const dt = t - evt.time;
        const rms = evt.rms;
        const centroid = evt.centroid;
        const color = getGalaxyColor(rms, centroid);

        const size = 10 + rms * 10;
        const glowSize = size * 2 + rms * 10;
        visualizeNote(
          ctx,
          dt, t,
          evt.time, evt.duration, evt.pitch, rms,
          cx, cy, baseRadius, color, size, glowSize,
          orbitDuration
        )
      });

      // Draw MIDI Features
      let activeChord = "";
      midiFeaturesRef.current.forEach((seg) => {
        if (t >= seg.startTime && t <= seg.startTime + seg.duration) {
          activeChord = seg.chord;
        }
        seg.notes.forEach((note) => {
          const dt = t - note.startTime;
          if (dt < 0) return;

          // --- strength from velocity ---
          const strength = Math.min(1, Math.max(0, note.velocity / 127)) * 0.5;

          const color = getMidiGalaxyColor(
            note.pitch,
            note.velocity,
            t,
            note.isChordTone
          );

          const size = 10 + strength * 5;
          const glowSize = size * 2 + strength * 5;
          visualizeNote(
            ctx,
            dt, t,
            note.startTime, note.duration, note.pitch, strength,
            cx, cy, baseRadius, color, size, glowSize,
            orbitDuration,
            true
          );
        });
      });

      // Spawn a nebula puff whenever the active chord changes
      if (activeChord && activeChord !== "N" && activeChord !== lastActiveChordRef.current) {
        lastActiveChordRef.current = activeChord;
        spawnNebulaPuff(activeChord, nebulaPuffsRef.current, w, h);
      }

      if (activeChord && activeChord !== "N") {
        // Get the visual properties for the current chord
        const currentHue = chordRootHue(activeChord);
        const isMinor = /min|dim|m7/.test(activeChord);
        const currentSat = isMinor ? 45 : 60;

        drawChordLabel(
          ctx,
          cx,
          cy,
          activeChord,
          currentHue,
          currentSat
        );
      }

      rafRef.current = requestAnimationFrame(draw);
    }


    rafRef.current = requestAnimationFrame(draw);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, [isVisible]);


  return (
    <div
      style={{
        backgroundColor: '#0d0d0d',
        flex: 1,
        minHeight: '200px',
        width: '100%',
        height: '100%',
        maxHeight: '100vh',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <canvas
        ref={nebulaCanvasRef}
        style={{ width: '100%', height: '100%', maxHeight: '80vh', position: 'absolute', top: 0, left: 0 }}
      />
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', maxHeight: '80vh', position: 'relative' }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none',
          color: 'rgba(234, 234, 234, 0.7)',
          fontSize: '13px',
          fontFamily: 'Inter, sans-serif',
          letterSpacing: '0.01em',
          textAlign: 'center',
          padding: '0 16px',
        }}
      >
        {statusText}
      </div>
    </div>
  );
}


// ==========================
//  FEATURE: HELPER FUNCTION
// ==========================

function normalizeFeatureArr (
  arr: number[],
  log = true,
  eps = 1e-6
): number[] {
  if (!arr.length) return [];
  const values = log ? arr.map(v => Math.log1p(v)) : [...arr];
  const sorted = [...values].sort((a, b) => a - b);

  const median = sorted[Math.floor(sorted.length * 0.5)];
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = Math.max(q3 - q1, eps);

  // --- linear normalization around median ---
  const normalized = values.map(v => {
    const centered = (v - median) / (2 * iqr); // roughly [-0.5, 0.5]
    return centered;
  });

  // --- remap to 0–1 ---
  let min = Math.min(...normalized);
  let max = Math.max(...normalized);

  // prevent collapse
  if (Math.abs(max - min) < eps) {
    return normalized.map(() => 0.5);
  }
  const scaled = normalized.map(v => (v - min) / (max - min));

  // --- gentle shaping ---
  return scaled.map(v => Math.pow(v, 0.9)); // subtle contrast boost

}

function mapPitch(pitch: number) {
  if (pitch <= 0) return 0;

  const minMidi = 20;
  const maxMidi = 90;

  // normalize to 0–1
  let x = (pitch - minMidi) / (maxMidi - minMidi);
  x = Math.min(1, Math.max(0, x));

  // sigmoid expansion of middle
  const k = 5; // controls how strong the "middle expansion" is
  const sigmoid = (t: number) => 1 / (1 + Math.exp(-k * (t - 0.5)));

  // normalize sigmoid output back to 0–1
  const s0 = sigmoid(0);
  const s1 = sigmoid(1);
  const y = (sigmoid(x) - s0) / (s1 - s0);

  return y;
}

function medianPitch(pitches: number[]): number {
  // Filter out unvoiced (0) pitches
  const voiced = pitches.filter(p => p > 0);
  if (voiced.length === 0) return 0;

  // Sort ascending
  const sorted = voiced.slice().sort((a, b) => a - b);

  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    // even number of elements → average middle two
    return (sorted[mid - 1] + sorted[mid]) / 2;
  } else {
    // odd → middle element
    return sorted[mid];
  }
}

function midiToNoteName(midi: number): string {
  if (!midi || midi <= 0) return "--";

  const noteNames = [
    "C", "C#/Db", "D", "D#/Eb", "E", "F",
    "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"
  ];

  const pitchClass = Math.round(midi) % 12;

  return `${noteNames[pitchClass]}`;
}

// ==========================
//  DRAWING: HELPER FUNCTION
// ==========================

// Map RMS and spectral centroid to galaxy-like color
function getGalaxyColor(rms: number, centroid: number): string {
  const hue = 220 + centroid * 80; // bluish-purple
  const sat = 40 + centroid * 40;
  const light = 50 + centroid * 20;
  const alpha = 0.3 + rms * 0.5;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

function pitchHash(pitch: number): number {
  // deterministic [-1, 1] per MIDI note
  const x = Math.sin(pitch * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}
function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function getMidiGalaxyColor(
  pitch: number, velocity: number, time: number, isChordTone: boolean
): string {
  const v = clamp01(velocity / 127);
  // shared galaxy motion (very small, slow)
  const drift = Math.sin(time * 0.08 + pitch * 0.03) * 5;
  // stable per-pitch variation
  const pitchJitter = pitchHash(pitch);
  const light = 38 + v * 32;
  return isChordTone
    ? chordColor(drift, pitchJitter, light, v)
    : nonChordColor(drift, pitchJitter, light, v);
}

function chordColor(
  drift: number, jitter: number, light: number, v: number
): string {
  // base band: blue → purple
  const baseHue = 235;
  // bounded variation inside band (NOT global hue shift)
  const hue = baseHue + drift + jitter * 10;
  const sat = 55 + v * 20;
  const alpha = 0.3 + v * 0.2;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

function nonChordColor(
  drift: number, jitter: number, light: number, v: number
): string {
  const baseHue = 350;

  const hue = baseHue + drift - jitter * 12;

  const sat = 50 + v * 15;

  const alpha = 0.3 + v * 0.2;

  return `hsla(${hue}, ${sat}%, ${light - 3}%, ${alpha})`;
}

function drawNote(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  glowSize: number,
  color: string,
  alpha: number
) {
  // glow
  ctx.globalAlpha = alpha * 0.5;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, glowSize, 0, Math.PI * 2);
  ctx.fill();

  // core
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, size, 0, Math.PI * 2);
  ctx.fill();
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  life: number,
  angle: number,
  strength: number
) {
  if (life < 0.08) return;

  // radial offset (stable, cheap)
  const offset = 14 + strength * 6;
  const lx = x + Math.cos(angle) * offset;
  const ly = y + Math.sin(angle) * offset;

  ctx.globalAlpha = life * 0.7;
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "18px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  ctx.fillText(text, lx, ly);
}
function drawChordLabel(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  text: string,
  hue: number,
  sat: number
) {
  const now = performance.now();
  // Smoother, slower breathing pulse
  const pulse = 1 + Math.sin(now * 0.0012) * 0.04;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(pulse, pulse);

  // 1. THE "VOLUMETRIC" UNDERGLOW
  // Large, very soft radial gradient that simulates light hitting distant gas.
  const bgGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, 100);
  bgGrad.addColorStop(0, `hsla(${hue}, ${sat}%, 50%, 0.18)`);
  bgGrad.addColorStop(1, `hsla(${hue}, ${sat}%, 50%, 0)`);
  ctx.fillStyle = bgGrad;
  ctx.beginPath();
  ctx.arc(0, 0, 100, 0, Math.PI * 2);
  ctx.fill();

  // 2. THE CHROMATIC BLOOM
  // We use 'screen' blending to make the glow additive (brighter)
  ctx.globalCompositeOperation = 'screen';
  ctx.shadowBlur = 15;
  ctx.shadowColor = `hsla(${hue}, ${sat}%, 65%, 0.6)`;

  // 3. THE CORE TEXT STYLING
  ctx.font = "bold 24px 'Outfit', 'Inter', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Gradient that goes from pure white to a pale tint of the chord's color
  const textGrad = ctx.createLinearGradient(0, -20, 0, 20);
  textGrad.addColorStop(0, "#ffffff");
  textGrad.addColorStop(0.5, "#ffffff");
  textGrad.addColorStop(1, `hsla(${hue}, ${sat}%, 90%, 1)`);

  ctx.fillStyle = textGrad;

  // Draw the text (this captures the shadowBlur bloom)
  ctx.fillText(text, 0, 0);

  // 4. THE LIGHT "STRIKE"
  // Draw it one more time with NO shadow and slightly thinner to create a sharp "hot" core
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.fillText(text, 0, 0);

  ctx.restore();
}

function visualizeNote(
  ctx: CanvasRenderingContext2D,
  dt: number, audioTime: number,
  startTime: number, duration: number, pitch: number, strength: number,
  cx: number, cy: number, baseRadius: number, color: string, size: number, glowSize: number,
  orbitDuration: number,
  isMidi = false
) {
  const minAlpha = 0.015;
  const fadeTime = 0.7 // seconds after note ends and fade
  // Life (transparency) of events
  let life;
  if (dt < 0) {
    life = 0;
  } else if (dt <= duration) {
    // while note is sounding
    life = 1;
  } else if (dt <= duration + fadeTime) {
    // fade after note ends
    const fadeProgress = (dt - duration) / fadeTime;
    life = 1 - fadeProgress;
  } else {
    life = minAlpha;
  }
  if (life === 0) return;

  const pitchNorm = mapPitch(pitch);
  const pitchSpread = 2.5; // larger -> more increased distance between notes
  const rBase = baseRadius + (pitchNorm - 0.5) * baseRadius * pitchSpread;

  const orbitIndex = Math.floor(startTime / orbitDuration);
  const orbitStart = orbitIndex * orbitDuration;

  const orbitProgress = (startTime - orbitStart) / orbitDuration;

  const angle =
    orbitProgress * Math.PI * 2 +
    orbitIndex * 0.3 +      // slight offset between rings
    audioTime * 0.1;               // slow rotation

  // how far through the note we are (0 → 1)
  const progress = Math.min(dt / duration, 1);

  // total possible trail resolution
  const maxTrailSteps = Math.max(
    Math.floor(duration * 100),
    1
  );
  // only draw up to current progress
  const trailSteps = Math.max(
    Math.floor(maxTrailSteps * progress),
    1
  );

  const trailLengthMultiplier = 0.006; // the larger multiplier, the longer the comet tail
  const trailR = rBase + (strength - 0.5) * 10;

  for (let j = 0; j < trailSteps; j++) {
    const trailAlpha = Math.max(
      life * (j / trailSteps),
      minAlpha
    );
    const trailSize = size * (j / trailSteps) * 0.7;
    const trailGlow = glowSize * (j / trailSteps) * 0.7;
    const trailAngle = angle + j * trailLengthMultiplier;

    const x = cx + Math.cos(trailAngle) * trailR;
    const y = cy + Math.sin(trailAngle) * trailR;

    drawNote(
      ctx,
      x,
      y,
      trailSize,
      trailGlow,
      color,
      trailAlpha
    );
  }

  // --- LABEL (only for long notes) ---
  const longNoteDurationThreshold = 0.2;
  const lowNoteThreshold = 47; // Anything below MIDI=40 is likely noise from polyphonic pitch detection

  if (duration > longNoteDurationThreshold && pitch > lowNoteThreshold) {
    const baseAngle = angle + trailSteps * trailLengthMultiplier;
    // inner notes lead more, outer notes lead less (distance along motion)
    const radialFactor = Math.max(0, 1 - (trailR / (baseRadius * 2)));
    // 1 = inner, 0 = outer
    const lead = (8 + strength * 14) * radialFactor;
    // forward motion along orbit
    const futureAngle = baseAngle + lead * 0.05;

    const headX = cx + Math.cos(futureAngle) * trailR;
    const headY = cy + Math.sin(futureAngle) * trailR;

    const label = `${midiToNoteName(pitch)}`;
    drawLabel(
      ctx,
      headX,
      headY,
      label,
      life,
      angle,
      strength
    );
    ctx.globalAlpha = 1.0;
  }
}
