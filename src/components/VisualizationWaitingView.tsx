import { useEffect, useRef, useState } from 'react';
import Meyda from "meyda";
import { PitchDetector } from "pitchy";
import { FrameFeatures, MidiFeatures, InputData, MidiNote } from '../types';
import { analyzeMidi } from '../utils/midiAnalysis';

interface VisualizationWaitingViewProps {
  concatenatedAudioUrl: string | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  inputs: InputData[] | null;
  isVisible?: boolean;
}

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

  // --- EFFECT 1: OFFLINE ANALYSIS ---
  useEffect(() => {
    console.log("Running 1st useEffect...");
    if (!concatenatedAudioUrl || !audioRef.current || !inputs) return;

    const runAudioAnalysis = async (input: InputData) => {
      console.log("Running audio analysis...")
      // Start total timer
      const t0 = performance.now();

      // Measure Fetch/Decode
      const tFetch = performance.now();
      const response = await fetch(input.audioUrl);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await audioContextRef.current!.decodeAudioData(arrayBuffer);

      const sampleRate = audioBuffer.sampleRate;
      const frameSize = 2048;
      const hopSize = 512;
      const channelData = audioBuffer.getChannelData(0);
      Meyda.sampleRate = sampleRate;
      Meyda.bufferSize = frameSize;
      console.log(`⏱ Decode Time: ${(performance.now() - tFetch).toFixed(2)}ms`);

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
          const time = i / sampleRate;
          frameTimes.push(time);
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

      }
      const normRms = normalizeFeatureArr(rawRms);
      const normCentroid = normalizeFeatureArr(rawCentroid);
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
        if (pitch > 0 && startIdx === -1) {
          startIdx = i;
        }
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
            time: input.inputStartTime + frameTimes[startIdx],
            duration: frameTimes[i] - frameTimes[startIdx],
            pitch: medPitch,
            pitchConf: pitchConfs[startIdx],
            rms: avgRms,
            centroid: avgCentroid,
          });

          startIdx = pitch > 0 ? i : -1;
        }
      }
      console.log(`🏁 TOTAL ANALYSIS TIME: ${(performance.now() - t0).toFixed(2)}ms`);
      console.log("Analysis complete.");
     
      frameFeaturesRef.current = frameFeaturesRef.current.concat(features);
    };

    const runMidiAnalysis = async (input: InputData) => {
      console.log("Running MIDI analysis...")
      const midiFeatures = analyzeMidi(input);
      console.log(midiFeatures);
      midiFeaturesRef.current = midiFeaturesRef.current.concat(midiFeatures);
    }

    const run = async () => {
      let audioInputs: InputData[] = [];
      let midiInputs: InputData[] = [];

      inputs.forEach(input => {
        const inputMidi = input.midiNotes;
        const isMidi = Array.isArray(inputMidi) && inputMidi.length > 0;

        if (isMidi) {
          inputMidi.forEach(note => {
            note.startTime += input.inputStartTime;
          });
          midiInputs.push(input);
        } else {
          audioInputs.push(input);
        }
      });

      console.log("Audio Inputs: ", audioInputs, "; MIDI Inputs: ", midiInputs);

      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      const ctx = audioContextRef.current;
      // decode concatenated audio first
      const response = await fetch(concatenatedAudioUrl);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      durationRef.current = audioBuffer.duration;

      await Promise.all(audioInputs.map(runAudioAnalysis));
      await Promise.all(midiInputs.map(runMidiAnalysis));

      console.log("midiFeaturesRef", midiFeaturesRef);
    };

    run();
  }, [concatenatedAudioUrl]);

  // --- EFFECT 2: RENDER LOOP ---
  useEffect(() => {
    console.log("Running 2nd useEffect...")
    if (!isVisible || !canvasRef.current) return;
    console.log("Drawing...")

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;

    // --- Window Resizing Observer ---
    const resize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);

      ctx.setTransform(1, 0, 0, 1, 0, 0); // reset first
      ctx.scale(dpr, dpr);
    };

    resize();
    const observer = new ResizeObserver(() => {
      resize();
    });
    observer.observe(canvasRef.current!);

    // --- Animation Loop ---
    const draw = () => {
      const audio = audioRef.current;
      if (!audio || !frameFeaturesRef.current.length) {
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

      if (activeChord && activeChord !== "N") {
        drawLabel(
          ctx,
          cx,
          cy,
          activeChord,
          1,          // always visible when active
          0,          // no angular offset
          1           // full strength
        );
        ctx.globalAlpha = 1.0;
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
        ref={canvasRef}
        style={{ width: '100%', height: '100%', maxHeight: '80vh', position: 'relative' }}
      />
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
  // const octave = Math.floor(midi / 12) - 1;

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

    const label = `${midiToNoteName(pitch)}`; //  | ${evt.pitchConf.toFixed(2)
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