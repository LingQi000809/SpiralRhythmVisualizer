import { useEffect, useRef, useState } from 'react';
import Meyda from "meyda";
import { PitchDetector } from "pitchy";

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
  audioUrl: string | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isVisible?: boolean;
}

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

// Map RMS and spectral centroid to galaxy-like color
function getGalaxyColor(rms: number, centroid: number): string {
  const hue = 220 + centroid * 80; // bluish-purple
  const sat = 40 + centroid * 40;
  const light = 50 + centroid * 20;
  const alpha = 0.3 + rms * 0.5;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
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
  rms: number
) {
  if (life < 0.08) return;

  // radial offset (stable, cheap)
  const offset = 14 + rms * 6;
  const lx = x + Math.cos(angle) * offset;
  const ly = y + Math.sin(angle) * offset;

  ctx.globalAlpha = life * 0.7;
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "18px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  ctx.fillText(text, lx, ly);
}

function mapPitch(pitch: number) {
  if (pitch <= 0) return 0;

  const lowCut = 20;   // below = noise/percussive
  const highCut = 90;  // above = unstable/harmonics

  // --- 1. LOW REGION (compressed inner circle) ---
  if (pitch < lowCut) {
    return (pitch / lowCut) * 0.15; // 0 → 0.15
  }

  // --- 2. MID REGION (expanded main band) ---
  if (pitch <= highCut) {
    let norm = (pitch - lowCut) / (highCut - lowCut);
    norm = Math.min(1, Math.max(0, norm));
    // expand spacing (important for melodic clarity)
    const expanded = Math.pow(norm, 0.7);
    return 0.15 + expanded * 0.7; // 0.15 → 0.85
  }

  // --- 3. HIGH REGION (compressed outer ring) ---
  const maxMidi = 127;
  let norm = (pitch - highCut) / (maxMidi - highCut);
  norm = Math.min(1, Math.max(0, norm));
  // compress strongly
  const compressed = Math.pow(norm, 2.5);
  return 0.85 + compressed * 0.15; // 0.85 → 1.0
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
  const octave = Math.floor(midi / 12) - 1;

  return `${noteNames[pitchClass]}${octave}`;
}

export function VisualizationWaitingView({
  audioUrl,
  audioRef,
  isVisible = true,
}: VisualizationWaitingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const featuresRef = useRef<FrameFeatures[]>([]);
  const durationRef = useRef<number>(0);
  const likelyPolyphonic = useRef<boolean>(false);
  const rafRef = useRef<number | null>(null);

  // Add a locking ref to prevent double execution
  const analysisStartedRef = useRef(false);

  const longNoteDurationThreshold = 0.2;
  const lowNoteThreshold = 47; // Anything below MIDI=40 is likely noise from polyphonic pitch detection

  // --- EFFECT 1: OFFLINE ANALYSIS ---
  useEffect(() => {
    if (!audioUrl || !audioRef.current || analysisStartedRef.current) return;
    analysisStartedRef.current = true;

    const runAnalysis = async () => {
      // Start total timer
      const t0 = performance.now();

      // Measure Fetch/Decode
      const tFetch = performance.now();
      const response = await fetch(audioUrl);
      const arrayBuffer = await response.arrayBuffer();
      
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const audioBuffer = await audioContextRef.current!.decodeAudioData(arrayBuffer);
      durationRef.current = audioBuffer.duration;

      // const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      // const response = await fetch(audioUrl);
      // const audioBuffer = await audioCtx.decodeAudioData(await response.arrayBuffer());
      // durationRef.current = audioBuffer.duration;
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
      featuresRef.current = features;
      // console.table(features);

      // Temporarily only display pitch label for monophonic audio, because there is no good real-time polyphonic audio pitch detection.
      const longNotes = features.filter(
        e => e.duration > longNoteDurationThreshold && e.pitch > 0
      );
      const lowPitchRatio =
        longNotes.length > 0
          ? longNotes.filter(e => e.pitch < lowNoteThreshold).length / longNotes.length
          : 0;
      likelyPolyphonic.current = lowPitchRatio > 0.6;
      if (likelyPolyphonic.current) console.log("⚠️Warning: Pitch detection doesn't work well on polyphonic audio!");
      
      console.log(`🏁 TOTAL ANALYSIS TIME: ${(performance.now() - t0).toFixed(2)}ms`);
      console.log("Analysis complete.");
    };

    runAnalysis();

    analysisStartedRef.current = false;
  }, [audioUrl]);

  // --- EFFECT 2: RENDER LOOP ---
  useEffect(() => {
    if (!isVisible || !canvasRef.current) return;
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
      if (!audio || !featuresRef.current.length) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const minAlpha = 0.02;
      const t = audio.currentTime;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const cx = w / 2;
      const cy = h / 2;
      const baseRadius = Math.min(w, h) * 0.3;
      const totalDuration = durationRef.current;
      const orbitDuration = Math.min(totalDuration, 10); // seconds per orbit

      ctx.clearRect(0, 0, w, h);

      // Reset alpha for subsequent drawing
      ctx.globalAlpha = 1.0;

      featuresRef.current.forEach((evt) => {
        const dt = t - evt.time;
        
        const fadeTime = 0.7 // seconds after note ends and fade
        // Life (transparency) of events
        let life;
        if (dt < 0) {
          life = 0;
        } else if (dt <= evt.duration) {
          // while note is sounding
          life = 1;
        } else if (dt <= evt.duration + fadeTime) {
          // fade after note ends
          const fadeProgress = (dt - evt.duration) / fadeTime;
          life = 1 - fadeProgress;
        } else {
          life = minAlpha;
        }
        if (life === 0) return;

        const rms = evt.rms;
        const centroid = evt.centroid;
        const pitchNorm = mapPitch(evt.pitch);
        const rBase = baseRadius + (pitchNorm - 0.5) * baseRadius;

        const orbitIndex = Math.floor(evt.time / orbitDuration);
        const orbitStart = orbitIndex * orbitDuration;

        const orbitProgress = (evt.time - orbitStart) / orbitDuration;

        const angle =
          orbitProgress * Math.PI * 2 +
          orbitIndex * 0.3 +      // slight offset between rings
          t * 0.1;               // slow rotation

        const size = 10 + rms * 10;
        const glowSize = size * 2 + rms * 10;
        const color = getGalaxyColor(rms, centroid);

        // how far through the note we are (0 → 1)
        const progress = Math.min(dt / evt.duration, 1);
        // total possible trail resolution
        const maxTrailSteps = Math.max(
          Math.floor(evt.duration * 100),
          1
        );
        // only draw up to current progress
        const trailSteps = Math.max(
          Math.floor(maxTrailSteps * progress),
          1
        );
        
        const trailLengthMultiplier = 0.006; // the larger multiplier, the longer the comet tail
        const trailR = rBase + (rms - 0.5) * 10;
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
        const outwardOffset = 30 + rms * 8; 
        if (evt.duration > longNoteDurationThreshold && evt.pitch > lowNoteThreshold) {
          const headX = cx + Math.cos(angle + trailSteps * trailLengthMultiplier) * (trailR + outwardOffset);
          const headY = cy + Math.sin(angle + trailSteps * trailLengthMultiplier) * (trailR + outwardOffset);
          const label = `${midiToNoteName(evt.pitch)}`; //  | ${evt.pitchConf.toFixed(2)
          drawLabel(
            ctx,
            headX,
            headY,
            label,
            life,
            angle,
            rms
          );
          ctx.globalAlpha = 1.0;
        }
      });

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
