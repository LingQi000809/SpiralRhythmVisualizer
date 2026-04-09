import { useEffect, useRef, useState } from 'react';
import Meyda from "meyda";
import { PitchDetector } from "pitchy";
import { Essentia } from 'essentia.js';
import { EssentiaWASM } from 'essentia.js/dist/essentia-wasm.es.js'

export interface FrameFeatures {
  time: number;        // seconds
  duration: number;    // seconds (per note segment)
  pitch: number;       // MIDI
  rms: number;         // 0-1 normalized
  centroid: number;    // 0-1 normalized
  flux?: number;
}

interface VisualizationWaitingViewProps {
  audioUrl: string | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isVisible?: boolean;
}

function normalizeFeatureArr(
  arr: number[],
  pMin = 20,
  pMax = 80
): number[] {
  const logArr = arr.map(v => Math.log1p(v));

  const sorted = [...logArr].sort((a, b) => a - b);

  const minVal = sorted[Math.floor((pMin / 100) * sorted.length)];
  const maxVal = sorted[Math.floor((pMax / 100) * sorted.length)];

  return logArr.map(v => {
    const norm = (v - minVal) / (maxVal - minVal);
    return Math.min(1, Math.max(0, norm));
  });
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

export function VisualizationWaitingView({
  audioUrl,
  audioRef,
  isVisible = true,
}: VisualizationWaitingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const featuresRef = useRef<FrameFeatures[]>([]);
  const durationRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);

  // Add a locking ref to prevent double execution
  const analysisStartedRef = useRef(false);

  // --- EFFECT 1: OFFLINE ANALYSIS ---
  useEffect(() => {
    if (!audioUrl || !audioRef.current || analysisStartedRef.current) return;
    analysisStartedRef.current = true;

    const runAnalysis = async () => {
      // Start total timer
      const t0 = performance.now();

      // Measure Fetch/Decode
      const tFetch = performance.now();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const response = await fetch(audioUrl);
      const audioBuffer = await audioCtx.decodeAudioData(await response.arrayBuffer());
      durationRef.current = audioBuffer.duration;
      const sampleRate = audioBuffer.sampleRate;
      const frameSize = 2048;
      const hopSize = 128;
      console.log(`⏱ Decode Time: ${(performance.now() - tFetch).toFixed(2)}ms`);

      // Essentia
      console.log("Starting Essentia analysis...");
      const essentia = new Essentia(EssentiaWASM);
      console.log("Essentia Version:", essentia.version);
      // Pitch Detector
      // const bufferSize = 2048;
      // const detector = PitchDetector.forFloat32Array(bufferSize);
      
      const signal = essentia.audioBufferToMonoSignal(audioBuffer) as Float32Array;

      // ===================
      //        PITCH
      // ===================
      console.log("Extracting pitch with PredominantPitchMelodia...");
      const pitchExtractionStartTime = performance.now();
      const eqloudnessSignal = essentia.EqualLoudness(essentia.arrayToVector(signal), sampleRate) as any;
      console.log(eqloudnessSignal);
      const pitchValues = essentia.PredominantPitchMelodia(eqloudnessSignal.signal) as any;
      console.log(pitchValues);
      const pitchHz = pitchValues.pitch;
      console.log(pitchHz);
      const pitchConfidence = pitchValues.pitchConfidence;
      const n = pitchHz.length;
      const pitchTimes = n === 1 ? [0] : Array.from({ length: n }, (_, i) =>
        i * (durationRef.current / (n - 1))
      );
      console.log(`⏱ Total Pitch Extraction Time: ${(performance.now() - pitchExtractionStartTime).toFixed(2)}ms`);

      // ===================
      //     FRAME-WISE
      //  RMS, Centroid 
      // ===================
      const frames = essentia.FrameGenerator(signal, frameSize, hopSize);
      const rawRms: number[] = [];
      const rawCentroid: number[] = [];
      const frameTimes: number[] = [];
      
      console.log("Analyzing frame-wise spectral features and rms...");
      const framewiseStartTime = performance.now();
      for (let i = 0; i < frames.size(); i++) {
        const frame = frames.get(i);
        const time = (i * hopSize) / sampleRate;
        const spectrum = (essentia.Spectrum(frame) as any).spectrum;
        const { rms } = essentia.RMS(frame) as any;
        const { centroid } = essentia.Centroid(spectrum) as any;

        rawRms.push(rms);
        rawCentroid.push(centroid);
        frameTimes.push(time);
      }
      const normRms = normalizeFeatureArr(rawRms);
      const normCentroid = normalizeFeatureArr(rawCentroid);
      console.log(`⏱ Total Frame Analysis Time: ${(performance.now() - framewiseStartTime).toFixed(2)}ms; Number of frames = ${frames.size()}`);

      // ============================
      //    Align pitch and frame
      // ============================
      let pitchIdx = 0;
      const rawPitchMidi: number[] = [];
      const rawPitchConf: number[] = [];
      for (let i = 0; i < frameTimes.length; i++) {
        const t = frameTimes[i];
        // advance pitch index
        while (
          pitchIdx < pitchTimes.length - 1 &&
          pitchTimes[pitchIdx] < t
        ) {
          pitchIdx++;
        }
        const freq = pitchHz[pitchIdx];
        const conf = pitchConfidence[pitchIdx];

        // --- filter unvoiced ---
        if (!freq || conf <= 0) {
          rawPitchMidi.push(0);
          rawPitchConf.push(conf);
          continue;
        }
        const midi = 69 + 12 * Math.log2(freq / 440);
        rawPitchMidi.push(midi);
        rawPitchConf.push(conf);
      }
      
      // ============================
      //    BUILD FEATURES TO DRAW
      // ============================
      console.log("Grouping features per note");
      const features: FrameFeatures[] = [];
      let startIdx = -1;
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
            Math.abs(pitch - rawPitchMidi[startIdx]) > 0.5 ||
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
            rms: avgRms,
            centroid: avgCentroid,
          });

          startIdx = pitch > 0 ? i : -1;
        }
      }
      featuresRef.current = features;
      console.table(features);
      
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

        for (let j = 0; j < trailSteps; j++) {
          const trailAlpha = Math.max(
            life * (j / trailSteps),
            minAlpha
          );

          const trailSize = size * (j / trailSteps) * 0.7;
          const trailGlow = glowSize * (j / trailSteps) * 0.7;

          const trailR = rBase + (rms - 0.5) * 10;
          const trailAngle = angle + j * 0.005;

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
