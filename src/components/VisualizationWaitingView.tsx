import { useEffect, useRef, useState } from 'react';
import Meyda from "meyda";
import { PitchDetector } from "pitchy";
import { Essentia } from 'essentia.js';
import { EssentiaWASM } from 'essentia.js/dist/essentia-wasm.es.js'
import { 
  EssentiaTFInputExtractor, 
  TensorflowMusiCNN, 
} from 'essentia.js/dist/essentia.js-model.es.js';
import * as tf from '@tensorflow/tfjs';

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

interface MoodPrediction {
  startTime: number;
  endTime: number;
  mood: string;
  score: number;
}

const MOOD_EMOJIS: Record<string, string> = {
  aggressive: "👊",
  danceable: "💃",
  happy: "😄",
  relaxed: "😌",
  sad: "😢",
};
const moods = Object.keys(MOOD_EMOJIS);

// See tutorial page for the basic workflow: https://mtg.github.io/essentia.js/docs/api/tutorial-3.%20Machine%20learning%20inference%20with%20Essentia.js.html
// For the list of available models: https://essentia.upf.edu/models.html
// Some of the functions are out-dated. Also check visualization/node_modules/essentia.js/dist/essentia.js-model.es.js for source code.
const runMLClassification = async (
  audioBuffer: AudioBuffer
  // windowSec: number = 6
) => {
  // Extract features
  const inputExtractor = new EssentiaTFInputExtractor(EssentiaWASM, "musicnn", false);
  const resampledSignal = await inputExtractor.downsampleAudioBuffer(audioBuffer);
  const inputFeatures = inputExtractor.computeFrameWise(resampledSignal, inputExtractor.frameSize);

  const resultsPerMood = await Promise.all(
    moods.map(async (mood) => {
      const modelDir = `/models/moods/${mood}/`; 
      const modelJson = modelDir + "model.json";
      const metadataJson = modelDir + "meta.json";
      const res = await fetch(metadataJson);
      const json = await res.json();
      const mood_idx = json.classes.indexOf(mood);

      const musicnn = new TensorflowMusiCNN(tf, modelJson, true);
      await (musicnn as any).initialize();

      const predictions = await musicnn.predict(inputFeatures, true);
      console.log("mood: ", mood, predictions);
      const mood_scores = predictions.map((prediction: number[]) => prediction[mood_idx]);
      console.log(mood_scores);
      return { mood, mood_scores };
    })
  );
  console.log(resultsPerMood);

  const numPatches = resultsPerMood[0].mood_scores.length;
  const patchDurationSec = audioBuffer.duration / numPatches;
  // const patchDurationSec = (187 * 256) / 16000; // MusiCNN fixed patch length (~3s)

  const moodsPredicted: MoodPrediction[] = [];

  for (let i = 0; i < numPatches; i++) {
    // Collect scores for this patch
    const patchScores = resultsPerMood.map(m => ({ mood: m.mood, score: m.mood_scores[i] }));
    const sorted = patchScores.sort((a, b) => b.score - a.score);
    // Select confident moods
    let selected = sorted.filter(s => s.score >= 0.8);
    // If none, select top 1 if somewhat certain
    if (selected.length === 0 && sorted[0].score >= 0.3) {
      selected = [sorted[0]];
    }
    // Skip if no mood meets criteria
    if (selected.length === 0) continue;

    const startTime = i * patchDurationSec;
    const endTime = startTime + patchDurationSec;
    selected.forEach(s => {
      moodsPredicted.push({ startTime, endTime, mood: s.mood, score: s.score });
    });
  }

  console.log(moodsPredicted);
  return moodsPredicted;
};

function mapPitch(pitch: number, center = 60, spread = 6) {
  const x = (pitch - center) / spread;
  return 1 / (1 + Math.exp(-x)); // 0–1
}

export function VisualizationWaitingView({
  audioUrl,
  audioRef,
  isVisible = true,
}: VisualizationWaitingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const featuresRef = useRef<FrameFeatures[]>([]);
  const moodLabelRef = useRef<MoodPrediction[]>([]);
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
      console.log(`⏱ Decode Time: ${(performance.now() - tFetch).toFixed(2)}ms`);

      // Essentia
      console.log("Starting Essentia analysis...");
      const essentia = new Essentia(EssentiaWASM);
      console.log("Essentia Version:", essentia.version);
      // Pitch Detector
      const bufferSize = 2048;
      const detector = PitchDetector.forFloat32Array(bufferSize);
      
      const signal = essentia.audioBufferToMonoSignal(audioBuffer);
      const frames = essentia.FrameGenerator(signal, 2048, 512);
      
      // Arrays to store timing data
      const times = { pitch: 0, spectral: 0, hpcp: 0 };

      const rawPitch: number[] = [];
      const rawRms: number[] = [];
      const rawCentroid: number[] = [];
      const frameTimes: number[] = [];    
      // const rawOnsets: number[] = [];
      // const rawFlux: number[] = [];

      const loopStart = performance.now();
      console.log("Analyzing frame-wise spectral features, pitch, rms, and chords");
      for (let i = 0; i < frames.size(); i++) {
        const frame = frames.get(i);
        const time = (i * 512) / sampleRate;
        
        // Spectral
        const ts = performance.now();
        const spectrum = (essentia.Spectrum(frame) as any).spectrum;
        const { rms } = essentia.RMS(frame) as any;
        const { centroid } = essentia.Centroid(spectrum) as any; 
        // const { hfc } = essentia.HFC(spectrum) as any;
        // rawOnsets.push(hfc);
        // const { flux } = essentia.Flux(spectrum) as any;
        // rawFlux.push(flux);
        times.spectral += (performance.now() - ts);

        // Pitch
        const tp = performance.now();
        const frameData = essentia.vectorToArray(frame);
        const [frequency, clarity] = detector.findPitch(frameData, sampleRate);
        rawPitch.push(frequency && clarity > 0.8 ? (69 + 12 * Math.log2(frequency / 440)) : 0);
        // Essentia PitchYin takes too long
        // const { pitch, pitchConfidence } = essentia.PitchYin(frame) as any;      
        // rawPitch.push(pitchConfidence > 0.8 ? (69 + 12 * Math.log2(pitch / 440)) : 0);
        times.pitch += (performance.now() - tp);

        rawRms.push(rms);
        rawCentroid.push(centroid);
        frameTimes.push(time);
      }
      const normRms = normalizeFeatureArr(rawRms);
      const normCentroid = normalizeFeatureArr(rawCentroid);

      console.log(`⏱ Total Frame Analysis Time: ${(performance.now() - loopStart).toFixed(2)}ms; Number of frames = ${frames.size()}`);
      console.log(`   - Spectral Avg: ${(times.spectral / frames.size()).toFixed(4)}ms/frame`);
      console.log(`   - Pitch Avg: ${(times.pitch / frames.size()).toFixed(4)}ms/frame`);

      // --- Group features into notes ---
      console.log("Grouping features per note");
      const features: FrameFeatures[] = [];
      let noteStartIdx = 0;
      for (let i = 1; i < rawPitch.length; i++) {
        if (Math.abs(rawPitch[i] - rawPitch[noteStartIdx]) > 0.5 || i === rawPitch.length - 1) {
          features.push({
            time: frameTimes[noteStartIdx],
            duration: frameTimes[i] - frameTimes[noteStartIdx],
            pitch: rawPitch[noteStartIdx],
            rms: normRms[noteStartIdx],
            centroid: normCentroid[noteStartIdx]
          });
          noteStartIdx = i;
        }
      }
      featuresRef.current = features;
      console.table(features);
      
      // --- Classification ML ---
      const tMLStart = performance.now();
      const timeStampedMoods = await runMLClassification(audioBuffer);
      moodLabelRef.current = timeStampedMoods;
      const tMLEnd = performance.now();

      console.log(`⏱ ML Classification Time: ${(tMLEnd - tMLStart).toFixed(2)}ms`);

      console.log(`🏁 TOTAL ANALYSIS TIME: ${(performance.now() - t0).toFixed(2)}ms`);
      console.log("Analysis complete.");
    };

    runAnalysis();
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

      const currentMoods = moodLabelRef.current.filter(
        (m) => t >= m.startTime && t <= m.endTime
      );

      // drawMoodBackground(ctx, "danceable", t, w, h, 0.8);

      currentMoods.forEach((m, index) => {

        // Set alpha based on the ML score (0.0 to 1.0)
        ctx.globalAlpha = Math.min(m.score, 1.0);
        
        ctx.font = "48px serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        // Offset multiple emojis if they overlap in the same time window
        const xOffset = (index - (currentMoods.length - 1) / 2) * 60;
        const emoji = MOOD_EMOJIS[m.mood] || "";

        ctx.fillText(emoji, cx + xOffset, cy);
      });

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

function drawMoodBackground(
  ctx: CanvasRenderingContext2D,
  mood: string,
  t: number,
  w: number,
  h: number,
  intensity: number
) {
  // ctx.save();

  switch (mood) {
    case "aggressive":
      drawAggressive(ctx, t, w, h, intensity);
      break;
    case "danceable":
      drawDanceable(ctx, t, w, h, intensity);
      break;
    case "happy":
      drawHappy(ctx, t, w, h, intensity);
      break;
    case "relaxed":
      drawRelaxed(ctx, t, w, h, intensity);
      break;
    case "sad":
      drawSad(ctx, t, w, h, intensity);
      break;
  }

  // ctx.restore();
}

// shock wave
function drawAggressive(
  ctx: CanvasRenderingContext2D,
  t: number,
  w: number,
  h: number,
  intensity: number
) {
  const cx = w / 2;
  const cy = h / 2;

  const speed = 200;
  const maxRadius = Math.max(w, h);

  for (let i = 0; i < 3; i++) {
    const phase = (t * speed + i * 100) % maxRadius;
    const alpha = (1 - phase / maxRadius) * 0.15 * intensity;

    ctx.beginPath();
    ctx.arc(cx, cy, phase, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255, 80, 80, ${alpha})`;
    ctx.lineWidth = 5;
    ctx.stroke();
  }
}

// color blobs
function drawDanceable(
  ctx: CanvasRenderingContext2D,
  t: number,
  w: number,
  h: number,
  intensity: number
) {
  const blobs = 5;

  for (let i = 0; i < blobs; i++) {
    const x = (Math.sin(t * 0.5 + i) * 0.5 + 0.5) * w;
    const y = (Math.cos(t * 0.4 + i * 2) * 0.5 + 0.5) * h;

    const radius = 150 + Math.sin(t + i) * 50;

    const hue = (t * 30 + i * 60) % 360;

    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);

    grad.addColorStop(0, `hsla(${hue}, 80%, 70%, ${0.12 * intensity})`);
    grad.addColorStop(1, "transparent");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHappy(ctx: CanvasRenderingContext2D, t: number, w: number, h: number, intensity: number) {
  const particles = 15;
  
  for (let i = 0; i < particles; i++) {
    // Unique seed for each particle based on index
    const x = ((i * 137.5) % 100) / 100 * w; 
    const speed = 20 + (i % 5) * 10;
    const y = h - ((t * speed + i * 100) % (h + 200)); // Moves upward
    
    const size = 40 + (i % 10) * 20;
    const alpha = Math.sin((y / h) * Math.PI) * 0.2 * intensity; // Fade in/out

    const grad = ctx.createRadialGradient(x, y, 0, x, y, size);
    grad.addColorStop(0, `hsla(${40 + i * 2}, 90%, 70%, ${alpha})`); // Warm golds/pinks
    grad.addColorStop(1, "transparent");

    ctx.fillStyle = grad;
    ctx.fillRect(x - size, y - size, size * 2, size * 2);
  }
}

function drawRelaxed(ctx: CanvasRenderingContext2D, t: number, w: number, h: number, intensity: number) {
  const layers = 3;
  ctx.globalCompositeOperation = "screen"; // Blends colors softly

  for (let i = 0; i < layers; i++) {
    ctx.beginPath();
    const fillAlpha = (0.1 + i * 0.05) * intensity;
    ctx.fillStyle = `rgba(130, 180, 230, ${fillAlpha})`;
    
    ctx.moveTo(0, h);
    // Create a wave using sine
    for (let x = 0; x <= w; x += 20) {
      const y = h - (50 + i * 40) - Math.sin(t * 0.5 + x * 0.005 + i) * 30;
      ctx.lineTo(x, y);
    }
    
    ctx.lineTo(w, h);
    ctx.fill();
  }
}

function drawSad(ctx: CanvasRenderingContext2D, t: number, w: number, h: number, intensity: number) {
  const clouds = 4;
  ctx.globalCompositeOperation = "source-over";

  for (let i = 0; i < clouds; i++) {
    const y = h * 0.3 + (i * h * 0.15);
    const xOffset = Math.sin(t * 0.2 + i) * 100;
    const opacity = (0.05 + Math.sin(t * 0.4 + i) * 0.02) * intensity;

    // Stretched ellipse gradient
    const grad = ctx.createRadialGradient(w/2 + xOffset, y, 0, w/2 + xOffset, y, w * 0.8);
    grad.addColorStop(0, `rgba(100, 120, 160, ${opacity})`);
    grad.addColorStop(1, "transparent");

    ctx.fillStyle = grad;
    ctx.setTransform(2, 0, 0, 0.5, -w/2, 0); // Stretch horizontally
    ctx.fillRect(0, 0, w * 2, h * 2);
    ctx.resetTransform();
  }
}