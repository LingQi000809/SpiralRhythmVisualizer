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

interface AnalysisLabels {
  chord: string;
  mood: string;
  genre: string;
  voice: string;
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

export function VisualizationWaitingView({
  audioUrl,
  audioRef,
  isVisible = true,
}: VisualizationWaitingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const featuresRef = useRef<FrameFeatures[]>([]);
  const chordTimelineRef = useRef<{time: number, chord: string}[]>([]);
  const durationRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);
  const currentLabelsRef = useRef<AnalysisLabels>({
    chord: "", mood: "Analyzing...", genre: "Analyzing...", voice: "Analyzing..."
  });
  const [displayLabels, setDisplayLabels] = useState<AnalysisLabels>(currentLabelsRef.current);

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
      const hpcpVectors = new essentia.module.VectorVectorFloat();  

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

        // HPCP (chord)
        const th = performance.now();
        const peaks = essentia.SpectralPeaks(spectrum) as any;
        const hpcp = (essentia.HPCP(peaks.frequencies, peaks.magnitudes) as any).hpcp;
        hpcpVectors.push_back(hpcp);
        times.hpcp += (performance.now() - th);
        rawRms.push(rms);
        rawCentroid.push(centroid);
        frameTimes.push(time);
      }
      const normRms = normalizeFeatureArr(rawRms);
      const normCentroid = normalizeFeatureArr(rawCentroid);

      console.log(`⏱ Total Frame Analysis Time: ${(performance.now() - loopStart).toFixed(2)}ms; Number of frames = ${frames.size()}`);
      console.log(`   - Spectral Avg: ${(times.spectral / frames.size()).toFixed(4)}ms/frame`);
      console.log(`   - Pitch Avg: ${(times.pitch / frames.size()).toFixed(4)}ms/frame`);
      console.log(`   - HPCP Avg: ${(times.hpcp / frames.size()).toFixed(4)}ms/frame`);

      // --- Chord Estimation ---
      const tc = performance.now();
      const chordsVector = (essentia.ChordsDetection(hpcpVectors) as any).chords;
      console.log(`⏱ Chord Estimation: ${(performance.now() - tc).toFixed(2)}ms`);
      const chordsArray = (essentia as any).vectorToArray(chordsVector) as string[];

      const offlineChords: {time: number, chord: string}[] = [];

      for (let i = 0; i < chordsArray.length; i++) {
        let currentChord = chordsArray[i];
        if (!currentChord || currentChord === "NaN") {
          currentChord = "";
        }
        const lastChord = offlineChords[offlineChords.length - 1]?.chord;

        // Only push if the chord changed to keep the timeline clean
        if (currentChord !== lastChord) {
          offlineChords.push({
            time: frameTimes[i],
            chord: currentChord
          });
        }
      }
      chordTimelineRef.current = offlineChords;
      console.table(chordTimelineRef.current);

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
      // TODO: classification ML
      // const voiceData = essentia.VoiceInstrumental(signal);
      // const newLabels = {
      //   ...currentLabelsRef.current,
      //   voice: voiceData.isVoice ? "Vocal" : "Instrumental",
      //   genre: "Detecting...", // Requires further ML models
      //   mood: "Detecting..." 
      // };
      // currentLabelsRef.current = newLabels;
      // setDisplayLabels(newLabels);
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

      // Update Chord Label from pre-calculated timeline
      const currentChordEntry = [...chordTimelineRef.current]
        .reverse()
        .find(entry => entry.time <= t);
      
      if (currentChordEntry && currentChordEntry.chord !== currentLabelsRef.current.chord) {
        currentLabelsRef.current.chord = currentChordEntry.chord;
        setDisplayLabels(prev => ({ ...prev, chord: currentChordEntry.chord }));
      }

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
        const pitchNorm = evt.pitch / 127;
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
      {/* The Center Chord Label */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          pointerEvents: 'none', // Allow clicks to pass through to canvas if needed
          textAlign: 'center',
          zIndex: 10,
        }}
      >
        <div
          style={{
            color: 'white',
            fontSize: '2.5rem',
            fontWeight: 'bold',
            fontFamily: 'monospace',
            textShadow: '0 0 20px rgba(255,255,255,0.5), 0 0 10px rgba(0,255,204,0.3)',
            letterSpacing: '4px',
            transition: 'all 0.2s ease-out', // Smooth transition when chord changes
          }}
        >
          {displayLabels.chord}
        </div>
      </div>
      {/* HUD for high-level classification */}
      <div style={{ position: 'absolute', bottom: 24, left: 24, pointerEvents: 'none' }}>
        <p style={{ color: '#00ffcc', margin: 0, fontSize: '12px' }}>GENRE: {displayLabels.genre}</p>
        <p style={{ color: '#00ffcc', margin: 0, fontSize: '12px' }}>MOOD: {displayLabels.mood}</p>
        <p style={{ color: '#00ffcc', margin: 0, fontSize: '12px' }}>TYPE: {displayLabels.voice}</p>
      </div>
    </div>
  );
}