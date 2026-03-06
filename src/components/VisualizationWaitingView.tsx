import { useEffect, useRef, useState } from 'react';
import Meyda from "meyda";
import { PitchDetector } from "pitchy";

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

export function VisualizationWaitingView({
  audioUrl,
  audioRef,
  isVisible = true,
}: VisualizationWaitingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const featuresRef = useRef<FrameFeatures[]>([]);
  const durationRef = useRef<number>(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);

  // Process audio and extract features
  useEffect(() => {
    if (!audioUrl) return;

    const processAudio = async () => {
      try {
        console.log('Decoding audio...');
        
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        const audioBuffer = await audioContextRef.current!.decodeAudioData(arrayBuffer);
        durationRef.current = audioBuffer.duration;

        console.log('Extracting features...');

        const bufferSize = 2048;
        const hopSize = 512;
        const channelData = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;
        Meyda.sampleRate = sampleRate;
        Meyda.bufferSize = bufferSize;

        // Extract Spectral Features and Pitch
        const detector = PitchDetector.forFloat32Array(bufferSize);

        const rawRms: number[] = [];
        const rawCentroid: number[] = [];
        const rawPitch: number[] = [];
        const frameTimes: number[] = [];

        for (let i = 0; i < channelData.length - bufferSize; i += hopSize) {
          const frame = channelData.slice(i, i + bufferSize);

          const f = Meyda.extract(['spectralCentroid', 'rms'], frame);
          if (!f) continue;

          const time = i / sampleRate;
          frameTimes.push(time);

          rawRms.push(f.rms || 0);
          rawCentroid.push(f.spectralCentroid || 0);

          const [frequency, clarity] = detector.findPitch(frame, sampleRate);
          if (frequency && clarity > 0.8) { // clarity threshold
            const midi = 69 + 12 * Math.log2(frequency / 440);
            rawPitch.push(midi);
          } else {
            rawPitch.push(0);
          }

          if (frameTimes.length % 200 === 0) {
            console.log(
              `Extracting... ${Math.round((i / channelData.length) * 100)}%`
            );
            await new Promise(r => setTimeout(r, 0));
          }
        }

        const normRms = normalizeFeatureArr(rawRms);
        const normCentroid = normalizeFeatureArr(rawCentroid);
        
        const features: FrameFeatures[] = [];

        // Compute Note Duration

        let noteStartIdx = 0;
        const pitchThreshold = 0.5; // MIDI tolerance
        for (let i = 1; i < rawPitch.length; i++) {
          const pitchChange =
            Math.abs(rawPitch[i] - rawPitch[noteStartIdx]) > pitchThreshold;

          if (pitchChange || i === rawPitch.length - 1) {
            const startTime = frameTimes[noteStartIdx];
            const endTime = frameTimes[i];
            const duration = endTime - startTime;

            features.push({
              time: startTime,
              duration,
              pitch: rawPitch[noteStartIdx],
              rms: normRms[noteStartIdx],
              centroid: normCentroid[noteStartIdx],
            });

            noteStartIdx = i;
          }
        }
        
        featuresRef.current = features;
        console.table(features);
        console.log('Ready');
      } catch (error) {
        console.error('Error processing audio:', error);
      }
    };

    processAudio();
    
    return () => {
      // Cleanup AudioContext when audioUrl changes or component unmounts
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    };
  }, [audioUrl]);


  // Update visualization based on currentTime (works for both playback and seeking)
  useEffect(()=>{
    if (!isVisible) return;

    const canvas = canvasRef.current as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

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

    const minAlpha = 0.02;

    function draw() {
      const audio = audioRef.current;
      if (!audio) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const events = featuresRef.current;
      const totalDuration = durationRef.current;
      const orbitDuration = Math.min(totalDuration, 10); // seconds per orbit

      if (!events.length || totalDuration === 0) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const t = audio.currentTime;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const cx = w / 2;
      const cy = h / 2;
      const baseRadius = Math.min(w, h) * 0.3;
      ctx.clearRect(0, 0, w, h);

      events.forEach((evt, idx) => {
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
    </div>
  );
}