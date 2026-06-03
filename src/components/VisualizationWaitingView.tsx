// ============================================================
// SYNCED WITH: Hearmi-Frontend/app/studio/components/VisualizationWaitingView.tsx
// Differences vs production:
//   1. Helper functions are imported from ../utils/visDrawHelpers instead of inline.
//   2. Import paths for visMidiHelpers point to local utils/.
// Keep component logic in sync with the Hearmi repo.
// ============================================================

import { useEffect, useRef, useState } from 'react';
import Meyda from 'meyda';
import { PitchDetector } from 'pitchy';
import type { InputData, MidiFeatures, NebulaPuff } from '../utils/visMidiHelpers';
import { analyzeMidi, chordRootHue, spawnNebulaPuff, drawNebulaLayer } from '../utils/visMidiHelpers';
import {
  type FrameFeatures,
  normalizeFeatureArr,
  medianPitch,
  mapPitch,
  getGalaxyColor,
  getMidiGalaxyColor,
  drawChordLabel,
  visualizeNote,
} from '../utils/visDrawHelpers';

export type { FrameFeatures };

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

  const nebulaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const nebulaPuffsRef = useRef<NebulaPuff[]>([]);
  const lastActiveChordRef = useRef<string>('');

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
      const t0 = performance.now();
      const sampleRate = fullBuffer.sampleRate;

      const startSample = Math.floor(input.inputStartTime * sampleRate);
      const endSample = Math.min(
        Math.ceil(input.inputEndTime * sampleRate),
        fullBuffer.length
      );
      const segmentLength = endSample - startSample;
      if (segmentLength <= 0) return;

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

      console.log('Analyzing frame-wise spectral features and rms...');
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

        if (frequency && clarity > 0) {
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
      console.log('Grouping features per note');
      const features: FrameFeatures[] = [];
      let startIdx = -1;
      const pitchThreshold = 0.8;
      for (let i = 1; i < rawPitchMidi.length; i++) {
        const pitch = rawPitchMidi[i];
        if (pitch > 0 && startIdx === -1) startIdx = i;

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
        const step = 8;
        const frameDuration = hopSize / sampleRate;

        for (let i = 0; i < frameTimes.length; i += step) {
          const rms = normRms[i] ?? 0;
          const centroid = normCentroid[i] ?? 0;
          const detectedPitch = rawPitchMidi[i] ?? 0;

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
      console.log('Analysis complete.');
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
    };

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

        const audioInputs: InputData[] = [];
        const midiInputs: InputData[] = [];
        inputs.forEach(input => {
          const isMidi = Array.isArray(input.midiNotes) && input.midiNotes.length > 0;
          if (isMidi) {
            midiInputs.push(input);
          } else {
            audioInputs.push(input);
          }
        });

        await Promise.all(audioInputs.map(inp => runAudioAnalysis(inp, fullBuffer)));
        await Promise.all(midiInputs.map(runMidiAnalysis));

        setStatusText('');
      } catch (error) {
        if (!cancelled) {
          console.error('[VisualizationWaitingView] Analysis failed:', error);
          setStatusText('Unable to analyze preview audio');
        }
      }
    };

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
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // --- Window Resizing Observer ---
    const resize = () => {
      const canvas = canvasRef.current;
      const nebula = nebulaCanvasRef.current;
      if (!canvas || !nebula) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
      }

      nebula.width = Math.round(rect.width * dpr);
      nebula.height = Math.round(rect.height * dpr);
      const nCtx = nebula.getContext('2d');
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
      const orbitDuration = Math.min(totalDuration, 10);

      // --- Nebula layer (drawn behind everything else) ---
      const nebulaCanvas = nebulaCanvasRef.current;
      if (nebulaCanvas) {
        const nCtx = nebulaCanvas.getContext('2d');
        if (nCtx) {
          drawNebulaLayer(nCtx, nebulaPuffsRef.current, performance.now());
        }
      }

      ctx.clearRect(0, 0, w, h);
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
        );
      });

      // Draw MIDI Features
      let activeChord = '';
      midiFeaturesRef.current.forEach((seg) => {
        if (t >= seg.startTime && t <= seg.startTime + seg.duration) {
          activeChord = seg.chord;
        }
        seg.notes.forEach((note) => {
          const dt = t - note.startTime;
          if (dt < 0) return;

          const strength = Math.min(1, Math.max(0, note.velocity / 127)) * 0.5;
          const color = getMidiGalaxyColor(note.pitch, note.velocity, t, note.isChordTone);
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
      if (activeChord && activeChord !== 'N' && activeChord !== lastActiveChordRef.current) {
        lastActiveChordRef.current = activeChord;
        spawnNebulaPuff(activeChord, nebulaPuffsRef.current, w, h);
      }

      if (activeChord && activeChord !== 'N') {
        const currentHue = chordRootHue(activeChord);
        const isMinor = /min|dim|m7/.test(activeChord);
        const currentSat = isMinor ? 45 : 60;
        drawChordLabel(ctx, cx, cy, activeChord, currentHue, currentSat);
      }

      rafRef.current = requestAnimationFrame(draw);
    };

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
