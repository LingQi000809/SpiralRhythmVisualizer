// ============================================================
// SYNCED WITH: Hearmi-Frontend/app/studio/components/VisualizationWaitingView.tsx
// Differences vs production:
//   1. Helper functions imported from ../utils/visAudioHelpers (not inline).
//   2. Import paths for visMidiHelpers point to local utils/.
//   3. suckInRef prop (local-only: canvas suck-to-center animation).
// Keep component logic in sync with the Hearmi repo.
// ============================================================

import { useEffect, useRef, useState } from 'react';
import type { InputData, MidiFeatures, NebulaPuff } from '../utils/visMidiHelpers';
import {
  analyzeMidi,
  chordRootHue,
  getMidiGalaxyColor,
  spawnNebulaPuff,
  drawNebulaLayer,
} from '../utils/visMidiHelpers';
import {
  type FrameFeatures,
  analyzeAudioUrl,
  drawFeatureNote,
  drawChordLabel,
  visualizeNote,
} from '../utils/visAudioHelpers';

interface SuckInState { isActive: boolean; startTime: number; durationMs: number; }

interface VisualizationWaitingViewProps {
  concatenatedAudioUrl: string | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  inputs: InputData[] | null;
  isVisible?: boolean;
  // Local-only: parent mutates this ref to trigger the suck-to-center animation.
  suckInRef?: React.RefObject<SuckInState>;
}

export function VisualizationWaitingView({
  concatenatedAudioUrl,
  audioRef,
  suckInRef,
  inputs,
  isVisible = true,
}: VisualizationWaitingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const frameFeaturesRef = useRef<FrameFeatures[]>([]);
  const midiFeaturesRef  = useRef<MidiFeatures[]>([]);
  const durationRef      = useRef<number>(0);
  const rafRef           = useRef<number | null>(null);

  const nebulaCanvasRef    = useRef<HTMLCanvasElement | null>(null);
  const nebulaPuffsRef     = useRef<NebulaPuff[]>([]);
  const lastActiveChordRef = useRef<string>('');

  const [statusText, setStatusText] = useState('Preparing visualization...');

  // --- EFFECT 1: OFFLINE ANALYSIS ---
  useEffect(() => {
    if (!concatenatedAudioUrl || !inputs) {
      frameFeaturesRef.current = [];
      midiFeaturesRef.current  = [];
      durationRef.current      = 0;
      setStatusText('Preparing MIDI preview...');
      return;
    }

    let cancelled = false;

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

        durationRef.current = Math.max(0, ...inputs.map(i => i.inputEndTime));

        const audioInputs = inputs.filter(i => !(Array.isArray(i.midiNotes) && i.midiNotes.length > 0));
        const midiInputs  = inputs.filter(i =>   Array.isArray(i.midiNotes) && i.midiNotes.length > 0);

        await Promise.all(audioInputs.map(inp =>
          analyzeAudioUrl(inp.audioUrl, (features) => {
            const shifted = features.map(f => ({ ...f, time: f.time + inp.inputStartTime }));
            frameFeaturesRef.current = frameFeaturesRef.current.concat(shifted);
          }, () => cancelled)
        ));
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

    return () => { cancelled = true; };
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

      const dpr  = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      canvas.width  = Math.round(rect.width  * dpr);
      canvas.height = Math.round(rect.height * dpr);
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
      }

      nebula.width  = Math.round(rect.width  * dpr);
      nebula.height = Math.round(rect.height * dpr);
      const nCtx = nebula.getContext('2d');
      if (nCtx) {
        nCtx.setTransform(1, 0, 0, 1, 0, 0);
        nCtx.scale(dpr, dpr);
      }
    };

    resize();
    const observer = new ResizeObserver(() => { resize(); });
    observer.observe(canvas);

    // --- Animation Loop ---
    const draw = () => {
      const audio = audioRef.current;
      if (!audio || (!frameFeaturesRef.current.length && !midiFeaturesRef.current.length)) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const t    = audio.currentTime;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      const cx = w / 2, cy = h / 2;
      const baseRadius      = Math.min(w, h) * 0.2;
      const totalDuration   = durationRef.current;
      const orbitDuration   = Math.min(totalDuration, 10); // seconds per orbit

      // Suck-to-center: shrink orbital radius and fade alpha using easeIn3 curve.
      const suck = suckInRef?.current;
      let effectiveBaseR = baseRadius;
      let suckAlpha = 1;
      if (suck?.isActive) {
        const p = Math.min((performance.now() - suck.startTime) / suck.durationMs, 1);
        const e = p * p * p; // easeIn3
        effectiveBaseR = baseRadius * (1 - e);
        suckAlpha      = 1 - e;
      }

      // --- Nebula layer (drawn behind everything else) ---
      const nebulaCanvas = nebulaCanvasRef.current;
      if (nebulaCanvas) {
        const nCtx = nebulaCanvas.getContext('2d');
        if (nCtx) drawNebulaLayer(nCtx, nebulaPuffsRef.current, performance.now());
      }

      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1.0;

      // Draw Frame Features
      frameFeaturesRef.current.forEach((evt) => {
        drawFeatureNote(ctx, evt, t, cx, cy, effectiveBaseR, orbitDuration, suckAlpha);
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
          const color    = getMidiGalaxyColor(note.pitch, note.velocity, t, note.isChordTone);
          const size     = 10 + strength * 5;
          const glowSize = size * 2 + strength * 5;
          visualizeNote(
            ctx,
            dt, t,
            note.startTime, note.duration, note.pitch, strength,
            cx, cy, effectiveBaseR, color, size, glowSize,
            orbitDuration,
            true,
            suckAlpha
          );
        });
      });

      // Spawn a nebula puff whenever the active chord changes
      if (activeChord && activeChord !== 'N' && activeChord !== lastActiveChordRef.current) {
        lastActiveChordRef.current = activeChord;
        spawnNebulaPuff(activeChord, nebulaPuffsRef.current, w, h);
      }

      // Hide chord label during suck — it stays at center and looks wrong
      if (suckAlpha > 0.05 && activeChord && activeChord !== 'N') {
        const currentHue = chordRootHue(activeChord);
        const isMinor    = /min|dim|m7/.test(activeChord);
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
