import { useRef, useEffect, useState } from "react";

/** Type for individual note in a chord segment */
interface MidiNote {
  midi: number;
  isChordTone: boolean;
  noteOn_sec: number;
  noteOff_sec: number;
}

/** Type for a chord segment */
export interface ChordSegment {
  noteOn_sec: number;
  noteOff_sec: number;
  chord: string;
  notes: MidiNote[];
}

interface MidiVisualizerProps {
  segments: ChordSegment[];
  audioRef: React.RefObject<HTMLAudioElement | null>;
}

export function MidiVisualizer({
  segments,
  audioRef
}: MidiVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chordRef = useRef<HTMLDivElement>(null);
  // Persistent refs to track state without triggering effect re-runs
  const currentChordInternal = useRef<string>("");

  const drawNote = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    glowSize: number,
    color: string,
    alpha: number
  ) => {
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
  };

  useEffect(() => {
    const audio = audioRef.current;
    const canvas = canvasRef.current;
    if (!canvas || !audio) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

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
    
    let rafId: number;
    const render = () => {
      const currentTime = audio.currentTime;
      
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const baseRadius = Math.min(width, height) * 0.25;
      const minAlpha = 0.007;

      let activeChord = "";
      const orbitDuration = Math.min(audio.duration, 10); // seconds per orbit
      segments.forEach((seg) => {
        // if (seg.noteOn_sec > currentTime + 0.5) return; 
        // if (seg.noteOff_sec < currentTime - 2.0) return;

        // Find the active chord based on time
        if (currentTime >= seg.noteOn_sec && currentTime <= seg.noteOff_sec) {
          activeChord = seg.chord;
        }

        seg.notes.forEach((note) => {
          const dt = currentTime - note.noteOn_sec;
          const duration = note.noteOff_sec - note.noteOn_sec;
          const fadeTime = 0.8;

          let life = 0;
          if (dt < 0) return;
          else if (dt <= duration) life = 1;
          else if (dt <= duration + fadeTime) life = 1 - (dt - duration) / fadeTime;
          else life = minAlpha;

          if (life <= 0) return;

          // Spatial mapping (88-key piano range normalization)
          const pitchNorm = (note.midi - 21) / (108 - 21);
          const rBase = baseRadius + (pitchNorm - 0.5) * (baseRadius * 5);
          
          // Orbital rotation logic
          const orbitIndex = Math.floor(note.noteOn_sec / orbitDuration);
          const orbitStart = orbitIndex * orbitDuration;
          const orbitProgress = (note.noteOn_sec - orbitStart) / orbitDuration;
          const angle = orbitProgress * Math.PI * 2 + currentTime * 0.2;

          const color = note.isChordTone ? "#8ecae6" : "#ff7f7f";
          const size = 6;

          // Trail rendering logic
          const progress = Math.min(dt / duration, 1);
          const maxTrailSteps = Math.max(Math.floor(duration * 40), 1);
          const trailSteps = Math.max(Math.floor(maxTrailSteps * progress), 1);

          for (let j = 0; j < trailSteps; j++) {
            const trailAlpha = Math.max(life * (j / trailSteps), minAlpha);
            const trailSize = size * (j / trailSteps);
            const trailGlow = trailSize * 2.5;

            // Spiral offset for the trail
            const trailAngle = angle + j * 0.015;
            const x = cx + Math.cos(trailAngle) * rBase;
            const y = cy + Math.sin(trailAngle) * rBase;

            drawNote(ctx, x, y, trailSize, trailGlow, color, trailAlpha);
          }
        });
      });
      
      if (activeChord !== currentChordInternal.current) {
        currentChordInternal.current = activeChord;
        if (chordRef.current) {
          chordRef.current.textContent = activeChord === "N" ? "" : activeChord;
        }
      }
      rafId = requestAnimationFrame(render);
    };
    
    rafId = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(rafId);
      observer.disconnect();
    };
  }, [segments, audioRef]);

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
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      <div 
        ref={chordRef}
        style={{
          position: "absolute",
          bottom: "0%",
          width: "100%",
          textAlign: "center",
          fontSize: "3.5rem",
          fontWeight: "bold",
          color: "#8ecae6",
          textShadow: "0 0 20px rgba(142, 202, 230, 0.8)",
          pointerEvents: "none",
          fontFamily: "monospace",
          letterSpacing: "4px"
        }}
      />
    </div>
  );
}