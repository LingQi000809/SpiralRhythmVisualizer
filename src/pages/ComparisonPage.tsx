// Scaffold for input vs. output audio similarity detection.
// Visualization matches production VisualizationView.tsx logic exactly —
// single orbital ring, same analyzeAudioUrl / drawFeatureNote pipeline.
// Fill in runSimilarityDetection() to add match detection.

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  type FrameFeatures,
  analyzeAudioUrl,
  drawFeatureNote,
  mapPitch,
  ringCenterRadius,
  ringBandHalf,
  drawOrbitRing,
  stemGalaxyColor,
  STEM_HEX,
} from '../utils/visAudioHelpers';
import { type NebulaPuff, drawNebulaLayer } from '../utils/visMidiHelpers';

const DEFAULT_INPUT_URL  = '/data/dragonBoyLofi.wav';
const DEFAULT_OUTPUT_URL = '/data/dragonBoy.wav';

// ─── Similarity detection (SCAFFOLD) ─────────────────────────────────────────

interface SimilarityMatch {
  inputStart:  number;
  inputEnd:    number;
  outputStart: number;
  outputEnd:   number;
  label: string;
  rgb:   [number, number, number];
}

function runSimilarityDetection(
  _inputFeatures:  FrameFeatures[],
  _outputFeatures: FrameFeatures[],
): SimilarityMatch[] {
  // TODO: implement similarity detection
  return [];
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ComparisonPage() {
  const outputAudioRef  = useRef<HTMLAudioElement | null>(null);
  const canvasRef       = useRef<HTMLCanvasElement | null>(null);
  const nebulaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef          = useRef<number | null>(null);

  const [inputUrl,       setInputUrl]       = useState(DEFAULT_INPUT_URL);
  const [outputUrl,      setOutputUrl]      = useState(DEFAULT_OUTPUT_URL);
  const [inputFileName,  setInputFileName]  = useState(DEFAULT_INPUT_URL);
  const [outputFileName, setOutputFileName] = useState(DEFAULT_OUTPUT_URL);
  const [loadingMsg,     setLoadingMsg]     = useState('');
  const [outPlaying,     setOutPlaying]     = useState(false);
  const [outTime,        setOutTime]        = useState(0);
  const [outDur,         setOutDur]         = useState(0);

  // RAF-accessible refs — mutated directly, never trigger re-renders
  const inputFeaturesRef  = useRef<FrameFeatures[]>([]);
  const trackRef  = useRef<HTMLDivElement | null>(null);
  const scrubRef  = useRef(false);
  const outputFeaturesRef = useRef<FrameFeatures[]>([]);
  const outputDurRef      = useRef(0);
  const pitchMedianRef    = useRef(0.5);
  const matchesRef        = useRef<SimilarityMatch[]>([]);
  const nebulaRef         = useRef<NebulaPuff[]>([]);

  // ── Input analysis (background, no loading state) ─────────────────────────
  useEffect(() => {
    let cancelled = false;
    inputFeaturesRef.current = [];
    void analyzeAudioUrl(
      inputUrl,
      (feats) => { if (!cancelled) inputFeaturesRef.current = feats; },
      () => cancelled,
    );
    return () => { cancelled = true; };
  }, [inputUrl]);

  // ── Output analysis ───────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    outputFeaturesRef.current = [];
    outputDurRef.current      = 0;
    pitchMedianRef.current    = 0.5;
    matchesRef.current        = [];
    setLoadingMsg('Analyzing audio…');

    void analyzeAudioUrl(
      outputUrl,
      (feats, dur) => {
        if (cancelled) return;
        outputFeaturesRef.current = feats;
        outputDurRef.current      = dur;
        const pitchNorms = feats
          .map(f => mapPitch(f.pitch))
          .filter(p => p > 0)
          .sort((a, b) => a - b);
        pitchMedianRef.current = pitchNorms.length
          ? pitchNorms[Math.floor(pitchNorms.length / 2)]
          : 0.5;
        setLoadingMsg('');
      },
      () => cancelled,
    );

    return () => { cancelled = true; setLoadingMsg(''); };
  }, [outputUrl]);

  // ── Audio player state ────────────────────────────────────────────────────
  useEffect(() => {
    setOutTime(0); setOutDur(0); setOutPlaying(false);
  }, [outputUrl]);

  useEffect(() => {
    const el = outputAudioRef.current;
    if (!el) return;
    const onTime   = () => setOutTime(el.currentTime);
    const onMeta   = () => setOutDur(isFinite(el.duration) ? el.duration : 0);
    const onPlay   = () => setOutPlaying(true);
    const onPause  = () => setOutPlaying(false);
    const onEnded  = () => { setOutPlaying(false); setOutTime(0); };
    el.addEventListener('timeupdate',     onTime);
    el.addEventListener('durationchange', onMeta);
    el.addEventListener('loadedmetadata', onMeta);
    el.addEventListener('play',           onPlay);
    el.addEventListener('pause',          onPause);
    el.addEventListener('ended',          onEnded);
    return () => {
      el.removeEventListener('timeupdate',     onTime);
      el.removeEventListener('durationchange', onMeta);
      el.removeEventListener('loadedmetadata', onMeta);
      el.removeEventListener('play',           onPlay);
      el.removeEventListener('pause',          onPause);
      el.removeEventListener('ended',          onEnded);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Scrubber helpers ──────────────────────────────────────────────────────
  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
  };

  const seekTo = (clientX: number) => {
    const el = outputAudioRef.current;
    const track = trackRef.current;
    if (!el || !track || !outDur) return;
    const rect = track.getBoundingClientRect();
    el.currentTime = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) * outDur;
  };

  const handleTrackPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    scrubRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    seekTo(e.clientX);
  };
  const handleTrackPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (scrubRef.current) seekTo(e.clientX);
  };
  const handleTrackPointerUp = () => { scrubRef.current = false; };

  // ── Similarity ────────────────────────────────────────────────────────────
  const handleRunSimilarity = useCallback(() => {
    matchesRef.current = runSimilarityDetection(
      inputFeaturesRef.current,
      outputFeaturesRef.current,
    );
    const el = outputAudioRef.current;
    if (el) { el.currentTime = 0; void el.play(); }
  }, []);

  // ── Draw loop (matches VisualizationView logic) ───────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const nebula = nebulaCanvasRef.current;
      if (!nebula) return;
      const dpr  = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      for (const c of [canvas, nebula]) {
        c.width  = Math.round(rect.width  * dpr);
        c.height = Math.round(rect.height * dpr);
        const cx = c.getContext('2d');
        if (cx) { cx.setTransform(1, 0, 0, 1, 0, 0); cx.scale(dpr, dpr); }
      }
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = () => {
      const now  = performance.now();
      const rect = canvas.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      const cx = w / 2, cy = h / 2;
      const baseR = Math.min(w, h) * 0.2;

      const nebula = nebulaCanvasRef.current;
      if (nebula) {
        const nCtx = nebula.getContext('2d');
        if (nCtx) drawNebulaLayer(nCtx, nebulaRef.current, now);
      }

      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1;

      const t        = outputAudioRef.current?.currentTime ?? 0;
      const orbitDur = Math.min(Math.max(outputDurRef.current, 1), 10);
      const feats    = outputFeaturesRef.current;
      const median   = pitchMedianRef.current;

      // Single ring — index 0 of 1, same ringCenterRadius/ringBandHalf as VisualizationView
      const numRings = 1;
      const ringR    = ringCenterRadius(0, numRings, baseR);
      const bandHalf = ringBandHalf(numRings, baseR);

      drawOrbitRing(ctx, cx, cy, ringR, STEM_HEX.other, false, false);

      for (const feat of feats) {
        const pn    = mapPitch(feat.pitch);
        const orbR  = ringR + (pn - median) * bandHalf * 2;
        const color = stemGalaxyColor('other', feat.rms, feat.centroid);
        drawFeatureNote(ctx, feat, t, cx, cy, baseR, orbitDur, 1.0, orbR, color);
      }

      // TODO: draw similarity match puffs — matchesRef.current contains the results

      ctx.globalAlpha = 1;
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, []);

  // ── File handlers ─────────────────────────────────────────────────────────
  const handleInputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setInputFileName(file.name);
    setInputUrl(prev => { if (prev.startsWith('blob:')) URL.revokeObjectURL(prev); return url; });
  }, []);

  const handleOutputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setOutputFileName(file.name);
    setOutputUrl(prev => { if (prev.startsWith('blob:')) URL.revokeObjectURL(prev); return url; });
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={s.page}>
      <audio ref={outputAudioRef} src={outputUrl} style={{ display: 'none' }} />

      {/* Upload row */}
      <div style={s.row}>
        <UploadZone id="cmp-in"  label="Input Audio"  fileName={inputFileName}  onFile={handleInputFile}
          hint="Original track (before separation)" />
        <UploadZone id="cmp-out" label="Output Audio" fileName={outputFileName} onFile={handleOutputFile}
          hint="Separated stem or transformed audio" />
        <button style={s.btn} onClick={handleRunSimilarity}>
          Run Similarity
        </button>
      </div>

      {/* Canvas */}
      <div style={s.canvasWrap}>
        <canvas ref={nebulaCanvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} />
        <canvas ref={canvasRef}       style={{ position: 'relative',  width: '100%', height: '100%' }} />
        {loadingMsg && <div style={s.overlay}>{loadingMsg}</div>}
      </div>

      {/* Output player */}
      <div style={s.playerRow}>
        <button style={s.playerBtn} onClick={() => {
          const el = outputAudioRef.current;
          if (!el) return;
          if (el.paused) void el.play(); else el.pause();
        }}>{outPlaying ? '⏸' : '▶'}</button>
        <div
          ref={trackRef}
          style={s.track}
          onPointerDown={handleTrackPointerDown}
          onPointerMove={handleTrackPointerMove}
          onPointerUp={handleTrackPointerUp}
          onPointerCancel={handleTrackPointerUp}
        >
          <div style={{ ...s.trackFill, width: `${outDur > 0 ? (outTime / outDur) * 100 : 0}%` }} />
          <div style={{ ...s.trackThumb, left: `${outDur > 0 ? (outTime / outDur) * 100 : 0}%` }} />
        </div>
        <span style={s.timeLabel}>{fmt(outTime)} / {fmt(outDur)}</span>
        <span style={s.fileLabel}>{outputFileName}</span>
      </div>
    </div>
  );
}

// ─── Upload zone ──────────────────────────────────────────────────────────────

function UploadZone({ id, label, fileName, onFile, hint }: {
  id: string; label: string; fileName: string; onFile: (f: File) => void; hint?: string;
}) {
  return (
    <div
      style={s.drop}
      onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) onFile(f); }}
      onDragOver={e => e.preventDefault()}
      onClick={() => document.getElementById(id)?.click()}
    >
      <input id={id} type="file" accept="audio/*" style={{ display: 'none' }}
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
      <div style={s.dropLabel}>{label}</div>
      {!fileName && hint && <div style={s.dropHint}>{hint}</div>}
      <div style={fileName ? s.fileLabel : s.dropHint}>{fileName || 'Drop WAV / click to select'}</div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  page:      { display: 'flex', flexDirection: 'column', height: '100%', gap: 10, minHeight: 0 },
  row:       { display: 'flex', gap: 10, alignItems: 'center', flexShrink: 0 },

  drop:      { flex: 1, border: '1px dashed rgba(255,255,255,0.15)', borderRadius: 8, padding: '10px 14px', cursor: 'pointer', userSelect: 'none', minWidth: 0 },
  dropLabel: { fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 2 },
  dropHint:  { fontSize: 12, color: 'rgba(255,255,255,0.25)' },
  fileLabel: { fontSize: 12, color: 'rgba(255,255,255,0.7)' },

  btn: {
    background: 'rgba(255,255,255,0.14)', color: '#fff', border: 'none',
    borderRadius: 6, padding: '7px 18px', fontSize: 13, cursor: 'pointer',
    fontFamily: 'inherit', flexShrink: 0,
  },

  canvasWrap: {
    flex: 1, minHeight: 0, borderRadius: 8,
    border: '1px solid rgba(255,255,255,0.07)',
    background: '#0d0d0d', position: 'relative', overflow: 'hidden',
  },

  overlay: {
    position: 'absolute', inset: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    pointerEvents: 'none',
    color: 'rgba(255,255,255,0.5)', fontSize: 13,
    fontFamily: 'Inter, sans-serif',
  },

  playerRow: { display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 },
  playerBtn: {
    background: 'rgba(255,255,255,0.08)', border: 'none', color: '#fff',
    borderRadius: 6, padding: '5px 14px', fontSize: 14, cursor: 'pointer',
    fontFamily: 'inherit', flexShrink: 0,
  },
  track: {
    flex: 1, height: 4, background: 'rgba(255,255,255,0.12)', borderRadius: 2,
    position: 'relative', cursor: 'pointer', userSelect: 'none',
  },
  trackFill: {
    position: 'absolute', left: 0, top: 0, height: '100%',
    background: 'rgba(255,255,255,0.5)', borderRadius: 2, pointerEvents: 'none',
  },
  trackThumb: {
    position: 'absolute', top: '50%',
    transform: 'translate(-50%, -50%)',
    width: 12, height: 12, borderRadius: '50%', background: '#fff',
    pointerEvents: 'none',
  },
  timeLabel: {
    fontSize: 11, color: 'rgba(255,255,255,0.45)', flexShrink: 0,
    fontVariantNumeric: 'tabular-nums',
  },
};
