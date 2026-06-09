// LOCAL DEV SCAFFOLDING — POC for audio music similarity detection.
// Accepts an input audio and a separated output stem, renders one orbital ring
// (output visualization style) and marks similar segments as motif puffs.
// Similarity algorithm TBA — currently uses hardcoded SIMILARITY_MATCHES.

import { useRef, useState, useCallback, useEffect } from 'react';
import {
  type FrameFeatures,
  type PitchDetectorType,
  mapPitch,
  drawFeatureNote,
  analyzeAudioUrl,
} from '../utils/visDrawHelpers';
import {
  ringCenterRadius,
  ringBandHalf,
  drawOrbitRing,
  stemGalaxyColor,
} from '../utils/stemVizHelpers';

// ─── Types ────────────────────────────────────────────────────────────────────

interface SimilarityMatch {
  inputStart: number; inputEnd: number;
  outputStart: number; outputEnd: number;
  label: string; rgb: [number, number, number];
}

type Phase = 'idle' | 'ready';

// ─── [Scaffold] Hard-coded similarity — replace with backend analysis ─────────
const SIMILARITY_MATCHES: SimilarityMatch[] = [
  { inputStart: 2.0,  inputEnd: 4.0,  outputStart: 0.0,  outputEnd: 1.0,  label: 'Motif A', rgb: [255, 140, 80]  },
  { inputStart: 2.0,  inputEnd: 4.0,  outputStart: 4.0,  outputEnd: 5.0,  label: 'Motif A', rgb: [255, 140, 80]  },
  { inputStart: 7.0,  inputEnd: 9.0,  outputStart: 12.0, outputEnd: 16.0, label: 'Motif B', rgb: [80,  220, 170] },
];

const DEFAULT_INPUT_URL  = '/data/compare1_hp.wav';
const DEFAULT_OUTPUT_URL = '/data/compare2_hp.wav';

const RING_COLOR   = '#7B8FFF';
const RING_LABEL   = 'output stem';
const PUFF_SPAWN_MS    = 1300;
const PUFF_HIT_PX      = 60;
const PUFF_FADE_DELAY_S = 2.0;
const PUFF_FADE_DUR_S   = 1.5;

// ─── Drawing helpers ──────────────────────────────────────────────────────────

function easeOut3(t: number) { return 1 - Math.pow(1 - t, 3); }
function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function drawSpanningPuff(
  ctx: CanvasRenderingContext2D,
  positions: { x: number; y: number }[],
  anchor:    { x: number; y: number },
  match: SimilarityMatch,
  now: number,
  fadeAlpha: number,
) {
  const [r, g, b] = match.rgb;
  const pulse    = 1 + Math.sin(now * 0.002) * 0.12;
  const n        = Math.max(1, positions.length);
  const strength = 0.28 / Math.sqrt(n) * fadeAlpha;
  const radius   = 22 * pulse;

  ctx.globalCompositeOperation = 'screen';
  for (const pos of positions) {
    const grad = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius * 2.8);
    grad.addColorStop(0,    `rgba(${r},${g},${b},${strength})`);
    grad.addColorStop(0.4,  `rgba(${r},${g},${b},${strength * 0.45})`);
    grad.addColorStop(0.75, `rgba(${r},${g},${b},${strength * 0.12})`);
    grad.addColorStop(1,    `rgba(${r},${g},${b},0)`);
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(pos.x, pos.y, radius * 2.8, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalCompositeOperation = 'source-over';

  ctx.globalAlpha = 0.65 * fadeAlpha;
  ctx.fillStyle   = `rgb(${r},${g},${b})`;
  ctx.font        = '11px Inter, sans-serif';
  ctx.textAlign   = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(match.label, anchor.x, anchor.y + radius + 4);
  ctx.globalAlpha = 1; ctx.textBaseline = 'alphabetic';
}

// Orbital XY for a feature event on the single ring — mirrors StemVisualizationView.featOrbitalPos
function featOrbitalPos(
  feat: FrameFeatures,
  audioTime: number,
  pitchMedian: number,
  ringR: number, bandHalf: number,
  cx: number, cy: number,
  orbitDur: number,
) {
  const pn   = mapPitch(feat.pitch);
  const dev  = (pn - pitchMedian) * bandHalf * 2;
  const oi   = Math.floor(feat.time / orbitDur);
  const op   = (feat.time - oi * orbitDur) / orbitDur;
  const ang  = op * Math.PI * 2 + oi * 0.3 + audioTime * 0.1;
  const orbR = ringR + dev;
  return { x: cx + Math.cos(ang) * orbR, y: cy + Math.sin(ang) * orbR };
}

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

// ─── Component ────────────────────────────────────────────────────────────────

type MatchState = { anchorIdx: number; windowIndices: number[]; spawnTime: number } | null;

export default function ComparisonPage() {
  const inputAudioRef  = useRef<HTMLAudioElement | null>(null);
  const outputAudioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef      = useRef<HTMLCanvasElement | null>(null);

  const [inputUrl,       setInputUrl]       = useState<string | null>(DEFAULT_INPUT_URL);
  const [outputUrl,      setOutputUrl]      = useState<string | null>(DEFAULT_OUTPUT_URL);
  const [inputFileName,  setInputFileName]  = useState('input.wav');
  const [outputFileName, setOutputFileName] = useState('output.wav');
  const [phase,           setPhase]          = useState<Phase>('idle');
  const [selectedMatch,   setSelectedMatch]  = useState<SimilarityMatch | null>(null);
  const [playingSnippet,  setPlayingSnippet] = useState<'input' | 'output' | null>(null);
  const [pitchDetector,   setPitchDetector]  = useState<PitchDetectorType>('pitchy');
  const [isAnalyzing,     setIsAnalyzing]    = useState(false);

  // Output player state
  const [outTime,    setOutTime]    = useState(0);
  const [outDur,     setOutDur]     = useState(0);
  const [outPlaying, setOutPlaying] = useState(false);

  // RAF-accessible refs
  const pendingPlayRef     = useRef(false);
  const outputFeaturesRef  = useRef<FrameFeatures[]>([]);
  const outputDurRef       = useRef(0);
  const pitchMedianRef     = useRef(0.5);
  const matchStatesRef    = useRef<MatchState[]>(SIMILARITY_MATCHES.map(() => null));
  const puffHitRef        = useRef<Array<{ anchor: {x:number;y:number}; windowPos: {x:number;y:number}[]; match: SimilarityMatch }>>([]);
  const selectedMatchRef  = useRef<SimilarityMatch | null>(null);
  const playingSnippetRef = useRef<'input' | 'output' | null>(null);
  const snippetTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => { selectedMatchRef.current  = selectedMatch;  }, [selectedMatch]);
  useEffect(() => { playingSnippetRef.current = playingSnippet; }, [playingSnippet]);

  // ── Output player listeners ───────────────────────────────────────────────
  useEffect(() => {
    const el = outputAudioRef.current;
    if (!el) return;
    const onTime  = () => setOutTime(el.currentTime);
    const onDur   = () => setOutDur(el.duration || 0);
    const onPlay  = () => setOutPlaying(true);
    const onPause = () => setOutPlaying(false);
    const onEnded = () => setOutPlaying(false);
    el.addEventListener('timeupdate',     onTime);
    el.addEventListener('durationchange', onDur);
    el.addEventListener('loadedmetadata', onDur);
    el.addEventListener('play',           onPlay);
    el.addEventListener('pause',          onPause);
    el.addEventListener('ended',          onEnded);
    return () => {
      el.removeEventListener('timeupdate',     onTime);
      el.removeEventListener('durationchange', onDur);
      el.removeEventListener('loadedmetadata', onDur);
      el.removeEventListener('play',           onPlay);
      el.removeEventListener('pause',          onPause);
      el.removeEventListener('ended',          onEnded);
    };
  }, []);

  // ── Output analysis (shared by file drop and default URL) ────────────────
  const loadOutputUrl = useCallback((url: string, name: string, detector: PitchDetectorType) => {
    setOutputFileName(name);
    setOutputUrl(prev => { if (prev?.startsWith('blob:')) URL.revokeObjectURL(prev); return url; });
    outputFeaturesRef.current = [];
    outputDurRef.current      = 0;
    pitchMedianRef.current    = 0.5;
    matchStatesRef.current    = SIMILARITY_MATCHES.map(() => null);
    puffHitRef.current        = [];
    if (detector === 'basic-pitch') setIsAnalyzing(true);
    let cancelled = false;
    void analyzeAudioUrl(
      url,
      (feats: FrameFeatures[], dur: number) => {
        if (cancelled) return;
        outputFeaturesRef.current = feats;
        outputDurRef.current      = dur;
        const pitches = feats.map(f => mapPitch(f.pitch)).filter(p => p > 0).sort((a, b) => a - b);
        pitchMedianRef.current = pitches.length ? pitches[Math.floor(pitches.length / 2)] : 0.5;
        setIsAnalyzing(false);
        if (pendingPlayRef.current) {
          pendingPlayRef.current = false;
          const el = outputAudioRef.current;
          if (el) { el.currentTime = 0; void el.play(); }
        }
      },
      () => cancelled,
      0.01,
      detector,
    );
    return () => { cancelled = true; setIsAnalyzing(false); };
  }, []);

  // ── Load defaults on mount ────────────────────────────────────────────────
  useEffect(() => {
    loadOutputUrl(DEFAULT_OUTPUT_URL, 'compare2.wav', 'pitchy');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── File handlers ──────────────────────────────────────────────────────────
  const handleInputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setInputFileName(file.name);
    setInputUrl(prev => { if (prev?.startsWith('blob:')) URL.revokeObjectURL(prev); return url; });
  }, []);

  const handleOutputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    loadOutputUrl(url, file.name, pitchDetector);
  }, [loadOutputUrl, pitchDetector]);

  // ── Phase control ──────────────────────────────────────────────────────────
  const handleStart = useCallback(() => {
    if (!outputUrl) return;
    matchStatesRef.current = SIMILARITY_MATCHES.map(() => null);
    puffHitRef.current     = [];
    setSelectedMatch(null); setPlayingSnippet(null);
    setPhase('ready');
    if (outputFeaturesRef.current.length > 0) {
      // Analysis already done — play immediately
      const el = outputAudioRef.current;
      if (el) { el.currentTime = 0; void el.play(); }
    } else {
      // Analysis still running — defer playback until onDone fires
      pendingPlayRef.current = true;
    }
  }, [outputUrl]);

  const handleReset = useCallback(() => {
    pendingPlayRef.current = false;
    outputAudioRef.current?.pause();
    inputAudioRef.current?.pause();
    matchStatesRef.current = SIMILARITY_MATCHES.map(() => null);
    puffHitRef.current     = [];
    setPhase('idle');
    setSelectedMatch(null); setPlayingSnippet(null);
  }, []);

  // ── Canvas draw loop ───────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const dpr  = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width  = Math.round(rect.width  * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(dpr, dpr);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    let rafId: number;

    const draw = () => {
      const rect  = canvas.getBoundingClientRect();
      const w = rect.width, h = rect.height, cx = w / 2, cy = h / 2;
      const baseR    = Math.min(w, h) * 0.25;
      const ringR    = ringCenterRadius(0, 1, baseR);
      const bandH    = ringBandHalf(1, baseR);
      const now      = performance.now();
      const outT     = outputAudioRef.current?.currentTime ?? 0;
      const dur      = outputDurRef.current || 30;
      const orbitDur = Math.min(dur, 10);
      const feats    = outputFeaturesRef.current;
      const median   = pitchMedianRef.current;

      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1;

      // ── Orbit ring ──────────────────────────────────────────────────────────
      drawOrbitRing(ctx, cx, cy, ringR, RING_COLOR, false, false);

      ctx.save();
      ctx.globalAlpha  = 0.4;
      ctx.fillStyle    = RING_COLOR;
      ctx.font         = 'bold 12px Inter, sans-serif';
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(RING_LABEL.toUpperCase(), cx + ringR + 12, cy);
      ctx.restore();

      // ── Similarity puffs — drawn behind stars ───────────────────────────────
      const newHits: typeof puffHitRef.current = [];

      SIMILARITY_MATCHES.forEach((match, idx) => {
        if (outT < match.outputStart) { matchStatesRef.current[idx] = null; return; }

        if (!matchStatesRef.current[idx]) {
          const windowIndices: number[] = [];
          feats.forEach((f, fi) => {
            if (f.time >= match.outputStart && f.time <= match.outputEnd) windowIndices.push(fi);
          });
          let anchorIdx = 0, bestD = Infinity;
          feats.forEach((f, fi) => {
            const d = Math.abs(f.time - match.outputStart);
            if (d < bestD) { bestD = d; anchorIdx = fi; }
          });
          if (!windowIndices.length) windowIndices.push(anchorIdx);
          matchStatesRef.current[idx] = { anchorIdx, windowIndices, spawnTime: now };
        }

        const state    = matchStatesRef.current[idx]!;
        const progress = Math.min((now - state.spawnTime) / PUFF_SPAWN_MS, 1);
        const aFeat    = feats[state.anchorIdx];
        const anchor   = aFeat
          ? featOrbitalPos(aFeat, outT, median, ringR, bandH, cx, cy, orbitDur)
          : { x: cx + ringR, y: cy };

        if (progress < 1) {
          // Fly-in comet from canvas center
          const [r, g, b] = match.rgb;
          for (let trail = 0; trail < 10; trail++) {
            const tp = Math.max(0, progress - trail * 0.018);
            const te = easeOut3(tp);
            ctx.globalAlpha = (0.85 - trail * 0.08) * (1 - progress * 0.15);
            ctx.fillStyle   = `rgba(${r},${g},${b},1)`;
            ctx.beginPath();
            ctx.arc(lerp(cx, anchor.x, te), lerp(cy, anchor.y, te),
              Math.max(10 - trail, 1) * 0.8, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.globalAlpha = 1;
        } else {
          const fadeAlpha = outT > match.outputEnd + PUFF_FADE_DELAY_S
            ? Math.max(0, 1 - (outT - match.outputEnd - PUFF_FADE_DELAY_S) / PUFF_FADE_DUR_S)
            : 1;
          if (fadeAlpha <= 0) return;
          const windowPositions = state.windowIndices.map(fi => {
            const f = feats[fi];
            return f ? featOrbitalPos(f, outT, median, ringR, bandH, cx, cy, orbitDur) : anchor;
          });
          drawSpanningPuff(ctx, windowPositions, anchor, match, now, fadeAlpha);
          newHits.push({ anchor, windowPos: windowPositions, match });
        }
      });
      puffHitRef.current = newHits;

      // ── Stars on ring ────────────────────────────────────────────────────────
      // Age-fade constants: start fading at 2 orbits, fully gone by 3 orbits.
      // The 1-orbit fade window prevents the abrupt-disappearance artifact.
      const AGE_FADE_START  = 2;
      const AGE_FADE_WINDOW = 1;

      for (const feat of feats) {
        // Smooth age-based fade — events older than 2 orbital cycles fade out
        // gradually over the next full cycle so there's no sudden pop.
        const ageOrbits = (outT - feat.time) / orbitDur;
        if (ageOrbits >= AGE_FADE_START + AGE_FADE_WINDOW) continue;
        const ageAlpha = ageOrbits < AGE_FADE_START ? 1
          : 1 - (ageOrbits - AGE_FADE_START) / AGE_FADE_WINDOW;

        const isHarmony = feat.isMelody === false;

        // Melody notes spread wider radially; harmony compressed to a tighter band
        const spread = isHarmony ? 1.2 : 3.5;
        const orbR   = ringR + (mapPitch(feat.pitch) - median) * bandH * spread;
        const color  = stemGalaxyColor('other', feat.rms, feat.centroid);
        const alpha  = isHarmony ? 0.18 : 1.0;
        const sizeScale = isHarmony ? 0.35 : 1.0;

        drawFeatureNote(ctx, feat, outT, cx, cy, baseR, orbitDur,
          alpha * ageAlpha, orbR, color, sizeScale);

        // Pitch label: salient melody only, full opacity, only while note is sounding
        if (!isHarmony && feat.pitchLabel) {
          const dt = outT - feat.time;
          if (dt >= 0 && dt <= feat.duration) {
            const pos = featOrbitalPos(feat, outT, median, ringR, bandH, cx, cy, orbitDur);
            ctx.save();
            ctx.globalAlpha  = 1;
            ctx.font         = 'bold 9px Inter, sans-serif';
            ctx.fillStyle    = '#fff';
            ctx.textAlign    = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText(feat.pitchLabel, pos.x, pos.y - 8);
            ctx.restore();
          }
        }
      }

      ctx.globalAlpha = 1;
      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(rafId); ro.disconnect(); };
  }, []);

  // ── Canvas click / hover ───────────────────────────────────────────────────
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedMatchRef.current) {
      if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
      inputAudioRef.current?.pause();
      setSelectedMatch(null); setPlayingSnippet(null);
      void outputAudioRef.current?.play();
      return;
    }
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    for (const hit of puffHitRef.current) {
      const over =
        Math.hypot(mx - hit.anchor.x, my - hit.anchor.y) < PUFF_HIT_PX ||
        hit.windowPos.some(p => Math.hypot(mx - p.x, my - p.y) < PUFF_HIT_PX);
      if (over) {
        outputAudioRef.current?.pause();
        if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
        setSelectedMatch(hit.match);
        setPlayingSnippet(null);
        return;
      }
    }
  }, []);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const over = puffHitRef.current.some(hit =>
      Math.hypot(mx - hit.anchor.x, my - hit.anchor.y) < PUFF_HIT_PX ||
      hit.windowPos.some(p => Math.hypot(mx - p.x, my - p.y) < PUFF_HIT_PX),
    );
    canvasRef.current.style.cursor = over ? 'pointer' : 'default';
  }, []);

  // ── Snippet playback ───────────────────────────────────────────────────────
  const playSnippet = useCallback((type: 'input' | 'output') => {
    const match = selectedMatchRef.current;
    if (!match) return;

    if (playingSnippetRef.current === type) {
      if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
      (type === 'input' ? inputAudioRef : outputAudioRef).current?.pause();
      setPlayingSnippet(null);
      return;
    }

    if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
    inputAudioRef.current?.pause();
    outputAudioRef.current?.pause();

    const audio = (type === 'input' ? inputAudioRef : outputAudioRef).current;
    const start = type === 'input' ? match.inputStart  : match.outputStart;
    const end   = type === 'input' ? match.inputEnd    : match.outputEnd;

    if (audio) {
      audio.currentTime = start;
      void audio.play();
      setPlayingSnippet(type);
      snippetTimerRef.current = setTimeout(() => {
        audio.pause();
        setPlayingSnippet(null);
      }, (end - start) * 1000);
    }
  }, []);

  const closePanel = useCallback(() => {
    if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
    inputAudioRef.current?.pause();
    setSelectedMatch(null); setPlayingSnippet(null);
    void outputAudioRef.current?.play();
  }, []);

  // ── Output player controls ─────────────────────────────────────────────────
  const toggleOutput = useCallback(() => {
    const el = outputAudioRef.current;
    if (!el) return;
    if (el.paused) void el.play(); else el.pause();
  }, []);

  const replayOutput = useCallback(() => {
    const el = outputAudioRef.current;
    if (!el) return;
    matchStatesRef.current = SIMILARITY_MATCHES.map(() => null);
    puffHitRef.current     = [];
    setSelectedMatch(null); setPlayingSnippet(null);
    el.currentTime = 0; void el.play();
  }, []);

  const seekPointer = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const rect  = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const el = outputAudioRef.current;
    if (el && outDur) el.currentTime = ratio * outDur;
  }, [outDur]);

  const handleTrackPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    seekPointer(e);
  }, [seekPointer]);

  const handleTrackPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.buttons === 0) return;
    seekPointer(e);
  }, [seekPointer]);

  // ── Render ─────────────────────────────────────────────────────────────────
  const pct = outDur > 0 ? (outTime / outDur) * 100 : 0;

  return (
    <div style={s.page}>
      <audio ref={inputAudioRef}  src={inputUrl  ?? undefined} style={{ display: 'none' }} />
      <audio ref={outputAudioRef} src={outputUrl ?? undefined} style={{ display: 'none' }} />

      {/* Upload row */}
      <div style={s.row}>
        <UploadZone id="sim-in"  label="Input Audio"  fileName={inputFileName}  onFile={handleInputFile}
          hint="Original track (before separation)" />
        <UploadZone id="sim-out" label="Output Stem"  fileName={outputFileName} onFile={handleOutputFile}
          hint="Separated stem (e.g. vocals, bass)" />
      </div>

      {/* Controls */}
      <div style={s.row}>
        {phase === 'idle' ? (
          <button
            style={{ ...s.btn, opacity: outputUrl ? 1 : 0.4 }}
            disabled={!outputUrl}
            onClick={handleStart}
          >
            Analyze
          </button>
        ) : (
          <button style={{ ...s.btn, ...s.btnGhost }} onClick={handleReset}>Reset</button>
        )}

        {/* Pitch detector toggle */}
        <div style={s.toggle}>
          {(['pitchy', 'basic-pitch'] as PitchDetectorType[]).map(d => (
            <button
              key={d}
              style={{ ...s.toggleBtn, ...(pitchDetector === d ? s.toggleBtnActive : {}) }}
              onClick={() => {
                setPitchDetector(d);
                if (outputUrl) loadOutputUrl(outputUrl, outputFileName, d);
              }}
            >
              {d}
            </button>
          ))}
        </div>

        <span style={s.statusText}>
          {phase === 'idle' && !outputFileName && 'Upload audio to begin'}
          {phase === 'idle' &&  outputFileName && 'Click Analyze to start'}
          {phase === 'ready' && !selectedMatch && 'Click a glowing puff to inspect the matching segment'}
        </span>
      </div>

      {/* Canvas + overlay panel */}
      <div style={s.canvasWrap}>
        <canvas
          ref={canvasRef}
          style={{ ...s.canvas, filter: selectedMatch ? 'brightness(0.35)' : 'none' }}
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMouseMove}
        />
        {isAnalyzing && (
          <div style={s.loadingOverlay}>
            <span style={s.loadingText}>Analyzing with basic-pitch…</span>
          </div>
        )}
        {selectedMatch && (
          <SnippetPanel
            match={selectedMatch}
            playing={playingSnippet}
            hasInput={!!inputUrl}
            onPlay={playSnippet}
            onClose={closePanel}
          />
        )}
      </div>

      {/* Output player bar */}
      {phase === 'ready' && !selectedMatch && (
        <div style={s.playerBar}>
          <button style={s.playerBtn} onClick={toggleOutput}>
            {outPlaying ? '⏸' : '▶'}
          </button>
          <div
            style={s.trackOuter}
            onPointerDown={handleTrackPointerDown}
            onPointerMove={handleTrackPointerMove}
          >
            <div style={s.trackInner}>
              <div style={{ ...s.trackFill, width: `${pct}%` }} />
              <div style={{ ...s.trackThumb, left: `${pct}%` }} />
            </div>
          </div>
          <span style={s.playerTime}>{fmt(outTime)} / {fmt(outDur)}</span>
          <button style={s.playerBtn} onClick={replayOutput} title="Replay">↺</button>
        </div>
      )}
    </div>
  );
}

// ─── Snippet panel ─────────────────────────────────────────────────────────────

function SnippetPanel({ match, playing, hasInput, onPlay, onClose }: {
  match: SimilarityMatch;
  playing: 'input' | 'output' | null;
  hasInput: boolean;
  onPlay: (t: 'input' | 'output') => void;
  onClose: () => void;
}) {
  const [r, g, b] = match.rgb;
  const accent    = `rgb(${r},${g},${b})`;
  const accentDim = `rgba(${r},${g},${b},0.18)`;

  return (
    <div style={s.panel}>
      <div style={s.panelHeader}>
        <span style={{ ...s.panelDot, background: accent }} />
        <span style={{ ...s.panelTitle, color: accent }}>{match.label}</span>
        <button style={s.panelClose} onClick={onClose}>✕</button>
      </div>
      <p style={s.panelSub}>Play each snippet to hear the similarity</p>
      <div style={s.snippetRow}>
        {hasInput && (
          <SnippetButton
            label="Input" start={match.inputStart} end={match.inputEnd}
            isActive={playing === 'input'} accent={accent} accentDim={accentDim}
            onClick={() => onPlay('input')}
          />
        )}
        <SnippetButton
          label="Output stem" start={match.outputStart} end={match.outputEnd}
          isActive={playing === 'output'} accent={accent} accentDim={accentDim}
          onClick={() => onPlay('output')}
        />
      </div>
      <button style={s.resumeBtn} onClick={onClose}>Resume playback</button>
    </div>
  );
}

function SnippetButton({ label, start, end, isActive, accent, accentDim, onClick }: {
  label: string; start: number; end: number;
  isActive: boolean; accent: string; accentDim: string; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        ...s.snippetBtn,
        background:  isActive ? accentDim : 'rgba(255,255,255,0.04)',
        border:      `1px solid ${isActive ? accent : 'rgba(255,255,255,0.09)'}`,
        color:       isActive ? accent : 'rgba(255,255,255,0.72)',
      }}
    >
      <span style={s.snippetIcon}>{isActive ? '⏸' : '▶'}</span>
      <span>
        <div style={s.snippetLabel}>{label}</div>
        <div style={s.snippetTime}>{start.toFixed(1)}s – {end.toFixed(1)}s</div>
      </span>
    </button>
  );
}

// ─── Upload zone ───────────────────────────────────────────────────────────────

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
      <div style={fileName ? s.fileName : s.dropHint}>{fileName || 'Drop WAV / click to select'}</div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  page:        { display: 'flex', flexDirection: 'column', height: '100%', gap: '10px', minHeight: 0 },
  row:         { display: 'flex', gap: '10px', alignItems: 'center', flexShrink: 0 },

  drop:        { flex: 1, border: '1px dashed rgba(255,255,255,0.15)', borderRadius: '8px', padding: '12px 16px', cursor: 'pointer', userSelect: 'none' },
  dropLabel:   { fontSize: '10px', fontWeight: 700, color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '2px' },
  dropHint:    { fontSize: '12px', color: 'rgba(255,255,255,0.25)' },
  fileName:    { fontSize: '12px', color: 'rgba(255,255,255,0.7)' },

  btn:         { background: 'rgba(255,255,255,0.14)', color: '#fff', border: 'none', borderRadius: '6px', padding: '7px 18px', fontSize: '13px', cursor: 'pointer', fontFamily: 'inherit', flexShrink: 0 },
  btnGhost:    { background: 'rgba(255,255,255,0.06)' },
  statusText:  { fontSize: '13px', color: 'rgba(255,255,255,0.4)' },

  canvasWrap:  { flex: 1, minHeight: 0, borderRadius: '10px', border: '1px solid rgba(255,255,255,0.07)', overflow: 'hidden', background: '#0d0d0d', position: 'relative' },
  canvas:      { width: '100%', height: '100%', display: 'block', transition: 'filter 0.4s ease' },

  panel:       { position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: '300px', background: 'rgba(14,14,20,0.97)', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.1)', padding: '22px 22px 18px', display: 'flex', flexDirection: 'column', gap: '14px', backdropFilter: 'blur(20px)', boxShadow: '0 28px 70px rgba(0,0,0,0.75)', zIndex: 20 },
  panelHeader: { display: 'flex', alignItems: 'center', gap: '10px' },
  panelDot:    { width: '10px', height: '10px', borderRadius: '50%', flexShrink: 0 },
  panelTitle:  { fontWeight: 700, fontSize: '16px', flex: 1 },
  panelClose:  { background: 'transparent', border: 'none', color: 'rgba(255,255,255,0.3)', cursor: 'pointer', fontSize: '16px', padding: '0 2px', fontFamily: 'inherit', lineHeight: 1 },
  panelSub:    { margin: 0, fontSize: '12px', color: 'rgba(255,255,255,0.3)' },

  snippetRow:  { display: 'flex', flexDirection: 'column', gap: '7px' },
  snippetBtn:  { display: 'flex', alignItems: 'center', gap: '12px', padding: '11px 14px', borderRadius: '8px', cursor: 'pointer', fontFamily: 'inherit', textAlign: 'left', transition: 'background 0.15s, border-color 0.15s, color 0.15s' },
  snippetIcon: { fontSize: '15px', width: '18px', textAlign: 'center', flexShrink: 0 },
  snippetLabel:{ fontSize: '13px', fontWeight: 600, lineHeight: '1.3' },
  snippetTime: { fontSize: '11px', color: 'rgba(255,255,255,0.35)', marginTop: '1px' },

  resumeBtn:   { background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.09)', color: 'rgba(255,255,255,0.5)', borderRadius: '7px', padding: '9px', fontSize: '12px', cursor: 'pointer', fontFamily: 'inherit', width: '100%' },

  playerBar:   { display: 'flex', alignItems: 'center', gap: '10px', padding: '9px 14px', background: 'rgba(255,255,255,0.04)', borderRadius: '8px', flexShrink: 0 },
  playerBtn:   { background: 'transparent', border: 'none', color: 'rgba(255,255,255,0.65)', cursor: 'pointer', fontSize: '16px', padding: '2px 4px', fontFamily: 'inherit', lineHeight: 1, flexShrink: 0 },

  trackOuter:  { flex: 1, height: '20px', display: 'flex', alignItems: 'center', cursor: 'pointer' },
  trackInner:  { width: '100%', height: '4px', background: 'rgba(255,255,255,0.12)', borderRadius: '2px', position: 'relative', overflow: 'visible' },
  trackFill:   { height: '100%', background: 'rgba(255,255,255,0.5)', borderRadius: '2px', pointerEvents: 'none' },
  trackThumb:  { position: 'absolute', top: '50%', transform: 'translate(-50%,-50%)', width: '12px', height: '12px', background: '#fff', borderRadius: '50%', pointerEvents: 'none', boxShadow: '0 0 4px rgba(0,0,0,0.4)' },

  playerTime:  { fontSize: '11px', color: 'rgba(255,255,255,0.35)', flexShrink: 0, fontVariantNumeric: 'tabular-nums' },

  loadingOverlay: { position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' },
  loadingText:    { fontSize: '13px', color: 'rgba(255,255,255,0.4)', fontFamily: 'Inter, sans-serif', letterSpacing: '0.03em' },

  toggle:         { display: 'flex', borderRadius: '6px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.12)', flexShrink: 0 },
  toggleBtn:      { background: 'transparent', border: 'none', color: 'rgba(255,255,255,0.35)', cursor: 'pointer', fontFamily: 'inherit', fontSize: '12px', padding: '5px 12px', transition: 'background 0.15s, color 0.15s' },
  toggleBtnActive:{ background: 'rgba(123,143,255,0.18)', color: '#7B8FFF' },
};
