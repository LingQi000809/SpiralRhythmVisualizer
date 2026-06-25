// Scaffold for input vs. output audio similarity detection.
// Visualization matches production VisualizationView.tsx logic exactly —
// single orbital ring, same analyzeAudioUrl / drawFeatureNote pipeline.
// Similarity detection calls /similarity on the backend (SSE streaming).

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
import { streamSimilarity, type BackendSimilarityMatch } from '../utils/api';

const DEFAULT_INPUT_URL  = '/data/piano1.wav';
const DEFAULT_OUTPUT_URL = '/data/piano2.wav';

// ─── Similarity types + colours ───────────────────────────────────────────────

interface SimilarityMatch {
  inputStart:  number;
  inputEnd:    number;
  outputStart: number;
  outputEnd:   number;
  label: string;
  rgb:   [number, number, number];
  // Visualization data
  inNoteTimes:    number[];
  inNotePitches:  number[];
  outNoteTimes:   number[];
  outNotePitches: number[];
  inMatchSlice:   [number, number];
  outMatchSlice:  [number, number];
  dtwPath:        [number, number][];
  similarity:     number;
}

const MATCH_COLORS: [number, number, number][] = [
  [224, 123,  57],  // orange
  [ 46, 204, 113],  // green
  [155,  89, 182],  // purple
  [231,  76,  60],  // red
  [ 26, 188, 156],  // teal
  [243, 156,  18],  // amber
];

// ─── Puff draw helpers (mirrors StemVisualizationView) ────────────────────────

const PUFF_SPAWN_MS     = 1300;
const PUFF_FADE_DELAY_S = 2.0;
const PUFF_FADE_DUR_S   = 1.5;
const PUFF_HIT_R        = 40;

function puffEaseOut3(t: number) { return 1 - Math.pow(1 - t, 3); }
function puffLerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function drawMatchPuff(
  ctx:       CanvasRenderingContext2D,
  positions: { x: number; y: number }[],
  anchor:    { x: number; y: number },
  rgb:       [number, number, number],
  label:     string,
  now:       number,
  fadeAlpha: number,
) {
  const [r, g, b] = rgb;
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
  ctx.fillText(label, anchor.x, anchor.y + radius + 4);
  ctx.globalAlpha = 1; ctx.textBaseline = 'alphabetic';
}

// ─────────────────────────────────────────────────────────────────────────────

function toFrontendMatch(m: BackendSimilarityMatch, i: number): SimilarityMatch {
  return {
    inputStart:  m.input_start,
    inputEnd:    m.input_end,
    outputStart: m.output_start,
    outputEnd:   m.output_end,
    label: `${Math.round(m.similarity * 100)}% match`,
    rgb:   MATCH_COLORS[i % MATCH_COLORS.length],
    inNoteTimes:    m.in_note_times,
    inNotePitches:  m.in_note_pitches,
    outNoteTimes:   m.out_note_times,
    outNotePitches: m.out_note_pitches,
    inMatchSlice:   m.in_match_slice,
    outMatchSlice:  m.out_match_slice,
    dtwPath:        m.dtw_path,
    similarity:     m.similarity,
  };
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
  const [similarityMsg,  setSimilarityMsg]  = useState('');
  const [outPlaying,     setOutPlaying]     = useState(false);
  const [outTime,        setOutTime]        = useState(0);
  const [outDur,         setOutDur]         = useState(0);
  const [selectedMatch,  setSelectedMatch]  = useState<SimilarityMatch | null>(null);

  // Uploaded File objects — needed for multipart upload to backend
  const inputFileRef  = useRef<File | null>(null);
  const outputFileRef = useRef<File | null>(null);

  // RAF-accessible refs — mutated directly, never trigger re-renders
  const inputFeaturesRef  = useRef<FrameFeatures[]>([]);
  const trackRef  = useRef<HTMLDivElement | null>(null);
  const scrubRef  = useRef(false);
  const outputFeaturesRef = useRef<FrameFeatures[]>([]);
  const outputDurRef      = useRef(0);
  const pitchMedianRef    = useRef(0.5);
  const matchesRef        = useRef<SimilarityMatch[]>([]);
  const matchStatesRef    = useRef<({ anchorFeatIdx: number; windowFeatIndices: number[]; spawnTime: number } | null)[]>([]);
  const nebulaRef         = useRef<NebulaPuff[]>([]);
  const puffHitRef        = useRef<{ x: number; y: number; matchIdx: number }[]>([]);

  // Keep selected match accessible in RAF without closure staleness
  const selectedMatchRef = useRef<SimilarityMatch | null>(null);
  useEffect(() => { selectedMatchRef.current = selectedMatch; }, [selectedMatch]);

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
  const handleRunSimilarity = useCallback(async () => {
    setSimilarityMsg('Uploading audio…');
    try {
      const getBlob = async (fileRef: React.RefObject<File | null>, url: string): Promise<Blob> => {
        if (fileRef.current) return fileRef.current;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to fetch ${url}`);
        return res.blob();
      };

      const [inBlob, outBlob] = await Promise.all([
        getBlob(inputFileRef,  inputUrl),
        getBlob(outputFileRef, outputUrl),
      ]);

      const matches = await streamSimilarity(inBlob, outBlob, (msg) => setSimilarityMsg(msg));
      matchesRef.current     = matches.map(toFrontendMatch);
      matchStatesRef.current = matches.map(() => null);
      setSimilarityMsg(`Done — ${matches.length} match(es) found`);

      const el = outputAudioRef.current;
      if (el) { el.currentTime = 0; void el.play(); }
    } catch (err) {
      setSimilarityMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [inputUrl, outputUrl]);

  // ── Canvas puff click / hover ─────────────────────────────────────────────
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedMatchRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    for (const hit of puffHitRef.current) {
      const dx = mx - hit.x, dy = my - hit.y;
      if (dx * dx + dy * dy < PUFF_HIT_R * PUFF_HIT_R) {
        const match = matchesRef.current[hit.matchIdx];
        if (match) {
          setSelectedMatch(match);
          outputAudioRef.current?.pause();
        }
        return;
      }
    }
  }, []);

  const handleCanvasPointerMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedMatchRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = puffHitRef.current.some(h => {
      const dx = mx - h.x, dy = my - h.y;
      return dx * dx + dy * dy < PUFF_HIT_R * PUFF_HIT_R;
    });
    canvas.style.cursor = hit ? 'pointer' : 'default';
  }, []);

  const closePanel = useCallback(() => {
    setSelectedMatch(null);
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

      // ── Similarity match puffs ──────────────────────────────────────────────
      const featOrbitalPos = (feat: { time: number; pitch: number }, orb: number) => {
        const oi  = Math.floor(feat.time / orbitDur);
        const op  = (feat.time - oi * orbitDur) / orbitDur;
        const ang = op * Math.PI * 2 + oi * 0.3 + t * 0.1;
        return { x: cx + Math.cos(ang) * orb, y: cy + Math.sin(ang) * orb };
      };

      puffHitRef.current = [];  // reset hit-test list each frame

      const states  = matchStatesRef.current;
      const matches = matchesRef.current;
      for (let mi = 0; mi < matches.length; mi++) {
        const match = matches[mi];
        if (t < match.outputStart) { states[mi] = null; continue; }

        if (!states[mi]) {
          const windowIndices: number[] = [];
          for (let fi = 0; fi < feats.length; fi++) {
            if (feats[fi].time >= match.outputStart && feats[fi].time <= match.outputEnd)
              windowIndices.push(fi);
          }
          let anchorIdx = 0, bestD = Infinity;
          for (let fi = 0; fi < feats.length; fi++) {
            const d = Math.abs(feats[fi].time - match.outputStart);
            if (d < bestD) { bestD = d; anchorIdx = fi; }
          }
          if (!windowIndices.length) windowIndices.push(anchorIdx);
          states[mi] = { anchorFeatIdx: anchorIdx, windowFeatIndices: windowIndices, spawnTime: now };
        }

        const state    = states[mi]!;
        const progress = Math.min((now - state.spawnTime) / PUFF_SPAWN_MS, 1);
        const pn       = mapPitch(feats[state.anchorFeatIdx]?.pitch ?? 0);
        const orbR     = ringR + (pn - median) * bandHalf * 2;
        const anchor   = featOrbitalPos(feats[state.anchorFeatIdx] ?? { time: 0, pitch: 0 }, orbR);

        const fadeAlpha = t > match.outputEnd + PUFF_FADE_DELAY_S
          ? Math.max(0, 1 - (t - match.outputEnd - PUFF_FADE_DELAY_S) / PUFF_FADE_DUR_S)
          : 1;
        if (fadeAlpha <= 0) continue;

        const [r, g, b] = match.rgb;
        if (progress < 1) {
          for (let trail = 0; trail < 10; trail++) {
            const tp = Math.max(0, progress - trail * 0.018);
            const te = puffEaseOut3(tp);
            ctx.globalAlpha = (0.85 - trail * 0.08) * (1 - progress * 0.15) * fadeAlpha;
            ctx.fillStyle   = `rgba(${r},${g},${b},1)`;
            ctx.beginPath();
            ctx.arc(puffLerp(cx, anchor.x, te), puffLerp(cy, anchor.y, te),
              Math.max(10 - trail, 1) * 0.8, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.globalAlpha = 1;
        } else {
          const windowPositions = state.windowFeatIndices.map(fi => {
            const pni  = mapPitch(feats[fi]?.pitch ?? 0);
            const orbRi = ringR + (pni - median) * bandHalf * 2;
            return featOrbitalPos(feats[fi] ?? { time: 0, pitch: 0 }, orbRi);
          });
          drawMatchPuff(ctx, windowPositions, anchor, match.rgb, match.label, now, fadeAlpha);
          puffHitRef.current.push({ x: anchor.x, y: anchor.y, matchIdx: mi });
        }
      }

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
    inputFileRef.current = file;
    const url = URL.createObjectURL(file);
    setInputFileName(file.name);
    setInputUrl(prev => { if (prev.startsWith('blob:')) URL.revokeObjectURL(prev); return url; });
  }, []);

  const handleOutputFile = useCallback((file: File) => {
    outputFileRef.current = file;
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
        <canvas ref={nebulaCanvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%',
          filter: selectedMatch ? 'brightness(0.3)' : 'none', transition: 'filter 0.3s' }} />
        <canvas ref={canvasRef}
          style={{ position: 'relative', width: '100%', height: '100%',
            filter: selectedMatch ? 'brightness(0.3)' : 'none', transition: 'filter 0.3s' }}
          onClick={handleCanvasClick}
          onPointerMove={handleCanvasPointerMove}
        />
        {loadingMsg && <div style={s.overlay}>{loadingMsg}</div>}

        {/* Match panel — rendered inside canvasWrap so it sits above the dimmed canvas */}
        {selectedMatch && (
          <MatchPanel
            match={selectedMatch}
            inputUrl={inputUrl}
            outputUrl={outputUrl}
            onClose={closePanel}
          />
        )}
      </div>

      {/* Similarity progress log */}
      {similarityMsg && (
        <div style={s.simLog}>{similarityMsg}</div>
      )}

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

// ─── Match panel ──────────────────────────────────────────────────────────────

const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const midiToNote = (midi: number) =>
  NOTE_NAMES[Math.round(midi) % 12] + (Math.floor(Math.round(midi) / 12) - 1);

const SNIPPET_CTX_S = 1.0;  // extra seconds before/after match in snippet playback

// Canvas layout constants (CSS pixels) — shared between draw and click handler
const VIZ_PAD_X  = 32;
const VIZ_IN_Y0  = 22;
const VIZ_IN_H   = 130;
const VIZ_OUT_Y0 = 248;   // gap = 248-152 = 96px for DTW lines
const VIZ_OUT_H  = 130;
const VIZ_GAP_MID = (VIZ_IN_Y0 + VIZ_IN_H + VIZ_OUT_Y0) / 2;  // ≈ 211
const VIZ_CSS_H   = VIZ_OUT_Y0 + VIZ_OUT_H + 14;               // ≈ 392

function MatchPanel({ match, inputUrl, outputUrl, onClose }: {
  match:     SimilarityMatch;
  inputUrl:  string;
  outputUrl: string;
  onClose:   () => void;
}) {
  const vizRef       = useRef<HTMLCanvasElement>(null);
  const snippetRef   = useRef(new Audio());
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [playing, setPlaying] = useState<'input' | 'output' | null>(null);
  // No playTime state — audio.currentTime is read directly inside the RAF loop

  useEffect(() => {
    return () => {
      snippetRef.current.pause();
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
    };
  }, []);

  const seekAndPlay = useCallback((type: 'input' | 'output', absTime: number) => {
    const audio = snippetRef.current;
    if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
    audio.pause();
    const [matchStart, matchEnd, url] = type === 'input'
      ? [match.inputStart,  match.inputEnd,  inputUrl]
      : [match.outputStart, match.outputEnd, outputUrl];
    const clipStart = Math.max(0, matchStart - SNIPPET_CTX_S);
    const clipEnd   = matchEnd + SNIPPET_CTX_S;
    const seekTo    = Math.max(clipStart, Math.min(clipEnd, absTime));
    if (audio.src !== url) audio.src = url;
    audio.currentTime = seekTo;
    void audio.play();
    setPlaying(type);
    stopTimerRef.current = setTimeout(() => {
      audio.pause();
      setPlaying(null);
    }, (clipEnd - seekTo) * 1000);
  }, [match, inputUrl, outputUrl]);

  const toggleSnippet = (type: 'input' | 'output') => {
    if (playing === type) {
      snippetRef.current.pause();
      setPlaying(null);
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
      return;
    }
    const matchStart = type === 'input' ? match.inputStart : match.outputStart;
    seekAndPlay(type, Math.max(0, matchStart - SNIPPET_CTX_S));
  };

  // Click on viz canvas → seek in the clicked lane
  const handleVizClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = vizRef.current;
    if (!canvas) return;
    const rect  = canvas.getBoundingClientRect();
    const cssX  = e.clientX - rect.left;
    const cssY  = e.clientY - rect.top;
    const matchDur = match.inputEnd - match.inputStart;
    const allTimes = [...match.inNoteTimes, ...match.outNoteTimes];
    const tMin = Math.max(Math.min(...allTimes, -0.1), -SNIPPET_CTX_S - 0.05);
    const tMax = Math.min(Math.max(...allTimes,  0.1),  matchDur + SNIPPET_CTX_S + 0.05);
    const relTime  = tMin + ((cssX - VIZ_PAD_X) / (rect.width - VIZ_PAD_X * 2)) * (tMax - tMin);
    const type     = cssY < VIZ_GAP_MID ? 'input' : 'output';
    const refStart = type === 'input' ? match.inputStart : match.outputStart;
    seekAndPlay(type, refStart + relTime);
  }, [match, seekAndPlay]);

  // ── RAF-based draw: smooth 60fps cursor, static otherwise ──────────────────
  useEffect(() => {
    const canvas = vizRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let rafId: number | null = null;

    const drawFrame = () => {
      const audio    = snippetRef.current;
      const nowTime  = audio.currentTime;  // read live for smooth cursor

      const dpr  = window.devicePixelRatio || 1;
      const cssW = canvas.getBoundingClientRect().width || 540;
      const cssH = VIZ_CSS_H;
      if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
        canvas.width  = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      const W = cssW;

      const { inNoteTimes, inNotePitches, outNoteTimes, outNotePitches,
              inMatchSlice, outMatchSlice, dtwPath } = match;
      if (!inNoteTimes.length && !outNoteTimes.length) return;

      // Clip time axis to audible window (1s context on each side)
      const matchDur = match.inputEnd - match.inputStart;
      const allTimes = [...inNoteTimes, ...outNoteTimes];
      const tMin = Math.max(Math.min(...allTimes, -0.1), -SNIPPET_CTX_S - 0.05);
      const tMax = Math.min(Math.max(...allTimes,  0.1),  matchDur + SNIPPET_CTX_S + 0.05);

      const allPitches = [...inNotePitches, ...outNotePitches].filter(p => p > 0);
      const pMin = allPitches.length ? Math.min(...allPitches) - 2 : 48;
      const pMax = allPitches.length ? Math.max(...allPitches) + 2 : 84;

      const toX    = (t: number) => VIZ_PAD_X + ((t - tMin) / (tMax - tMin)) * (W - VIZ_PAD_X * 2);
      const toInY  = (p: number) => VIZ_IN_Y0  + VIZ_IN_H  - ((p - pMin) / (pMax - pMin)) * VIZ_IN_H;
      const toOutY = (p: number) => VIZ_OUT_Y0 + VIZ_OUT_H - ((p - pMin) / (pMax - pMin)) * VIZ_OUT_H;

      const [r, g, b] = match.rgb;

      // Closest note to playhead (search all notes, not just match window)
      const inRelT  = playing === 'input'  ? nowTime - match.inputStart  : null;
      const outRelT = playing === 'output' ? nowTime - match.outputStart : null;
      let activeIn = -1, activeOut = -1;
      if (inRelT  !== null) { let bd = Infinity; for (let k = 0; k < inNoteTimes.length;  k++) { const d = Math.abs(inNoteTimes[k]  - inRelT);  if (d < bd) { bd = d; activeIn  = k; } } }
      if (outRelT !== null) { let bd = Infinity; for (let k = 0; k < outNoteTimes.length; k++) { const d = Math.abs(outNoteTimes[k] - outRelT); if (d < bd) { bd = d; activeOut = k; } } }

      // Lane backgrounds
      ctx.fillStyle = 'rgba(255,255,255,0.03)';
      ctx.fillRect(0, VIZ_IN_Y0,  W, VIZ_IN_H);
      ctx.fillRect(0, VIZ_OUT_Y0, W, VIZ_OUT_H);

      // Match-window highlight bands
      const inBL  = inNoteTimes[inMatchSlice[0]]     !== undefined ? toX(inNoteTimes[inMatchSlice[0]])     : VIZ_PAD_X;
      const inBR  = inNoteTimes[inMatchSlice[1] - 1] !== undefined ? toX(inNoteTimes[inMatchSlice[1] - 1]) : W - VIZ_PAD_X;
      const outBL = outNoteTimes[outMatchSlice[0]]     !== undefined ? toX(outNoteTimes[outMatchSlice[0]])     : VIZ_PAD_X;
      const outBR = outNoteTimes[outMatchSlice[1] - 1] !== undefined ? toX(outNoteTimes[outMatchSlice[1] - 1]) : W - VIZ_PAD_X;
      ctx.fillStyle = `rgba(${r},${g},${b},0.09)`;
      ctx.fillRect(inBL,  VIZ_IN_Y0,  inBR  - inBL,  VIZ_IN_H);
      ctx.fillRect(outBL, VIZ_OUT_Y0, outBR - outBL, VIZ_OUT_H);

      // Lane labels — above each lane, crisp
      ctx.font = 'bold 10px Inter, system-ui, sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.fillText('INPUT',  VIZ_PAD_X, VIZ_IN_Y0  - 5);
      ctx.fillText('OUTPUT', VIZ_PAD_X, VIZ_OUT_Y0 - 5);

      // Match-start dashed tick
      const tickX = toX(0);
      ctx.strokeStyle = `rgba(${r},${g},${b},0.4)`;
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(tickX, VIZ_IN_Y0); ctx.lineTo(tickX, VIZ_OUT_Y0 + VIZ_OUT_H); ctx.stroke();
      ctx.setLineDash([]);

      // Playhead (solid, 60fps smooth)
      if (playing !== null) {
        const relT  = playing === 'input' ? nowTime - match.inputStart : nowTime - match.outputStart;
        const hx    = toX(relT);
        const lY0   = playing === 'input' ? VIZ_IN_Y0  : VIZ_OUT_Y0;
        const lH    = playing === 'input' ? VIZ_IN_H   : VIZ_OUT_H;
        ctx.strokeStyle = `rgba(${r},${g},${b},0.85)`;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(hx, lY0); ctx.lineTo(hx, lY0 + lH); ctx.stroke();
      }

      // DTW alignment lines — note[pi]→note[pj] + final trailing notes
      if (dtwPath.length > 0) {
        ctx.strokeStyle = `rgba(${r},${g},${b},0.38)`; ctx.lineWidth = 1.5;
        for (const [pi, pj] of dtwPath) {
          const ik = inMatchSlice[0] + pi, ok = outMatchSlice[0] + pj;
          if (ik >= inNoteTimes.length || ok >= outNoteTimes.length) continue;
          ctx.beginPath();
          ctx.moveTo(toX(inNoteTimes[ik]),  toInY(inNotePitches[ik]));
          ctx.lineTo(toX(outNoteTimes[ok]), toOutY(outNotePitches[ok]));
          ctx.stroke();
        }
        // Connect final notes (interval path misses the trailing note of each sequence)
        const [lPi, lPj] = dtwPath[dtwPath.length - 1];
        const fIk = inMatchSlice[0] + lPi + 1, fOk = outMatchSlice[0] + lPj + 1;
        if (fIk < inNoteTimes.length && fOk < outNoteTimes.length) {
          ctx.beginPath();
          ctx.moveTo(toX(inNoteTimes[fIk]),  toInY(inNotePitches[fIk]));
          ctx.lineTo(toX(outNoteTimes[fOk]), toOutY(outNotePitches[fOk]));
          ctx.stroke();
        }
      }

      // Note dots + pitch labels — skip notes outside the clipped time window
      const drawNotes = (
        times: number[], pitches: number[], matchSlice: [number, number],
        toY: (p: number) => number, activeIdx: number, labelBelow: boolean,
      ) => {
        for (let k = 0; k < times.length; k++) {
          if (times[k] < tMin - 0.01 || times[k] > tMax + 0.01) continue;
          const inWin    = k >= matchSlice[0] && k < matchSlice[1];
          const isActive = k === activeIdx;
          const x = toX(times[k]), y = toY(pitches[k]);
          const dotR = isActive ? 6 : inWin ? 4 : 2.5;

          if (isActive) {
            ctx.globalAlpha = 0.35;
            const gl = ctx.createRadialGradient(x, y, 0, x, y, 18);
            gl.addColorStop(0,   `rgba(${r},${g},${b},1)`);
            gl.addColorStop(0.5, `rgba(${r},${g},${b},0.3)`);
            gl.addColorStop(1,   `rgba(${r},${g},${b},0)`);
            ctx.fillStyle = gl;
            ctx.beginPath(); ctx.arc(x, y, 18, 0, Math.PI * 2); ctx.fill();
          }

          ctx.globalAlpha = isActive ? 1 : inWin ? 0.92 : 0.28;
          ctx.fillStyle   = inWin ? `rgb(${r},${g},${b})` : 'rgba(255,255,255,0.8)';
          ctx.beginPath(); ctx.arc(x, y, dotR, 0, Math.PI * 2); ctx.fill();

          if (inWin && pitches[k] > 0) {
            ctx.globalAlpha = isActive ? 1 : 0.75;
            ctx.font        = `${isActive ? 'bold ' : ''}10px Inter, system-ui, sans-serif`;
            ctx.fillStyle   = isActive ? `rgb(${r},${g},${b})` : 'rgba(255,255,255,0.72)';
            ctx.textAlign   = 'center'; ctx.textBaseline = 'alphabetic';
            if (labelBelow) ctx.fillText(midiToNote(pitches[k]), x, y + dotR + 11);
            else            ctx.fillText(midiToNote(pitches[k]), x, y - dotR - 3);
          }
        }
      };

      drawNotes(inNoteTimes,  inNotePitches,  inMatchSlice,  toInY,  activeIn,  false);
      drawNotes(outNoteTimes, outNotePitches, outMatchSlice, toOutY, activeOut, true);
      ctx.globalAlpha = 1; ctx.textBaseline = 'alphabetic';
    };

    drawFrame();  // static draw on mount / match change

    if (playing !== null) {
      const loop = () => { drawFrame(); rafId = requestAnimationFrame(loop); };
      rafId = requestAnimationFrame(loop);
    }

    return () => { if (rafId !== null) cancelAnimationFrame(rafId); };
  }, [match, playing]);  // no playTime — read live from audio element

  const [r, g, b] = match.rgb;
  const fmtS = (s: number) => s.toFixed(1) + 's';
  const circ = 2 * Math.PI * 10;  // circumference for r=10

  return (
    <div style={ps.panel} onClick={e => e.stopPropagation()}>
      {/* Header */}
      <div style={ps.header}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: `rgb(${r},${g},${b})`,
          display: 'inline-block', flexShrink: 0, marginRight: 8 }} />
        <span style={ps.title}>{match.label}</span>
        {/* Circular similarity score */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 10 }}>
          <svg width="28" height="28" viewBox="0 0 28 28" style={{ flexShrink: 0 }}>
            <circle cx="14" cy="14" r="10" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="2.5" />
            <circle cx="14" cy="14" r="10" fill="none"
              stroke={`rgb(${r},${g},${b})`} strokeWidth="2.5"
              strokeDasharray={`${match.similarity * circ} ${circ}`}
              strokeLinecap="round"
              transform="rotate(-90 14 14)"
            />
          </svg>
          <span style={{ color: `rgb(${r},${g},${b})`, fontWeight: 700, fontSize: 17,
            letterSpacing: '-0.03em', lineHeight: 1 }}>
            {Math.round(match.similarity * 100)}%
          </span>
        </div>
        <button style={ps.closeBtn} onClick={onClose}>✕</button>
      </div>

      {/* Snippet players */}
      <div style={ps.players}>
        <div style={ps.playerRow}>
          <button style={{ ...ps.playBtn, borderColor: `rgba(${r},${g},${b},0.55)` }}
                  onClick={() => toggleSnippet('input')}>
            {playing === 'input' ? '⏸' : '▶'}&nbsp;Input
          </button>
          <span style={ps.timeRange}>
            {fmtS(Math.max(0, match.inputStart - SNIPPET_CTX_S))} – {fmtS(match.inputEnd + SNIPPET_CTX_S)}
          </span>
        </div>
        <div style={ps.playerRow}>
          <button style={{ ...ps.playBtn, borderColor: `rgba(${r},${g},${b},0.55)` }}
                  onClick={() => toggleSnippet('output')}>
            {playing === 'output' ? '⏸' : '▶'}&nbsp;Output
          </button>
          <span style={ps.timeRange}>
            {fmtS(Math.max(0, match.outputStart - SNIPPET_CTX_S))} – {fmtS(match.outputEnd + SNIPPET_CTX_S)}
          </span>
        </div>
      </div>

      {/* Pitch contour + DTW — click to seek */}
      <div style={ps.vizLabel}>Pitch contour · DTW alignment · click to seek</div>
      <canvas
        ref={vizRef}
        style={{ ...ps.vizCanvas, height: VIZ_CSS_H, cursor: 'crosshair' }}
        onClick={handleVizClick}
      />
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

  simLog: {
    fontSize: 11, color: 'rgba(255,255,255,0.5)',
    fontFamily: 'ui-monospace, monospace',
    padding: '4px 6px',
    background: 'rgba(255,255,255,0.04)',
    borderRadius: 4,
    flexShrink: 0,
    whiteSpace: 'pre',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
};

// ─── Panel styles ─────────────────────────────────────────────────────────────

const ps: Record<string, React.CSSProperties> = {
  panel: {
    position: 'absolute', top: '50%', left: '50%',
    transform: 'translate(-50%, -50%)',
    background: 'rgba(14,14,18,0.97)',
    border: '1px solid rgba(255,255,255,0.11)',
    borderRadius: 14, padding: '20px 24px 24px',
    width: 600, zIndex: 100,
    boxShadow: '0 24px 64px rgba(0,0,0,0.8)',
    backdropFilter: 'blur(16px)',
  },
  header: {
    display: 'flex', alignItems: 'center', marginBottom: 16,
  },
  title: {
    fontSize: 14, fontWeight: 600, color: '#fff', flex: 1,
  },
  score: {
    fontSize: 11, color: 'rgba(255,255,255,0.4)', marginRight: 12,
  },
  closeBtn: {
    background: 'none', border: 'none', color: 'rgba(255,255,255,0.45)',
    fontSize: 15, cursor: 'pointer', padding: '2px 4px', lineHeight: 1,
    flexShrink: 0,
  },
  players: {
    display: 'flex', flexDirection: 'column', gap: 9, marginBottom: 18,
  },
  playerRow: {
    display: 'flex', alignItems: 'center', gap: 10,
  },
  playBtn: {
    background: 'rgba(255,255,255,0.06)', color: '#fff',
    border: '1px solid', borderRadius: 6, padding: '5px 14px',
    fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
    flexShrink: 0, minWidth: 88,
  },
  timeRange: {
    fontSize: 11, color: 'rgba(255,255,255,0.38)',
    fontVariantNumeric: 'tabular-nums',
  },
  vizLabel: {
    fontSize: 9, fontWeight: 700, color: 'rgba(255,255,255,0.22)',
    textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 7,
  },
  vizCanvas: {
    width: '100%', height: 268, display: 'block',
    background: 'rgba(255,255,255,0.025)', borderRadius: 6,
  },
};
