// LOCAL DEV SCAFFOLDING — POC for input-output visualization comparison.

import { useRef, useState, useCallback, useEffect } from 'react';
import Meyda from 'meyda';
import { PitchDetector } from 'pitchy';
import {
  type FrameFeatures,
  normalizeFeatureArr,
  medianPitch,
  mapPitch,
  getGalaxyColor,
  drawFeatureNote,
} from '../utils/visDrawHelpers';

// ─── Types ────────────────────────────────────────────────────────────────────

interface SimilarityMatch {
  inputStart: number; inputEnd: number;
  outputStart: number; outputEnd: number;
  label: string; rgb: [number, number, number];
}

type Phase = 'idle' | 'input' | 'transitioning' | 'output';

// ─── [POC] Hard-coded similarity — replace with backend analysis in production ──
const SIMILARITY_MATCHES: SimilarityMatch[] = [
  { inputStart: 2.0,  inputEnd: 4.0,  outputStart: 0.0,  outputEnd: 1.0,  label: 'Motif A', rgb: [255, 140, 80] },
  { inputStart: 2.0,  inputEnd: 4.0,  outputStart: 4.0,  outputEnd: 5.0,  label: 'Motif A', rgb: [255, 140, 80] },
  { inputStart: 7.0,  inputEnd: 9.0,  outputStart: 12.0, outputEnd: 16.0, label: 'Motif B', rgb: [80, 220, 170] },
];

const TRANSITION_MS      = 2800;
const OUTPUT_READY_DELAY = 10;

// ─── Audio analysis ───────────────────────────────────────────────────────────

function avgOf(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

async function analyzeFile(
  url: string,
  onDone: (features: FrameFeatures[], dur: number) => void,
  isCancelled: () => boolean
) {
  try {
    const buf = await (await fetch(url)).arrayBuffer();
    if (isCancelled()) return;
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const full = await audioCtx.decodeAudioData(buf);
    void audioCtx.close();
    if (isCancelled()) return;

    const sr = full.sampleRate;
    const frameSize = 2048;
    const hopSize = Math.max(512, Math.floor(sr / 20));
    Meyda.sampleRate = sr; Meyda.bufferSize = frameSize;
    const det = PitchDetector.forFloat32Array(frameSize);
    const ch = full.getChannelData(0);

    const rawRms: number[] = [], rawC: number[] = [],
          rawP: number[] = [], confs: number[] = [], times: number[] = [];

    for (let i = 0; i < ch.length - frameSize; i += hopSize) {
      const frame = ch.slice(i, i + frameSize);
      const f = Meyda.extract(['spectralCentroid', 'rms'], frame);
      if (!f) continue;
      times.push(i / sr); rawRms.push(f.rms || 0); rawC.push(f.spectralCentroid || 0);
      const [freq, cl] = det.findPitch(frame, sr);
      rawP.push(freq && cl > 0 ? 69 + 12 * Math.log2(freq / 440) : 0); confs.push(cl);
      if (rawRms.length % 50 === 0) await new Promise(r => setTimeout(r, 0));
      if (isCancelled()) return;
    }

    const nRms = normalizeFeatureArr(rawRms), nC = normalizeFeatureArr(rawC);
    const feats: FrameFeatures[] = [];
    let si = -1;
    for (let i = 1; i < rawP.length; i++) {
      const p = rawP[i];
      if (p > 0 && si === -1) si = i;
      const end = si !== -1 && (p === 0 || Math.abs(p - rawP[si]) > 0.8 || i === rawP.length - 1);
      if (end) {
        feats.push({
          time: times[si], duration: times[i] - times[si],
          pitch: medianPitch(rawP.slice(si, i + 1)), pitchConf: confs[si],
          rms: avgOf(nRms.slice(si, i + 1)), centroid: avgOf(nC.slice(si, i + 1)),
        });
        si = p > 0 ? i : -1;
      }
    }
    // Energy fallback for polyphonic content
    if (!feats.length && times.length) {
      const fd = hopSize / sr;
      for (let i = 0; i < times.length; i += 8) {
        const r = nRms[i] ?? 0, c = nC[i] ?? 0, p = rawP[i] ?? 0;
        feats.push({
          time: times[i], duration: Math.max(fd * 8, 0.06),
          pitch: p > 0 ? p : 48 + c * 24, pitchConf: confs[i] ?? 0, rms: r, centroid: c,
        });
      }
    }
    if (!isCancelled()) onDone(feats, full.duration);
  } catch (e) { console.error('[ComparisonPage] analysis:', e); }
}

// ─── Drawing helpers ──────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
function easeIn3(t: number) { return t * t * t; }
function easeOut3(t: number) { return 1 - Math.pow(1 - t, 3); }

// Compute the live orbital position of a feature event — used for puff anchoring.
// Uses the same radius formula as visualizeNote so puff positions track the spiral correctly.
function orbitalPos(
  startTime: number, audioTime: number, pitch: number,
  orbitDur: number, baseR: number, cx: number, cy: number
) {
  const pn = mapPitch(pitch);
  const r = baseR + (pn - 0.5) * baseR * 2.5; // same as production visualizeNote
  const oi = Math.floor(startTime / orbitDur);
  const op = (startTime - oi * orbitDur) / orbitDur;
  const angle = op * Math.PI * 2 + oi * 0.3 + audioTime * 0.1;
  return { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r, angle, r };
}

// Spanning glow — one soft blob per output onset in the match window.
// Blobs overlap via 'screen' blend, creating a luminous smear across the snippet.
// Opacity per blob scales by 1/√n so the cloud stays readable regardless of density.
function drawSpanningPuff(
  ctx: CanvasRenderingContext2D,
  positions: { x: number; y: number }[],
  anchor: { x: number; y: number },
  match: SimilarityMatch,
  now: number,
  selected: boolean
) {
  const [r, g, b] = match.rgb;
  const pulse = 1 + Math.sin(now * 0.002) * 0.12;
  const n = Math.max(1, positions.length);
  const baseStrength = selected ? 0.45 : 0.28;
  const strength = baseStrength / Math.sqrt(n);
  const radius = (selected ? 30 : 22) * pulse;

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

  ctx.globalAlpha = selected ? 0.95 : 0.65;
  ctx.fillStyle = `rgb(${r},${g},${b})`;
  ctx.font = `${selected ? 12 : 11}px Inter,sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(match.label, anchor.x, anchor.y + radius + 4);
  ctx.globalAlpha = 1; ctx.textBaseline = 'alphabetic';
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ComparisonPage() {
  const inputAudioRef  = useRef<HTMLAudioElement | null>(null);
  const outputAudioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef      = useRef<HTMLCanvasElement | null>(null);

  const [inputUrl,       setInputUrl]       = useState<string | null>(null);
  const [outputUrl,      setOutputUrl]      = useState<string | null>(null);
  const [inputFileName,  setInputFileName]  = useState('');
  const [outputFileName, setOutputFileName] = useState('');

  const [phase,          setPhase]          = useState<Phase>('idle');
  const [countdown,      setCountdown]      = useState(OUTPUT_READY_DELAY);
  const [selectedMatch,  setSelectedMatch]  = useState<SimilarityMatch | null>(null);
  const [playingSnippet, setPlayingSnippet] = useState<'input' | 'output' | null>(null);

  // Output audio player state
  const [outTime,    setOutTime]    = useState(0);
  const [outDur,     setOutDur]     = useState(0);
  const [outPlaying, setOutPlaying] = useState(false);

  // RAF-accessible refs
  const phaseRef           = useRef<Phase>('idle');
  const transitionStartRef = useRef(0);
  const inputFeaturesRef   = useRef<FrameFeatures[]>([]);
  const outputFeaturesRef  = useRef<FrameFeatures[]>([]);
  const inputDurRef        = useRef(0);
  const outputDurRef       = useRef(0);
  const puffPositionsRef   = useRef<Array<{ x: number; y: number; match: SimilarityMatch }>>([]);
  const selectedMatchRef   = useRef<SimilarityMatch | null>(null);
  const playingSnippetRef  = useRef<'input' | 'output' | null>(null);
  const snippetTimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);

  const snapshotTakenRef = useRef(false);
  const lastPhaseRef     = useRef<Phase>('idle');

  useEffect(() => { phaseRef.current = phase; }, [phase]);
  useEffect(() => { selectedMatchRef.current = selectedMatch; }, [selectedMatch]);
  useEffect(() => { playingSnippetRef.current = playingSnippet; }, [playingSnippet]);

  // ── Output audio player listeners ──────────────────────────────────────────
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
    el.addEventListener('play',  onPlay);
    el.addEventListener('pause', onPause);
    el.addEventListener('ended', onEnded);
    return () => {
      el.removeEventListener('timeupdate',     onTime);
      el.removeEventListener('durationchange', onDur);
      el.removeEventListener('loadedmetadata', onDur);
      el.removeEventListener('play',  onPlay);
      el.removeEventListener('pause', onPause);
      el.removeEventListener('ended', onEnded);
    };
  }, []);

  // ── File handlers ──────────────────────────────────────────────────────────
  const handleInputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setInputFileName(file.name);
    setInputUrl(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
    inputFeaturesRef.current = []; inputDurRef.current = 0;
    let cancelled = false;
    new Audio(url).addEventListener('loadedmetadata', function() {
      analyzeFile(url, (feats, dur) => {
        inputFeaturesRef.current = feats; inputDurRef.current = dur;
      }, () => cancelled);
    }, { once: true });
    return () => { cancelled = true; };
  }, []);

  const handleOutputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setOutputFileName(file.name);
    setOutputUrl(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
    outputFeaturesRef.current = []; outputDurRef.current = 0;
    let cancelled = false;
    new Audio(url).addEventListener('loadedmetadata', function() {
      analyzeFile(url, (feats, dur) => {
        outputFeaturesRef.current = feats; outputDurRef.current = dur;
      }, () => cancelled);
    }, { once: true });
    return () => { cancelled = true; };
  }, []);

  // ── Phase control ──────────────────────────────────────────────────────────
  const handleStart = useCallback(() => {
    if (!inputUrl) return;
    snapshotTakenRef.current = false;
    setPhase('input');
    setCountdown(OUTPUT_READY_DELAY);
    setSelectedMatch(null); setPlayingSnippet(null);
    const a = inputAudioRef.current;
    if (a) { a.currentTime = 0; void a.play(); }
  }, [inputUrl]);

  const handleReset = useCallback(() => {
    snapshotTakenRef.current = false;
    setPhase('idle'); setSelectedMatch(null); setPlayingSnippet(null);
    setCountdown(OUTPUT_READY_DELAY);
    inputAudioRef.current?.pause();
    outputAudioRef.current?.pause();
  }, []);

  // Countdown → transitioning → output
  useEffect(() => {
    if (phase !== 'input') return;
    const iv = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(iv);
          setPhase('transitioning');
          transitionStartRef.current = performance.now();
          setTimeout(() => {
            inputAudioRef.current?.pause();
            setPhase('output');
            const out = outputAudioRef.current;
            if (out) { out.currentTime = 0; void out.play(); }
          }, TRANSITION_MS);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(iv);
  }, [phase]);

  // ── Canvas draw loop ───────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width  = Math.round(rect.width  * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(dpr, dpr);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    type MatchState = { anchorIdx: number; windowIndices: number[]; spawnTime: number };
    const matchStates: (MatchState | null)[] = SIMILARITY_MATCHES.map(() => null);

    let rafId: number;

    const draw = () => {
      const p = phaseRef.current;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width, h = rect.height, cx = w / 2, cy = h / 2;
      const baseR = Math.min(w, h) * 0.2;
      const now = performance.now();

      if (p !== lastPhaseRef.current) {
        if (p !== 'transitioning') {
          snapshotTakenRef.current = false;
        }
        if (p !== 'output') {
          matchStates.fill(null);
          puffPositionsRef.current = [];
        }
        lastPhaseRef.current = p;
      }

      ctx.clearRect(0, 0, w, h); ctx.globalAlpha = 1;

      const inF  = inputFeaturesRef.current;
      const outF = outputFeaturesRef.current.length > 0 ? outputFeaturesRef.current : inF;
      const inDur  = inputDurRef.current  || 30;
      const outDur = outputDurRef.current || inDur;
      const inOrb  = Math.min(inDur,  10);
      const outOrb = Math.min(outDur, 10);
      const inT  = inputAudioRef.current?.currentTime  ?? 0;
      const outT = outputAudioRef.current?.currentTime ?? 0;
      const sel  = selectedMatchRef.current;

      // ── input: normal spiral ─────────────────────────────────────────────
      if (p === 'input') {
        inF.forEach(evt => drawFeatureNote(ctx, evt, inT, cx, cy, baseR, inOrb));
      }

      // ── transitioning: whole spiral scales toward center ─────────────────
      else if (p === 'transitioning') {
        const progress = Math.min((now - transitionStartRef.current) / TRANSITION_MS, 1);
        const eased = easeIn3(progress);
        const effectiveBaseR = baseR * (1 - eased);
        inF.forEach(evt =>
          drawFeatureNote(ctx, evt, inT, cx, cy, effectiveBaseR, inOrb, lerp(1, 0, eased))
        );
      }

      // ── output: ghost cloud + output spiral + similarity puffs ───────────
      else if (p === 'output') {
        // Ghost cloud — semi-transparent input particles drifting near center
        inF.forEach((evt, i) => {
          const a  = (i / Math.max(inF.length, 1)) * Math.PI * 2;
          const dr = Math.sin(now * 0.0003 + i * 0.7) * 0.35;
          const rr = 10 + (i % 9) * 3.5 + Math.sin(now * 0.0005 + i) * 4;
          ctx.globalAlpha = 0.04 + evt.rms * 0.07;
          ctx.fillStyle = getGalaxyColor(evt.rms, evt.centroid);
          ctx.beginPath(); ctx.arc(cx + Math.cos(a + dr) * rr, cy + Math.sin(a + dr) * rr,
            2 + evt.rms * 2, 0, Math.PI * 2); ctx.fill();
        });
        ctx.globalAlpha = 1;

        // Output spiral
        outF.forEach(evt => drawFeatureNote(ctx, evt, outT, cx, cy, baseR, outOrb));

        // Per-match: fly-in + settled puff, driven by outT so scrubbing resets them
        const newPuffs: typeof puffPositionsRef.current = [];
        SIMILARITY_MATCHES.forEach((match, idx) => {
          if (outT < match.outputStart) {
            // Scrubbed back — clear so fly-in replays next time outT crosses threshold
            matchStates[idx] = null;
            return;
          }

          if (matchStates[idx] === null) {
            // First frame past trigger — compute anchor/window and record wall-clock start
            const windowIndices: number[] = [];
            outF.forEach((f, fi) => {
              if (f.time >= match.outputStart && f.time <= match.outputEnd) windowIndices.push(fi);
            });
            let anchorIdx = -1, bestD = Infinity;
            outF.forEach((f, fi) => {
              const d = Math.abs(f.time - match.outputStart);
              if (d < bestD) { bestD = d; anchorIdx = fi; }
            });
            if (!windowIndices.length && anchorIdx >= 0) windowIndices.push(anchorIdx);
            matchStates[idx] = { anchorIdx, windowIndices, spawnTime: now };
          }

          const state = matchStates[idx]!;
          const progress = Math.min((now - state.spawnTime) / 1300, 1);
          const anchorEvt = outF[state.anchorIdx];
          const anchorPos = anchorEvt
            ? orbitalPos(anchorEvt.time, outT, anchorEvt.pitch, outOrb, baseR, cx, cy)
            : { x: cx + 80, y: cy };

          if (progress < 1) {
            const [r, g, b] = match.rgb;
            for (let t = 0; t < 10; t++) {
              const tp = Math.max(0, progress - t * 0.018);
              const te = easeOut3(tp);
              ctx.globalAlpha = (0.85 - t * 0.08) * (1 - progress * 0.15);
              ctx.fillStyle = `rgba(${r},${g},${b},1)`;
              ctx.beginPath(); ctx.arc(
                lerp(cx, anchorPos.x, te), lerp(cy, anchorPos.y, te),
                Math.max(10 - t, 1) * 0.8, 0, Math.PI * 2
              ); ctx.fill();
            }
            ctx.globalAlpha = 1;
          } else {
            const windowPositions = state.windowIndices.map((i: number) => {
              const f = outF[i];
              return f ? orbitalPos(f.time, outT, f.pitch, outOrb, baseR, cx, cy) : anchorPos;
            });
            const isSel = sel?.label === match.label;
            drawSpanningPuff(ctx, windowPositions, anchorPos, match, now, isSel);
            for (const pos of windowPositions) newPuffs.push({ x: pos.x, y: pos.y, match });
          }
        });
        puffPositionsRef.current = newPuffs;
      }

      ctx.globalAlpha = 1;
      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(rafId); ro.disconnect(); };
  }, []);

  // ── Canvas interaction ─────────────────────────────────────────────────────
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phaseRef.current !== 'output') return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    for (const puff of puffPositionsRef.current) {
      if (Math.hypot(mx - puff.x, my - puff.y) < 42) {
        setSelectedMatch(puff.match);
        setPlayingSnippet(null);
        outputAudioRef.current?.pause();
        if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
        break;
      }
    }
  }, []);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phaseRef.current !== 'output' || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const over = puffPositionsRef.current.some(p => Math.hypot(mx - p.x, my - p.y) < 42);
    canvasRef.current.style.cursor = over ? 'pointer' : 'default';
  }, []);

  // ── Snippet playback — toggles pause if already playing that type ──────────
  const playSnippet = useCallback((type: 'input' | 'output') => {
    if (playingSnippetRef.current === type) {
      if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
      (type === 'input' ? inputAudioRef : outputAudioRef).current?.pause();
      setPlayingSnippet(null);
      return;
    }

    if (!selectedMatchRef.current) return;
    if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
    inputAudioRef.current?.pause();
    outputAudioRef.current?.pause();

    const match = selectedMatchRef.current;
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
    if (phaseRef.current === 'output') void outputAudioRef.current?.play();
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
    el.currentTime = 0; void el.play();
    setSelectedMatch(null); setPlayingSnippet(null);
  }, []);

  const seekPointer = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
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
  const effectiveOutputSrc = outputUrl ?? inputUrl ?? undefined;
  const pct = outDur > 0 ? (outTime / outDur) * 100 : 0;

  return (
    <div style={s.page}>
      {/* Hidden audio elements */}
      <audio ref={inputAudioRef}  src={inputUrl ?? undefined}  style={{ display: 'none' }} />
      <audio ref={outputAudioRef} src={effectiveOutputSrc}     style={{ display: 'none' }} />

      {/* Upload row */}
      <div style={s.row}>
        <UploadZone id="cmp-in"  label="Input Audio"  fileName={inputFileName}  onFile={handleInputFile} />
        <UploadZone id="cmp-out" label="Output Audio" fileName={outputFileName} onFile={handleOutputFile}
          hint="Optional — uses input as demo fallback" />
      </div>

      {/* Controls */}
      <div style={s.row}>
        {phase === 'idle' ? (
          <button style={{ ...s.btn, opacity: inputUrl ? 1 : 0.4 }} disabled={!inputUrl} onClick={handleStart}>
            Start
          </button>
        ) : (
          <button style={{ ...s.btn, ...s.btnGhost }} onClick={handleReset}>Reset</button>
        )}
        <span style={s.statusText}>
          {phase === 'input'         && <>Output ready in <strong style={{ color: '#fff' }}>{countdown}s</strong></>}
          {phase === 'transitioning' && 'Processing output…'}
          {phase === 'output'        && 'Click a glowing puff to inspect the source'}
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
        {selectedMatch && (
          <SnippetPanel
            match={selectedMatch}
            playing={playingSnippet}
            onPlay={playSnippet}
            onClose={closePanel}
          />
        )}
      </div>

      {/* Output audio player bar — shown in output phase when panel is closed */}
      {phase === 'output' && !selectedMatch && (
        <div style={s.playerBar}>
          <button style={s.playerBtn} onClick={toggleOutput} title={outPlaying ? 'Pause' : 'Play'}>
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
          <button style={s.playerBtn} onClick={replayOutput} title="Replay from start">↺</button>
        </div>
      )}
    </div>
  );
}

// ─── Snippet panel ─────────────────────────────────────────────────────────────

function SnippetPanel({ match, playing, onPlay, onClose }: {
  match: SimilarityMatch;
  playing: 'input' | 'output' | null;
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
        <button style={s.panelClose} onClick={onClose} title="Close">✕</button>
      </div>
      <p style={s.panelSub}>Play each snippet to hear the similarity</p>
      <div style={s.snippetRow}>
        <SnippetButton
          label="Input" start={match.inputStart} end={match.inputEnd}
          isActive={playing === 'input'} accent={accent} accentDim={accentDim}
          onClick={() => onPlay('input')}
        />
        <SnippetButton
          label="Output" start={match.outputStart} end={match.outputEnd}
          isActive={playing === 'output'} accent={accent} accentDim={accentDim}
          onClick={() => onPlay('output')}
        />
      </div>
      <button style={s.resumeBtn} onClick={onClose}>Resume output playback</button>
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
        background: isActive ? accentDim : 'rgba(255,255,255,0.04)',
        border: `1px solid ${isActive ? accent : 'rgba(255,255,255,0.09)'}`,
        color: isActive ? accent : 'rgba(255,255,255,0.72)',
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
    <div style={s.drop}
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

  panel:       { position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: '300px', background: 'rgba(14,14,20,0.97)', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.1)', padding: '22px 22px 18px', display: 'flex', flexDirection: 'column', gap: '14px', backdropFilter: 'blur(20px)', boxShadow: '0 28px 70px rgba(0,0,0,0.75)' },
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
};
