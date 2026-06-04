// ============================================================
// Experiment replacement for Hearmi-Frontend's VisualizationView.tsx.
// Uses a 2D canvas spiral (matching VisualizationWaitingView aesthetics)
// with one concentric orbit ring per stem instead of Three.js particles.
//
// Migration to production:
//   1. Copy this file to Hearmi-Frontend/app/studio/components/
//   2. Change imports of processVisualization/toVisualizationCacheData
//      to use ../../utils/api (production URL already set there).
//   3. Redirect stemVizHelpers / visDrawHelpers / visMidiHelpers imports
//      to ../utils/.
//   4. The suck-in transition + audio routing lives in the parent
//      (OutputPanel in production); this component is view-only.
// ============================================================

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  type FrameFeatures,
  mapPitch,
  drawFeatureNote,
  analyzeAudioUrl,
} from '../utils/visDrawHelpers';
import type { NebulaPuff } from '../utils/visMidiHelpers';
import { drawNebulaLayer } from '../utils/visMidiHelpers';
import {
  processVisualization,
  toVisualizationCacheData,
  type VisualizationData,
  type VisualizationCacheData,
  type SongSection,
  type FlamingoChord,
} from '../utils/api';

// Deduplicate expensive visualization processing by source audio URL.
const visualizationProcessingByAudioUrl = new Map<string, Promise<VisualizationData>>();
const visualizationDataByAudioUrl = new Map<string, VisualizationData>();
const shouldCacheVisualizationData = (audioUrl: string): boolean => !audioUrl.startsWith('blob:');

const processVisualizationForAudioUrl = async (audioUrl: string): Promise<VisualizationData> => {
  if (shouldCacheVisualizationData(audioUrl)) {
    const cachedData = visualizationDataByAudioUrl.get(audioUrl);
    if (cachedData) return cachedData;
  }
  const existingProcess = visualizationProcessingByAudioUrl.get(audioUrl);
  if (existingProcess) return existingProcess;

  const processPromise = (async () => {
    const response = await fetch(audioUrl);
    if (!response.ok) throw new Error(`Failed to fetch audio: ${response.status}`);
    const blob = await response.blob();
    const file = new File([blob], 'audio.wav', { type: blob.type });
    const result = await processVisualization(file);
    if (!result.success || !result.data) throw new Error(result.error || 'Processing failed');
    if (shouldCacheVisualizationData(audioUrl)) visualizationDataByAudioUrl.set(audioUrl, result.data);
    return result.data;
  })();

  visualizationProcessingByAudioUrl.set(audioUrl, processPromise);
  try { return await processPromise; }
  finally { visualizationProcessingByAudioUrl.delete(audioUrl); }
};

import {
  STEM_HEX,
  DEFAULT_STEM_ORDER,
  ringCenterRadius,
  ringBandHalf,
  drawOrbitRing,
  drawStemLabel,
  stemGalaxyColor,
} from '../utils/stemVizHelpers';

// ── Props ──────────────────────────────────────────────────────────────────────

export interface StemVisualizationViewProps {
  audioUrl: string;
  initialVisualizationData?: VisualizationCacheData | null;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  isVisible?: boolean;
  audioRef?: React.RefObject<HTMLAudioElement | null>;
  /** URL of the original (unseparated) input audio — used for snippet playback in puff panels. */
  inputAudioUrl?: string;
  onStemSelectionChange?: (stems: string[], stemAudioUris?: Record<string, string>) => void;
  onSeek?: (time: number) => void;
  onDataLoaded?: (data: { songStructure: SongSection[]; flamingoChords: FlamingoChord[] }) => void;
  onVisualizationDataReady?: (data: VisualizationCacheData) => void;
  onVisualizationComplete?: () => void;
  onBackendDone?: () => void;
  onVisualizationError?: (message: string) => void;
}

// ── Constants ──────────────────────────────────────────────────────────────────

const STEM_NAMES = ['drums', 'bass', 'vocals', 'other'] as const;
const CLICK_TOLERANCE_FACTOR = 0.55;
const DRAG_GRAB_RADIUS = 32;

// Per-stem loudness floor passed to analyzeAudioUrl.
// Drums need 0 — hi-hats and ghost notes are quiet but musically important.
const STEM_MIN_RMS: Record<string, number> = {
  drums:  0,
  bass:   0.008,
  vocals: 0.02,
  other:  0.01,
};

// ── Melodic similarity puffs ──────────────────────────────────────────────────
// Each puff marks a matching window between the input audio and one stem.
// inputStart/End = time range in the original input audio.
// stemStart/End  = corresponding time range in the separated stem audio.
// TODO: replace hardcoded values with backend similarity analysis results.

interface StemPuff {
  stem: string;
  inputStart: number; inputEnd: number;
  stemStart:  number; stemEnd:  number;
  label: string;
  rgb: [number, number, number];  // highlight color — will carry similarity score in the future
}

const STEM_PUFFS: StemPuff[] = [
  { stem: 'other',  inputStart: 2.0, inputEnd: 4.0,  stemStart: 0.0, stemEnd: 1.0,  label: 'Motif A',    rgb: [140, 160, 255] },
  { stem: 'vocals', inputStart: 5.0, inputEnd: 10.0, stemStart: 5.0, stemEnd: 10.0, label: 'Motif B',    rgb: [ 80, 220, 170] },
  { stem: 'other',  inputStart: 2.0, inputEnd: 4.0,  stemStart: 8.0, stemEnd: 9.0,  label: 'Motif A',    rgb: [140, 160, 255] },
];

const PUFF_SPAWN_MS     = 1300;
const PUFF_FADE_DELAY_S = 2.0;   // seconds after max(inputEnd, stemEnd) before fading
const PUFF_FADE_DUR_S   = 1.5;   // seconds to fade to invisible
const PUFF_HIT_PX       = 60;    // hover/click radius per window position

type PuffState = { anchorFeatIdx: number; windowFeatIndices: number[]; spawnTime: number } | null;

// ── Puff drawing helpers ──────────────────────────────────────────────────────

function puffEaseOut3(t: number) { return 1 - Math.pow(1 - t, 3); }
function puffLerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function drawStemSpanningPuff(
  ctx: CanvasRenderingContext2D,
  positions: { x: number; y: number }[],
  anchor: { x: number; y: number },
  rgb: [number, number, number],
  label: string,
  now: number,
  fadeAlpha = 1,
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

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

// ── Component ──────────────────────────────────────────────────────────────────

export function StemVisualizationView({
  audioUrl,
  initialVisualizationData,
  currentTime,
  isVisible = true,
  audioRef: externalAudioRef,
  inputAudioUrl,
  onStemSelectionChange,
  onDataLoaded,
  onVisualizationDataReady,
  onVisualizationComplete,
  onBackendDone,
  onVisualizationError,
}: StemVisualizationViewProps) {
  // ── Loading ───────────────────────────────────────────────────────────────────
  const [isLoading, setIsLoading]             = useState(true);
  const [loadingProgress, setLoadingProgress] = useState('');
  const backendDoneRef = useRef(false);

  // ── Interaction ───────────────────────────────────────────────────────────────
  const [selectedStems, setSelectedStems] = useState<string[]>([]);
  const [stemOrder, setStemOrder]         = useState<string[]>(DEFAULT_STEM_ORDER);

  // ── Snippet playback (for puff panel) ────────────────────────────────────────
  const [selectedPuff,   setSelectedPuff]   = useState<StemPuff | null>(null);
  const [playingSnippet, setPlayingSnippet] = useState<'input' | 'stem' | null>(null);
  const selectedPuffRef       = useRef<StemPuff | null>(null);
  const playingSnippetRef     = useRef<'input' | 'stem' | null>(null);
  const snippetTimerRef       = useRef<ReturnType<typeof setTimeout> | null>(null);
  const snippetAudioRef       = useRef<HTMLAudioElement>(new Audio());
  // Whether main audio was playing when a puff panel was opened (used to resume on close)
  const wasPlayingBeforePuffRef = useRef(false);

  useEffect(() => { selectedPuffRef.current   = selectedPuff;   }, [selectedPuff]);
  useEffect(() => { playingSnippetRef.current = playingSnippet; }, [playingSnippet]);

  // ── Canvas ────────────────────────────────────────────────────────────────────
  const canvasRef       = useRef<HTMLCanvasElement | null>(null);
  const nebulaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef          = useRef<number | null>(null);

  // ── Visualization data ────────────────────────────────────────────────────────
  const vizDataRef            = useRef<VisualizationCacheData | null>(null);
  const stemFeaturesRef       = useRef<Record<string, FrameFeatures[]>>({});
  const stemPitchMediansRef   = useRef<Record<string, number>>({});
  const durationRef           = useRef(0);
  const stemAudioUrisRef      = useRef<Record<string, string>>({});
  const processingAudioUrlRef = useRef<string | null>(null);

  // Puff animation state and hit-testing positions
  const matchStatesRef = useRef<PuffState[]>(STEM_PUFFS.map(() => null));
  // Updated each frame with settled puff positions for cursor/click detection.
  // windowPos covers the full glow region; anchor is the label anchor.
  const puffHitRef = useRef<Array<{ anchor: {x:number;y:number}; windowPos: {x:number;y:number}[]; puff: StemPuff }>>([]);

  // ── RAF-accessible mirrors of state ──────────────────────────────────────────
  const selectedStemsRef = useRef<string[]>([]);
  const stemOrderRef     = useRef<string[]>(DEFAULT_STEM_ORDER);
  const currentTimeRef   = useRef(0);

  useEffect(() => { selectedStemsRef.current = selectedStems; }, [selectedStems]);
  useEffect(() => { stemOrderRef.current     = stemOrder;     }, [stemOrder]);
  useEffect(() => { currentTimeRef.current   = currentTime;   }, [currentTime]);

  // ── Drag state ────────────────────────────────────────────────────────────────
  const dragStemRef      = useRef<string | null>(null);
  const dragStartIdxRef  = useRef(-1);
  const dragTargetIdxRef = useRef(-1);
  const nebulaPuffsRef   = useRef<NebulaPuff[]>([]);

  // Snippet audio cleanup
  useEffect(() => {
    const sa = snippetAudioRef.current;
    return () => { sa.pause(); };
  }, []);

  // ── Effect 1: load and process audio ─────────────────────────────────────────
  useEffect(() => {
    if (!audioUrl) return;
    if (processingAudioUrlRef.current === audioUrl) return;
    processingAudioUrlRef.current = audioUrl;
    let cancelled = false;

    stemFeaturesRef.current     = {};
    stemPitchMediansRef.current = {};
    stemAudioUrisRef.current    = {};
    nebulaPuffsRef.current      = [];
    backendDoneRef.current      = false;
    matchStatesRef.current      = STEM_PUFFS.map(() => null);
    puffHitRef.current          = [];

    const run = async () => {
      try {
        setIsLoading(true);
        setLoadingProgress('');

        let vizData: VisualizationCacheData;

        if (initialVisualizationData) {
          backendDoneRef.current = true;
          vizData = initialVisualizationData;
          if (vizData.stemAudioUris) stemAudioUrisRef.current = vizData.stemAudioUris as Record<string, string>;
          onBackendDone?.();
        } else {
          const raw = await processVisualizationForAudioUrl(audioUrl);
          if (cancelled) return;
          backendDoneRef.current = true;
          vizData = toVisualizationCacheData(raw);
          stemAudioUrisRef.current = (raw.stemAudioUris ?? {}) as Record<string, string>;
          onVisualizationDataReady?.(vizData);
          onBackendDone?.();
        }

        if (cancelled) return;
        vizDataRef.current  = vizData;
        durationRef.current = vizData.duration;
        onDataLoaded?.({ songStructure: vizData.songStructure ?? [], flamingoChords: vizData.flamingoChords ?? [] });

        const stemEntries = Object.entries(stemAudioUrisRef.current).filter(([s]) => STEM_NAMES.includes(s as any));
        const totalStems  = stemEntries.length;
        let stemsCompleted = 0;

        await Promise.all(
          stemEntries.map(([stem, uri]) =>
            analyzeAudioUrl(
              uri,
              (feats) => {
                if (cancelled) return;
                stemFeaturesRef.current[stem] = feats;
                const pitchNorms = feats.map(f => mapPitch(f.pitch)).filter(p => p > 0).sort((a, b) => a - b);
                stemPitchMediansRef.current[stem] = pitchNorms.length
                  ? pitchNorms[Math.floor(pitchNorms.length / 2)]
                  : 0.5;
                stemsCompleted++;
                setLoadingProgress(`Analyzing stem ${stemsCompleted}/${totalStems}…`);
              },
              () => cancelled,
              STEM_MIN_RMS[stem] ?? 0.01
            )
          )
        );

        if (cancelled) return;
        setIsLoading(false);
        setLoadingProgress('');
        onVisualizationComplete?.();
      } catch (err) {
        if (!cancelled) {
          const raw = err instanceof Error ? err.message : 'Unknown error';
          const isConnRefused = raw.includes('Failed to fetch') || raw.includes('NetworkError') || raw.includes('ERR_CONNECTION_REFUSED');
          const msg = isConnRefused
            ? 'Backend not reachable. Start it with: uvicorn audio_analysis:app --reload'
            : `Error: ${raw}`;
          setLoadingProgress(msg);
          onVisualizationError?.(msg);
          onVisualizationComplete?.();
        }
        if (processingAudioUrlRef.current === audioUrl) processingAudioUrlRef.current = null;
      }
    };

    run();
    return () => {
      cancelled = true;
      if (processingAudioUrlRef.current === audioUrl) processingAudioUrlRef.current = null;
    };
  }, [audioUrl, initialVisualizationData]);

  // ── Effect 2: render loop ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!isVisible || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const nebula = nebulaCanvasRef.current;
      if (!canvas || !nebula) return;
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
        if (nCtx) drawNebulaLayer(nCtx, nebulaPuffsRef.current, now);
      }

      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1;

      if (!backendDoneRef.current) {
        // Phase 1: Demucs still running — parent keeps input visualization visible.
      } else {
        const t        = externalAudioRef?.current?.currentTime ?? currentTimeRef.current;
        const order    = stemOrderRef.current;
        const selected = selectedStemsRef.current;
        const numRings = order.length;
        const orbitDur = Math.min(Math.max(durationRef.current, 1), 10);

        const dragTargetIdx = dragTargetIdxRef.current;
        const dragStem      = dragStemRef.current;

        // Helper: orbital XY for a feature on its stem ring
        const featOrbitalPos = (feat: FrameFeatures, stemName: string) => {
          const stemIdx = order.indexOf(stemName);
          if (stemIdx < 0) return { x: cx, y: cy };
          const rR   = ringCenterRadius(stemIdx, numRings, baseR);
          const bH   = ringBandHalf(numRings, baseR);
          const med  = stemPitchMediansRef.current[stemName] ?? 0.5;
          const pn   = mapPitch(feat.pitch);
          const dev  = (pn - med) * bH * 2;
          const oi   = Math.floor(feat.time / orbitDur);
          const op   = (feat.time - oi * orbitDur) / orbitDur;
          const ang  = op * Math.PI * 2 + oi * 0.3 + t * 0.1;
          const orbR = rR + dev;
          return { x: cx + Math.cos(ang) * orbR, y: cy + Math.sin(ang) * orbR };
        };

        // ── 1. Similarity puffs — drawn BEFORE stars so they appear behind ───────
        const newHits: typeof puffHitRef.current = [];
        for (let mi = 0; mi < STEM_PUFFS.length; mi++) {
          const puff    = STEM_PUFFS[mi];
          const stemIdx = order.indexOf(puff.stem);
          if (stemIdx < 0) continue;
          const feats = stemFeaturesRef.current[puff.stem];
          if (!feats?.length) continue;

          if (t < puff.stemStart) { matchStatesRef.current[mi] = null; continue; }

          if (!matchStatesRef.current[mi]) {
            const windowIndices: number[] = [];
            for (let fi = 0; fi < feats.length; fi++) {
              if (feats[fi].time >= puff.stemStart && feats[fi].time <= puff.stemEnd)
                windowIndices.push(fi);
            }
            let anchorIdx = 0, bestD = Infinity;
            for (let fi = 0; fi < feats.length; fi++) {
              const d = Math.abs(feats[fi].time - puff.stemStart);
              if (d < bestD) { bestD = d; anchorIdx = fi; }
            }
            if (!windowIndices.length) windowIndices.push(anchorIdx);
            matchStatesRef.current[mi] = { anchorFeatIdx: anchorIdx, windowFeatIndices: windowIndices, spawnTime: now };
          }

          const state    = matchStatesRef.current[mi]!;
          const progress = Math.min((now - state.spawnTime) / PUFF_SPAWN_MS, 1);
          const rgb      = puff.rgb;
          const anchor   = featOrbitalPos(feats[state.anchorFeatIdx], puff.stem);

          // Fade out after max(inputEnd, stemEnd) + delay
          const puffEnd  = Math.max(puff.inputEnd, puff.stemEnd);
          const fadeAlpha = t > puffEnd + PUFF_FADE_DELAY_S
            ? Math.max(0, 1 - (t - puffEnd - PUFF_FADE_DELAY_S) / PUFF_FADE_DUR_S)
            : 1;
          if (fadeAlpha <= 0) continue;

          if (progress < 1) {
            // Fly-in comet from canvas center to anchor (same as ComparisonPage)
            const [r, g, b] = rgb;
            for (let trail = 0; trail < 10; trail++) {
              const tp = Math.max(0, progress - trail * 0.018);
              const te = puffEaseOut3(tp);
              ctx.globalAlpha = (0.85 - trail * 0.08) * (1 - progress * 0.15) * fadeAlpha;
              ctx.fillStyle = `rgba(${r},${g},${b},1)`;
              ctx.beginPath();
              ctx.arc(puffLerp(cx, anchor.x, te), puffLerp(cy, anchor.y, te),
                Math.max(10 - trail, 1) * 0.8, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.globalAlpha = 1;
          } else {
            const windowPositions = state.windowFeatIndices.map(fi => featOrbitalPos(feats[fi], puff.stem));
            drawStemSpanningPuff(ctx, windowPositions, anchor, rgb, puff.label, now, fadeAlpha);
            newHits.push({ anchor, windowPos: windowPositions, puff });
          }
        }
        puffHitRef.current = newHits;

        // ── 2. Orbit ring lines ──────────────────────────────────────────────────
        for (let i = 0; i < numRings; i++) {
          const stem             = order[i];
          const ringR            = ringCenterRadius(i, numRings, baseR);
          const isSelected       = selected.includes(stem);
          const isTheDraggedStem = dragStem === stem;
          const isDragTarget     = dragStem !== null && dragTargetIdx === i && !isTheDraggedStem;

          if (dragStem !== null && !isTheDraggedStem) {
            // During drag: all non-source rings become dashed drop-zone hints.
            // The ring the cursor is closest to goes bold to confirm "drop here".
            ctx.save();
            ctx.beginPath();
            ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
            ctx.strokeStyle = STEM_HEX[stem] ?? '#888';
            if (isDragTarget) {
              ctx.globalAlpha = 0.85;
              ctx.lineWidth   = 2.5;
              ctx.setLineDash([]);
            } else {
              ctx.globalAlpha = 0.35;
              ctx.lineWidth   = 1;
              ctx.setLineDash([5, 6]);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
          } else {
            drawOrbitRing(ctx, cx, cy, ringR, STEM_HEX[stem] ?? '#888', isSelected, false);
          }
        }

        // ── 3. Stars per stem ────────────────────────────────────────────────────
        for (let i = 0; i < numRings; i++) {
          const stem = order[i];
          const feats = stemFeaturesRef.current[stem];
          if (!feats?.length) continue;

          const ringR     = ringCenterRadius(i, numRings, baseR);
          const bandHalf  = ringBandHalf(numRings, baseR);
          const median    = stemPitchMediansRef.current[stem] ?? 0.5;
          const isSelected    = selected.includes(stem);
          const isAnySelected = selected.length > 0;
          const dimAlpha  = isAnySelected && !isSelected ? 0.18 : 1.0;

          for (const feat of feats) {
            const pn  = mapPitch(feat.pitch);
            const dev = (pn - median) * bandHalf * 2;
            const orbR = ringR + dev;
            const color = stemGalaxyColor(stem, feat.rms, feat.centroid);
            drawFeatureNote(ctx, feat, t, cx, cy, baseR, orbitDur, dimAlpha, orbR, color);
          }

          const isBeingDragged = dragStem === stem;
          // Don't dim the dragged stem's label — it should stay identifiable while moving
          const isDimmed = !isBeingDragged && isAnySelected && !isSelected;
          drawStemLabel(ctx, cx, cy, ringR, stem, STEM_HEX[stem] ?? '#888', isSelected, isDimmed);
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
  }, [isVisible]);

  // ── Stem selection ────────────────────────────────────────────────────────────

  const handleStemClick = useCallback((stem: string) => {
    setSelectedStems(prev => {
      const next = prev.includes(stem) ? prev.filter(s => s !== stem) : [...prev, stem];
      selectedStemsRef.current = next;
      onStemSelectionChange?.(next, stemAudioUrisRef.current);
      return next;
    });
  }, [onStemSelectionChange]);

  const handleClearSelection = useCallback(() => {
    setSelectedStems([]);
    selectedStemsRef.current = [];
    onStemSelectionChange?.([], stemAudioUrisRef.current);
  }, [onStemSelectionChange]);

  // ── Snippet playback (mirrors ComparisonPage.playSnippet) ────────────────────

  const playSnippet = useCallback((type: 'input' | 'stem') => {
    const puff = selectedPuffRef.current;
    if (!puff) return;
    const sa = snippetAudioRef.current;

    if (playingSnippetRef.current === type) {
      if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
      sa.pause();
      setPlayingSnippet(null);
      return;
    }

    if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
    sa.pause();

    const [src, start, end] = type === 'input'
      ? [inputAudioUrl ?? '', puff.inputStart, puff.inputEnd]
      : [stemAudioUrisRef.current[puff.stem] ?? '', puff.stemStart, puff.stemEnd];

    if (!src) return;
    sa.src = src;
    sa.currentTime = start;
    void sa.play().catch(() => {});
    setPlayingSnippet(type);
    snippetTimerRef.current = setTimeout(() => {
      sa.pause();
      setPlayingSnippet(null);
    }, (end - start) * 1000);
  }, [inputAudioUrl]);

  const closePanel = useCallback(() => {
    if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
    snippetAudioRef.current.pause();
    setSelectedPuff(null);
    setPlayingSnippet(null);
    // Resume main audio (and any active stem elements via App.tsx's play event listener)
    if (wasPlayingBeforePuffRef.current) {
      wasPlayingBeforePuffRef.current = false;
      void externalAudioRef?.current?.play().catch(() => {});
    }
  }, [externalAudioRef]);

  // ── Canvas hit-testing ────────────────────────────────────────────────────────

  const getCanvasMetrics = () => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect  = canvas.getBoundingClientRect();
    const baseR = Math.min(rect.width, rect.height) * 0.2;
    return { rect, baseR, cx: rect.width / 2, cy: rect.height / 2 };
  };

  const stemAtDistance = useCallback((dist: number, baseR: number): string | null => {
    const order = stemOrderRef.current;
    const num   = order.length;
    const band  = ringBandHalf(num, baseR);
    for (let i = num - 1; i >= 0; i--) {
      const r = ringCenterRadius(i, num, baseR);
      if (Math.abs(dist - r) <= band * CLICK_TOLERANCE_FACTOR * 2) return order[i];
    }
    return null;
  }, []);

  // ── Pointer events ────────────────────────────────────────────────────────────

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const m = getCanvasMetrics();
    if (!m) return;
    const mx = e.clientX - m.rect.left, my = e.clientY - m.rect.top;
    const order = stemOrderRef.current, num = order.length;
    for (let i = 0; i < num; i++) {
      const r = ringCenterRadius(i, num, m.baseR);
      if (Math.hypot(mx - (m.cx + r), my - m.cy) < DRAG_GRAB_RADIUS) {
        dragStemRef.current      = order[i];
        dragStartIdxRef.current  = i;
        dragTargetIdxRef.current = i;
        e.currentTarget.setPointerCapture(e.pointerId);
        return;
      }
    }
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;

    if (dragStemRef.current) {
      const m = getCanvasMetrics();
      if (!m) return;
      const dist = Math.hypot(e.clientX - m.rect.left - m.cx, e.clientY - m.rect.top - m.cy);
      const num  = stemOrderRef.current.length;
      let closest = 0, minDiff = Infinity;
      for (let i = 0; i < num; i++) {
        const d = Math.abs(dist - ringCenterRadius(i, num, m.baseR));
        if (d < minDiff) { minDiff = d; closest = i; }
      }
      dragTargetIdxRef.current = closest;
      if (canvas) canvas.style.cursor = 'grabbing';
      return;
    }

    // Update cursor based on what's under the pointer
    if (canvas && !selectedPuffRef.current) {
      const rect  = canvas.getBoundingClientRect();
      const mx    = e.clientX - rect.left, my = e.clientY - rect.top;
      const baseR = Math.min(rect.width, rect.height) * 0.2;
      const cxC   = rect.width / 2, cyC = rect.height / 2;

      // Puff glow → pointer
      const overPuff = puffHitRef.current.some(hit =>
        Math.hypot(mx - hit.anchor.x, my - hit.anchor.y) < PUFF_HIT_PX ||
        hit.windowPos.some(p => Math.hypot(mx - p.x, my - p.y) < PUFF_HIT_PX)
      );
      if (overPuff) { canvas.style.cursor = 'pointer'; return; }

      // Stem label → grab (signals draggability)
      const order = stemOrderRef.current, num = order.length;
      for (let i = 0; i < num; i++) {
        const r = ringCenterRadius(i, num, baseR);
        if (Math.hypot(mx - (cxC + r + 10), my - cyC) < 44) {
          canvas.style.cursor = 'grab';
          return;
        }
      }

      canvas.style.cursor = 'default';
    }
  }, []);

  const handlePointerUp = useCallback(() => {
    const dragStem = dragStemRef.current, startIdx = dragStartIdxRef.current, targetIdx = dragTargetIdxRef.current;
    if (dragStem !== null && startIdx !== targetIdx && targetIdx >= 0) {
      setStemOrder(prev => {
        const next    = [...prev];
        const fromIdx = next.indexOf(dragStem);
        if (fromIdx < 0) return prev;
        [next[targetIdx], next[fromIdx]] = [next[fromIdx], next[targetIdx]];
        stemOrderRef.current = next;
        return next;
      });
    }
    dragStemRef.current = null; dragStartIdxRef.current = -1; dragTargetIdxRef.current = -1;
  }, []);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragStartIdxRef.current !== -1) return;
    // Click outside panel closes it
    if (selectedPuffRef.current) { closePanel(); return; }
    const m = getCanvasMetrics();
    if (!m) return;
    const mx = e.clientX - m.rect.left, my = e.clientY - m.rect.top;

    // ── 1. Stem label click takes highest priority (prevents puff overlap stealing it) ──
    const order = stemOrderRef.current, num = order.length;
    for (let i = 0; i < num; i++) {
      const r      = ringCenterRadius(i, num, m.baseR);
      const labelX = m.cx + r + 10;
      if (Math.hypot(mx - labelX, my - m.cy) < 44) {
        handleStemClick(order[i]);
        return;
      }
    }

    // ── 2. Puff glow region — any window position within hit radius ──────────────
    for (const hit of puffHitRef.current) {
      const overPuff =
        Math.hypot(mx - hit.anchor.x, my - hit.anchor.y) < PUFF_HIT_PX ||
        hit.windowPos.some(p => Math.hypot(mx - p.x, my - p.y) < PUFF_HIT_PX);
      if (overPuff) {
        // Pause main audio (App.tsx's pause listener cascades to stem elements)
        wasPlayingBeforePuffRef.current = !!(externalAudioRef?.current && !externalAudioRef.current.paused);
        externalAudioRef?.current?.pause();
        setSelectedPuff(hit.puff);
        setPlayingSnippet(null);
        if (snippetTimerRef.current) clearTimeout(snippetTimerRef.current);
        snippetAudioRef.current.pause();
        return;
      }
    }

    // ── 3. Ring arc ──────────────────────────────────────────────────────────────
    const dist = Math.hypot(mx - m.cx, my - m.cy);
    const stem = stemAtDistance(dist, m.baseR);
    if (stem) { handleStemClick(stem); } else { handleClearSelection(); }
  }, [stemAtDistance, handleStemClick, handleClearSelection, closePanel]);

  // ── Render ────────────────────────────────────────────────────────────────────

  const dimCanvas = !!selectedPuff;

  return (
    <div style={{
      backgroundColor: '#0d0d0d', flex: 1, width: '100%', height: '100%',
      borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)',
      overflow: 'hidden', position: 'relative',
    }}>
      <canvas ref={nebulaCanvasRef} style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} />
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', position: 'relative',
          filter: dimCanvas ? 'brightness(0.3)' : 'none',
          transition: 'filter 0.3s ease',
        }}
        onClick={handleCanvasClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      />

      {/* Error message */}
      {!isLoading && loadingProgress && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
          pointerEvents: 'none', color: 'rgba(234,234,234,0.6)', fontSize: 13,
          fontFamily: 'Inter, sans-serif', textAlign: 'center', padding: '0 16px',
        }}>
          {loadingProgress}
        </div>
      )}

      {/* Puff snippet panel — identical interaction model to ComparisonPage.SnippetPanel */}
      {selectedPuff && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
          width: 300, background: 'rgba(14,14,20,0.97)', borderRadius: 14,
          border: '1px solid rgba(255,255,255,0.1)', padding: '22px 22px 18px',
          display: 'flex', flexDirection: 'column', gap: 14,
          backdropFilter: 'blur(20px)', boxShadow: '0 28px 70px rgba(0,0,0,0.75)',
          zIndex: 20,
        }}>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
              background: `rgb(${selectedPuff.rgb.join(',')})` }} />
            <span style={{ fontWeight: 700, fontSize: 16, flex: 1,
              color: `rgb(${selectedPuff.rgb.join(',')})` }}>
              {selectedPuff.label}
            </span>
            <button
              onClick={closePanel}
              style={{ background: 'transparent', border: 'none', color: 'rgba(255,255,255,0.3)',
                cursor: 'pointer', fontSize: 16, padding: '0 2px', fontFamily: 'inherit', lineHeight: 1 }}
            >✕</button>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: 'rgba(255,255,255,0.3)' }}>
            Play each snippet to hear the similarity
          </p>
          {/* Snippet buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(['input', 'stem'] as const).map(type => {
              const isActive   = playingSnippet === type;
              const puffColor  = `rgb(${selectedPuff.rgb.join(',')})`;
              const accent     = type === 'input' ? '#aaa' : puffColor;
              const start    = type === 'input' ? selectedPuff.inputStart : selectedPuff.stemStart;
              const end      = type === 'input' ? selectedPuff.inputEnd   : selectedPuff.stemEnd;
              const stemName = type === 'input' ? 'Input' : selectedPuff.stem.charAt(0).toUpperCase() + selectedPuff.stem.slice(1);
              return (
                <button
                  key={type}
                  onClick={() => playSnippet(type)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12, padding: '11px 14px',
                    borderRadius: 8, cursor: 'pointer', fontFamily: 'inherit', textAlign: 'left',
                    background: isActive
                      ? (type === 'input' ? 'rgba(170,170,170,0.15)' : `rgba(${selectedPuff.rgb.join(',')},0.15)`)
                      : 'rgba(255,255,255,0.04)',
                    border: `1px solid ${isActive ? accent : 'rgba(255,255,255,0.09)'}`,
                    color: isActive ? accent : 'rgba(255,255,255,0.72)',
                    transition: 'background 0.15s, border-color 0.15s',
                  }}
                >
                  <span style={{ fontSize: 15, width: 18, textAlign: 'center', flexShrink: 0 }}>
                    {isActive ? '⏸' : '▶'}
                  </span>
                  <span>
                    <div style={{ fontSize: 13, fontWeight: 600, lineHeight: '1.3' }}>{stemName}</div>
                    <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', marginTop: 1 }}>
                      {fmt(start)} – {fmt(end)}
                    </div>
                  </span>
                </button>
              );
            })}
          </div>
          <button
            onClick={closePanel}
            style={{
              background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.09)',
              color: 'rgba(255,255,255,0.5)', borderRadius: 7, padding: 9, fontSize: 12,
              cursor: 'pointer', fontFamily: 'inherit', width: '100%',
            }}
          >Resume playback</button>
        </div>
      )}

      {/* Stem legend */}
      {!isLoading && Object.keys(stemFeaturesRef.current).length > 0 && (
        <div style={{
          position: 'absolute', bottom: 10, left: 10,
          background: 'rgba(0,0,0,0.75)', border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 6, padding: '10px 14px',
          display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', marginBottom: 2, letterSpacing: '0.06em' }}>
            CLICK RING TO SOLO · DRAG NAME TO REORDER
          </div>
          {stemOrder.map(stem => {
            const isSelected    = selectedStems.includes(stem);
            const isAnySelected = selectedStems.length > 0;
            return (
              <div key={stem} onClick={() => handleStemClick(stem)} style={{
                display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                opacity: isAnySelected && !isSelected ? 0.3 : 1,
              }}>
                <div style={{
                  width: 10, height: 10, borderRadius: '50%', background: STEM_HEX[stem], flexShrink: 0,
                  boxShadow: isSelected ? `0 0 6px ${STEM_HEX[stem]}` : 'none',
                }} />
                <span style={{ fontSize: 12, color: isSelected ? '#fff' : 'rgba(255,255,255,0.6)',
                  fontWeight: isSelected ? 600 : 400, textTransform: 'capitalize' }}>
                  {stem}
                </span>
              </div>
            );
          })}
          {selectedStems.length > 0 && (
            <div onClick={handleClearSelection} style={{
              marginTop: 4, fontSize: 11, color: 'rgba(255,255,255,0.3)', cursor: 'pointer',
              borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 6, textAlign: 'center',
            }}>
              Clear selection
            </div>
          )}
        </div>
      )}
    </div>
  );
}
