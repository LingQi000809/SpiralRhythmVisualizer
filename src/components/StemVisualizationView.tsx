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
import { drawNebulaLayer, spawnNebulaPuff, chordRootHue } from '../utils/visMidiHelpers';
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
    if (!response.ok) {
      throw new Error(`Failed to fetch audio for visualization: ${response.status} ${response.statusText}`);
    }
    const blob = await response.blob();
    const file = new File([blob], 'audio.wav', { type: blob.type });
    const result = await processVisualization(file);
    if (!result.success || !result.data) throw new Error(result.error || 'Processing failed');
    if (shouldCacheVisualizationData(audioUrl)) visualizationDataByAudioUrl.set(audioUrl, result.data);
    return result.data;
  })();

  visualizationProcessingByAudioUrl.set(audioUrl, processPromise);
  try {
    return await processPromise;
  } finally {
    visualizationProcessingByAudioUrl.delete(audioUrl);
  }
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
  onStemSelectionChange?: (stems: string[], stemAudioUris?: Record<string, string>) => void;
  onSeek?: (time: number) => void;
  onDataLoaded?: (data: { songStructure: SongSection[]; flamingoChords: FlamingoChord[] }) => void;
  onVisualizationDataReady?: (data: VisualizationCacheData) => void;
  onVisualizationComplete?: () => void;
  // Fires as soon as the Demucs backend returns (before per-stem analysis).
  // Parent uses this to start the suck-in transition on the input visualization.
  onBackendDone?: () => void;
  // Experiment-only: called when analysis fails so the parent can suppress auto-transition.
  onVisualizationError?: (message: string) => void;
}

// ── Constants ──────────────────────────────────────────────────────────────────

const STEM_NAMES = ['drums', 'bass', 'vocals', 'other'] as const;
const CLICK_TOLERANCE_FACTOR = 0.55;
const DRAG_GRAB_RADIUS = 32;

// ── Component ──────────────────────────────────────────────────────────────────

export function StemVisualizationView({
  audioUrl,
  initialVisualizationData,
  currentTime,
  isVisible = true,
  audioRef: externalAudioRef,
  onStemSelectionChange,
  onDataLoaded,
  onVisualizationDataReady,
  onVisualizationComplete,
  onBackendDone,
  onVisualizationError,
}: StemVisualizationViewProps) {
  // ── Loading state ────────────────────────────────────────────────────────────
  const [isLoading, setIsLoading]         = useState(true);
  const [loadingProgress, setLoadingProgress] = useState('');
  const backendDoneRef = useRef(false);

  // ── Interaction state ────────────────────────────────────────────────────────
  const [selectedStems, setSelectedStems] = useState<string[]>([]);
  const [stemOrder, setStemOrder]         = useState<string[]>(DEFAULT_STEM_ORDER);

  // ── Canvas ───────────────────────────────────────────────────────────────────
  const canvasRef       = useRef<HTMLCanvasElement | null>(null);
  const nebulaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef          = useRef<number | null>(null);

  // ── Visualization data ────────────────────────────────────────────────────────
  const vizDataRef          = useRef<VisualizationCacheData | null>(null);
  const stemFeaturesRef     = useRef<Record<string, FrameFeatures[]>>({});
  const stemPitchMediansRef = useRef<Record<string, number>>({});
  const durationRef         = useRef(0);
  const stemAudioUrisRef    = useRef<Record<string, string>>({});
  const processingAudioUrlRef = useRef<string | null>(null);

  // ── RAF-accessible mirrors of state ─────────────────────────────────────────
  const selectedStemsRef = useRef<string[]>([]);
  const stemOrderRef     = useRef<string[]>(DEFAULT_STEM_ORDER);
  const currentTimeRef   = useRef(0);

  useEffect(() => { selectedStemsRef.current = selectedStems; }, [selectedStems]);
  useEffect(() => { stemOrderRef.current     = stemOrder;     }, [stemOrder]);
  useEffect(() => { currentTimeRef.current   = currentTime;   }, [currentTime]);

  // ── Drag state ───────────────────────────────────────────────────────────────
  const dragStemRef      = useRef<string | null>(null);
  const dragStartIdxRef  = useRef(-1);
  const dragTargetIdxRef = useRef(-1);
  const nebulaPuffsRef   = useRef<NebulaPuff[]>([]);
  const lastChordRef     = useRef('');

  // ── Effect 1: load and process audio ─────────────────────────────────────────
  useEffect(() => {
    if (!audioUrl) return;
    if (processingAudioUrlRef.current === audioUrl) return;
    processingAudioUrlRef.current = audioUrl;
    let cancelled = false;

    // Reset all analysis state
    stemFeaturesRef.current     = {};
    stemPitchMediansRef.current = {};
    stemAudioUrisRef.current    = {};
    nebulaPuffsRef.current      = [];
    lastChordRef.current        = '';
    backendDoneRef.current      = false;

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
          // Signal parent that Demucs is done — parent will start the suck-in transition
          // on the input visualization before revealing the stem rings here.
          onBackendDone?.();
        }

        if (cancelled) return;
        vizDataRef.current  = vizData;
        durationRef.current = vizData.duration;

        onDataLoaded?.({
          songStructure:  vizData.songStructure  ?? [],
          flamingoChords: vizData.flamingoChords ?? [],
        });

        // Per-stem analysis — runs while parent plays the suck-in animation
        const stemEntries = Object.entries(stemAudioUrisRef.current).filter(
          ([stem]) => STEM_NAMES.includes(stem as any)
        );
        const totalStems = stemEntries.length;
        let stemsCompleted = 0;

        await Promise.all(
          stemEntries.map(([stem, uri]) =>
            analyzeAudioUrl(
              uri,
              (feats) => {
                if (cancelled) return;
                stemFeaturesRef.current[stem] = feats;
                const pitchNorms = feats
                  .map(f => mapPitch(f.pitch))
                  .filter(p => p > 0)
                  .sort((a, b) => a - b);
                stemPitchMediansRef.current[stem] =
                  pitchNorms.length > 0
                    ? pitchNorms[Math.floor(pitchNorms.length / 2)]
                    : 0.5;
                stemsCompleted++;
                setLoadingProgress(`Analyzing stem ${stemsCompleted}/${totalStems}…`);
              },
              () => cancelled
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

      // Nebula background
      const nebula = nebulaCanvasRef.current;
      if (nebula) {
        const nCtx = nebula.getContext('2d');
        if (nCtx) drawNebulaLayer(nCtx, nebulaPuffsRef.current, now);
      }

      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1;

      if (!backendDoneRef.current) {
        // Phase 1: Demucs still running.
        // Draw nothing — parent keeps the input visualization visible above us.
      } else {
        // Phase 2+: draw stem rings as analysis fills them in.
        const t = externalAudioRef?.current?.currentTime ?? currentTimeRef.current;
        const order    = stemOrderRef.current;
        const selected = selectedStemsRef.current;
        const numRings = order.length;
        const orbitDur = Math.min(Math.max(durationRef.current, 1), 10);

        const dragTargetIdx = dragTargetIdxRef.current;
        const dragStem      = dragStemRef.current;

        // Orbit ring lines
        for (let i = 0; i < numRings; i++) {
          const stem         = order[i];
          const ringR        = ringCenterRadius(i, numRings, baseR);
          const isSelected   = selected.includes(stem);
          const isDragTarget = dragStem !== null && dragTargetIdx === i && dragStem !== stem;
          drawOrbitRing(ctx, cx, cy, ringR, STEM_HEX[stem] ?? '#888', isSelected, isDragTarget);
        }

        // Notes per stem
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
            const pitchNorm = mapPitch(feat.pitch);
            const deviation = (pitchNorm - median) * bandHalf * 2;
            const orbitalR  = ringR + deviation;
            const color = stemGalaxyColor(stem, feat.rms, feat.centroid);
            drawFeatureNote(ctx, feat, t, cx, cy, baseR, orbitDur, dimAlpha, orbitalR, color);
          }

          const isBeingDragged = dragStem === stem;
          drawStemLabel(ctx, cx, cy, ringR, stem, STEM_HEX[stem] ?? '#888', isSelected, isBeingDragged || (isAnySelected && !isSelected));
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
      const next = prev.includes(stem)
        ? prev.filter(s => s !== stem)
        : [...prev, stem];
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

  // ── Canvas hit-testing ────────────────────────────────────────────────────────

  const getCanvasMetrics = () => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect  = canvas.getBoundingClientRect();
    const baseR = Math.min(rect.width, rect.height) * 0.2;
    const cx    = rect.width  / 2;
    const cy    = rect.height / 2;
    return { rect, baseR, cx, cy };
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
    const mx = e.clientX - m.rect.left;
    const my = e.clientY - m.rect.top;
    const order = stemOrderRef.current;
    const num   = order.length;
    for (let i = 0; i < num; i++) {
      const r      = ringCenterRadius(i, num, m.baseR);
      const labelX = m.cx + r;
      const labelY = m.cy;
      if (Math.hypot(mx - labelX, my - labelY) < DRAG_GRAB_RADIUS) {
        dragStemRef.current      = order[i];
        dragStartIdxRef.current  = i;
        dragTargetIdxRef.current = i;
        e.currentTarget.setPointerCapture(e.pointerId);
        return;
      }
    }
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragStemRef.current) return;
    const m = getCanvasMetrics();
    if (!m) return;
    const mx   = e.clientX - m.rect.left;
    const my   = e.clientY - m.rect.top;
    const dist = Math.hypot(mx - m.cx, my - m.cy);
    const num  = stemOrderRef.current.length;
    let closest = 0, minDiff = Infinity;
    for (let i = 0; i < num; i++) {
      const r    = ringCenterRadius(i, num, m.baseR);
      const diff = Math.abs(dist - r);
      if (diff < minDiff) { minDiff = diff; closest = i; }
    }
    dragTargetIdxRef.current = closest;
  }, []);

  const handlePointerUp = useCallback(() => {
    const dragStem  = dragStemRef.current;
    const startIdx  = dragStartIdxRef.current;
    const targetIdx = dragTargetIdxRef.current;
    if (dragStem !== null && startIdx !== targetIdx && targetIdx >= 0) {
      setStemOrder(prev => {
        const next    = [...prev];
        const fromIdx = next.indexOf(dragStem);
        if (fromIdx < 0) return prev;
        const displaced = next[targetIdx];
        next[targetIdx] = dragStem;
        next[fromIdx]   = displaced;
        stemOrderRef.current = next;
        return next;
      });
    }
    dragStemRef.current      = null;
    dragStartIdxRef.current  = -1;
    dragTargetIdxRef.current = -1;
  }, []);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragStartIdxRef.current !== -1) return;
    const m = getCanvasMetrics();
    if (!m) return;
    const mx   = e.clientX - m.rect.left;
    const my   = e.clientY - m.rect.top;
    const dist = Math.hypot(mx - m.cx, my - m.cy);
    const stem = stemAtDistance(dist, m.baseR);
    if (stem) { handleStemClick(stem); } else { handleClearSelection(); }
  }, [stemAtDistance, handleStemClick, handleClearSelection]);

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div style={{
      backgroundColor: '#0d0d0d',
      flex: 1,
      width: '100%',
      height: '100%',
      borderRadius: '8px',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      overflow: 'hidden',
      position: 'relative',
    }}>
      <canvas
        ref={nebulaCanvasRef}
        style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }}
      />
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', position: 'relative', cursor: dragStemRef.current ? 'grabbing' : 'default' }}
        onClick={handleCanvasClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      />

      {/* Error message */}
      {!isLoading && loadingProgress && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          pointerEvents: 'none',
          color: 'rgba(234, 234, 234, 0.6)',
          fontSize: 13, fontFamily: 'Inter, sans-serif',
          textAlign: 'center', padding: '0 16px',
        }}>
          {loadingProgress}
        </div>
      )}

      {/* Stem legend — shown only when fully loaded */}
      {!isLoading && Object.keys(stemFeaturesRef.current).length > 0 && (
        <div style={{
          position: 'absolute', bottom: 10, left: 10,
          background: 'rgba(0,0,0,0.75)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '6px',
          padding: '10px 14px',
          display: 'flex', flexDirection: 'column', gap: '6px',
        }}>
          <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.3)', marginBottom: '2px', letterSpacing: '0.06em' }}>
            CLICK RING OR NAME TO SOLO · DRAG NAME TO REORDER
          </div>
          {stemOrder.map(stem => {
            const isSelected    = selectedStems.includes(stem);
            const isAnySelected = selectedStems.length > 0;
            return (
              <div
                key={stem}
                onClick={() => handleStemClick(stem)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  cursor: 'pointer',
                  opacity: isAnySelected && !isSelected ? 0.3 : 1,
                }}
              >
                <div style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: STEM_HEX[stem], flexShrink: 0,
                  boxShadow: isSelected ? `0 0 6px ${STEM_HEX[stem]}` : 'none',
                }} />
                <span style={{
                  fontSize: '12px',
                  color: isSelected ? '#fff' : 'rgba(255,255,255,0.6)',
                  fontWeight: isSelected ? 600 : 400,
                  textTransform: 'capitalize',
                }}>
                  {stem}
                </span>
              </div>
            );
          })}
          {selectedStems.length > 0 && (
            <div
              onClick={handleClearSelection}
              style={{
                marginTop: '4px', fontSize: '11px', color: 'rgba(255,255,255,0.3)',
                cursor: 'pointer', borderTop: '1px solid rgba(255,255,255,0.08)',
                paddingTop: '6px', textAlign: 'center',
              }}
            >
              Clear selection
            </div>
          )}
        </div>
      )}
    </div>
  );
}
