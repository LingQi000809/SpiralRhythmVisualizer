import { useState, useRef, useEffect, useCallback } from 'react';
import { VisualizationWaitingView } from './components/VisualizationWaitingView';
import { StemVisualizationView } from './components/StemVisualizationView';
import type { InputData } from './utils/visMidiHelpers';

// ── Phase model ───────────────────────────────────────────────────────────────
// idle       → nothing active
// input      → Play pressed: input audio + VisualizationWaitingView
// generating → Generate: Demucs running, VisualizationWaitingView loops input
// output     → stem rings visible, output audio playing
type Phase = 'idle' | 'input' | 'generating' | 'output';

const STEM_NAMES = ['drums', 'bass', 'vocals', 'other'] as const;
const SUCK_IN_MS = 2800; // matches ComparisonPage TRANSITION_MS
const SYNC_DRIFT_MS = 60;

export default function App() {
  const [phase,          setPhase]          = useState<Phase>('idle');
  const [inputUrl,       setInputUrl]       = useState<string | null>(null);
  const [outputUrl,      setOutputUrl]      = useState<string | null>(null);
  const [inputFileName,  setInputFileName]  = useState('');
  const [outputFileName, setOutputFileName] = useState('');
  const [inputDuration,  setInputDuration]  = useState(0);
  const [showOutput,     setShowOutput]     = useState(false);
  const [statusText,     setStatusText]     = useState('');
  const [stemVizVisible, setStemVizVisible] = useState(false);
  const [inputMuted,     setInputMuted]     = useState(false);

  const inputAudioRef      = useRef<HTMLAudioElement | null>(null);
  const outputAudioRef     = useRef<HTMLAudioElement | null>(null);
  const transitionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const suckInTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);
  const outputErrorRef     = useRef(false);
  const stemAudioUrisRef   = useRef<Record<string, string>>({});
  const outputMixUrlRef    = useRef<string | null>(null);
  const selectedStemsRef   = useRef<string[]>([]);
  const stemVizVisibleRef  = useRef(false);
  const pendingVizComplete = useRef(false);
  const stemAudioEls = useRef<Partial<Record<typeof STEM_NAMES[number], HTMLAudioElement>>>({});

  // Canvas-based suck state — mutated directly so the draw loop reads it at 60fps
  // without triggering React re-renders. Mirrors the ComparisonPage approach.
  const suckInRef = useRef<{ isActive: boolean; startTime: number; durationMs: number }>({
    isActive: false, startTime: 0, durationMs: SUCK_IN_MS,
  });

  useEffect(() => { stemVizVisibleRef.current = stemVizVisible; }, [stemVizVisible]);

  const inputData: InputData[] | null =
    inputUrl && inputDuration > 0
      ? [{ inputStartTime: 0, inputEndTime: inputDuration, audioUrl: inputUrl }]
      : null;

  // ── Sync outputAudioRef events → stem elements ────────────────────────────
  useEffect(() => {
    const main = outputAudioRef.current;
    if (!main) return;
    const pauseAll  = () => { for (const el of Object.values(stemAudioEls.current)) el?.pause(); };
    const playAll   = () => {
      for (const name of selectedStemsRef.current) {
        const el = stemAudioEls.current[name as typeof STEM_NAMES[number]];
        if (el?.src) void el.play().catch(() => {});
      }
    };
    const syncAll   = () => {
      const ct = main.currentTime;
      for (const name of selectedStemsRef.current) {
        const el = stemAudioEls.current[name as typeof STEM_NAMES[number]];
        if (el?.src) el.currentTime = ct;
      }
    };
    main.addEventListener('pause',  pauseAll);
    main.addEventListener('play',   playAll);
    main.addEventListener('seeked', syncAll);
    main.addEventListener('ended',  pauseAll);
    return () => {
      main.removeEventListener('pause',  pauseAll);
      main.removeEventListener('play',   playAll);
      main.removeEventListener('seeked', syncAll);
      main.removeEventListener('ended',  pauseAll);
    };
  }, []);

  // ── Periodic drift correction ─────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      const main = outputAudioRef.current;
      if (!main || main.paused) return;
      for (const name of selectedStemsRef.current) {
        const el = stemAudioEls.current[name as typeof STEM_NAMES[number]];
        if (!el?.src || el.readyState < 2) continue;
        if (Math.abs(el.currentTime - main.currentTime) * 1000 > SYNC_DRIFT_MS)
          el.currentTime = main.currentTime;
      }
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // ── Input duration ────────────────────────────────────────────────────────
  useEffect(() => {
    const el = inputAudioRef.current;
    if (!el || !inputUrl) { setInputDuration(0); return; }
    const onMeta = () => setInputDuration(el.duration || 0);
    el.addEventListener('loadedmetadata', onMeta, { once: true });
    return () => el.removeEventListener('loadedmetadata', onMeta);
  }, [inputUrl]);

  // ── File handlers ─────────────────────────────────────────────────────────
  const handleInputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setInputFileName(file.name);
    setInputUrl(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
    setInputDuration(0);
  }, []);

  const handleOutputFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setOutputFileName(file.name);
    outputMixUrlRef.current = url;
    setOutputUrl(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
  }, []);

  // ── Play ──────────────────────────────────────────────────────────────────
  const handlePlay = useCallback(() => {
    if (!inputUrl) return;
    if (phase === 'idle') setPhase('input');
    const a = inputAudioRef.current;
    if (a) { a.loop = false; a.currentTime = 0; void a.play(); }
  }, [inputUrl, phase]);

  // ── Generate ─────────────────────────────────────────────────────────────
  const handleGenerate = useCallback(() => {
    if (!outputUrl) return;
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    if (suckInTimerRef.current)     clearTimeout(suckInTimerRef.current);
    outputErrorRef.current     = false;
    pendingVizComplete.current = false;
    suckInRef.current.isActive = false;
    setPhase('generating');
    setShowOutput(false);
    setStemVizVisible(false);
    setStatusText('');
    setInputMuted(false);
    // Loop input audio while waiting for Demucs
    const a = inputAudioRef.current;
    if (a) {
      a.loop   = true;
      a.muted  = false;
      if (a.paused) { a.currentTime = 0; void a.play(); }
    }
  }, [outputUrl]);

  // ── Output transition helper ──────────────────────────────────────────────
  const doOutputTransition = useCallback(() => {
    transitionTimerRef.current = setTimeout(() => {
      setPhase('output');
      setShowOutput(true);
      const o = outputAudioRef.current;
      if (o) { o.muted = false; o.currentTime = 0; void o.play(); }
    }, 600);
  }, []);

  // ── Demucs returned: freeze input audio, start canvas suck ───────────────
  const handleBackendDone = useCallback(() => {
    const a = inputAudioRef.current;
    if (a) { a.loop = false; a.pause(); }
    // Mutate the ref directly — VisualizationWaitingView's RAF loop reads it each frame
    suckInRef.current = { isActive: true, startTime: performance.now(), durationMs: SUCK_IN_MS };

    suckInTimerRef.current = setTimeout(() => {
      suckInRef.current.isActive = false;
      setStemVizVisible(true);
      if (pendingVizComplete.current) {
        pendingVizComplete.current = false;
        doOutputTransition();
      }
    }, SUCK_IN_MS);
  }, [doOutputTransition]);

  // ── Per-stem analysis finished ────────────────────────────────────────────
  const handleVisualizationComplete = useCallback(() => {
    if (outputErrorRef.current) { setStatusText(''); return; }
    setStatusText('');
    if (!stemVizVisibleRef.current) {
      // Suck still in progress — defer until it finishes
      pendingVizComplete.current = true;
      return;
    }
    doOutputTransition();
  }, [doOutputTransition]);

  const handleVisualizationError = useCallback((msg: string) => {
    outputErrorRef.current = true;
    setStatusText(`Error: ${msg}`);
  }, []);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    if (suckInTimerRef.current)     clearTimeout(suckInTimerRef.current);
    outputErrorRef.current     = false;
    pendingVizComplete.current = false;
    suckInRef.current.isActive = false;
    setPhase('idle');
    setShowOutput(false);
    setStemVizVisible(false);
    setStatusText('');
    setInputMuted(false);
    const a = inputAudioRef.current;
    if (a) { a.loop = false; a.muted = false; a.pause(); }
    const o = outputAudioRef.current;
    if (o) { o.muted = false; o.pause(); }
    for (const el of Object.values(stemAudioEls.current)) el?.pause();
    selectedStemsRef.current = [];
  }, []);

  // ── Mute toggle for input audio ───────────────────────────────────────────
  const handleToggleInputMute = useCallback(() => {
    const a = inputAudioRef.current;
    if (!a) return;
    const next = !a.muted;
    a.muted = next;
    setInputMuted(next);
  }, []);

  // ── Stem audio routing (mirrors OutputPanel.tsx) ──────────────────────────
  const handleStemSelectionChange = useCallback((stems: string[], uris?: Record<string, string>) => {
    if (uris) {
      stemAudioUrisRef.current = uris;
      for (const name of STEM_NAMES) {
        if (!uris[name]) continue;
        let el = stemAudioEls.current[name];
        if (!el) { el = new Audio(); stemAudioEls.current[name] = el; }
        if (el.src !== uris[name]) el.src = uris[name];
      }
    }
    selectedStemsRef.current = stems;
    const main = outputAudioRef.current;
    if (!main) return;
    const ct = main.currentTime, wasPlaying = !main.paused;
    if (stems.length === 0) {
      main.muted = false;
      for (const el of Object.values(stemAudioEls.current)) el?.pause();
      return;
    }
    main.muted = true;
    for (const name of STEM_NAMES) {
      const el = stemAudioEls.current[name];
      if (!el) continue;
      if (stems.includes(name)) {
        el.currentTime = ct;
        if (wasPlaying) {
          if (el.readyState >= 3) { void el.play().catch(() => {}); }
          else { el.addEventListener('canplaythrough', () => { el.currentTime = ct; void el.play().catch(() => {}); }, { once: true }); }
        }
      } else {
        el.pause();
      }
    }
  }, []);

  const inActivePhase = phase === 'generating' || phase === 'output';
  // Show Demucs status row while waiting (generating and not yet revealed stem viz)
  const showDemucsStatus = phase === 'generating' && !stemVizVisible;

  return (
    <div style={s.shell}>
      <audio ref={inputAudioRef}  src={inputUrl ?? undefined} style={{ display: 'none' }} />
      <audio
        ref={outputAudioRef}
        src={outputUrl ?? undefined}
        controls={showOutput}
        style={{ display: showOutput ? 'block' : 'none', width: '100%', height: 36, flexShrink: 0 }}
      />

      {/* Single header row: uploads + buttons + demucs status (inline to keep viz area stable) */}
      <div style={s.row}>
        <UploadZone id="in-file"  label="Input"  fileName={inputFileName}  onFile={handleInputFile} />
        <UploadZone id="out-file" label="Output" fileName={outputFileName} onFile={handleOutputFile}
          hint="Simulates model output — calls the AI model in production" />
        {inActivePhase ? (
          <>
            <button style={{ ...s.btn, ...s.btnGhost }} onClick={handleReset}>Reset</button>
            {showDemucsStatus && (
              <>
                <span style={s.demucsText}>Analyzing with Demucs…</span>
                <button style={{ ...s.btn, ...s.btnSm }} onClick={handleToggleInputMute}>
                  {inputMuted ? 'Unmute input' : 'Mute input'}
                </button>
              </>
            )}
          </>
        ) : (
          <>
            <button style={{ ...s.btn, opacity: inputUrl  ? 1 : 0.4 }} disabled={!inputUrl}  onClick={handlePlay}>Play</button>
            <button style={{ ...s.btn, opacity: outputUrl ? 1 : 0.4 }} disabled={!outputUrl} onClick={handleGenerate}>Generate</button>
          </>
        )}
        {statusText && <span style={s.statusText}>{statusText}</span>}
      </div>

      {/* Visualization area */}
      {phase === 'idle' ? (
        <div style={s.placeholder}>Upload input and output audio, then click Play or Generate</div>
      ) : (
        <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>

          {/* Input visualization — looping during 'generating', frozen + sucked when Demucs returns */}
          <div style={{
            position: 'absolute', inset: 0, zIndex: 1,
            opacity:      stemVizVisible || showOutput ? 0 : 1,
            pointerEvents: phase !== 'input' ? 'none' : 'auto',
          }}>
            <VisualizationWaitingView
              concatenatedAudioUrl={inputUrl}
              audioRef={inputAudioRef}
              inputs={inputData}
              isVisible={!stemVizVisible && !showOutput}
              suckInRef={suckInRef}
            />
          </div>

          {/* Stem visualization — invisible until suck completes, then fades in */}
          {(phase === 'generating' || phase === 'output') && (
            <div style={{
              position: 'absolute', inset: 0, zIndex: 2,
              opacity:    stemVizVisible ? 1 : 0,
              transition: stemVizVisible ? 'opacity 0.4s ease' : 'none',
              pointerEvents: stemVizVisible ? 'auto' : 'none',
            }}>
              <StemVisualizationView
                audioUrl={outputUrl!}
                inputAudioUrl={inputUrl ?? undefined}
                currentTime={0}
                duration={0}
                isPlaying={false}
                audioRef={outputAudioRef}
                isVisible={stemVizVisible}
                onBackendDone={handleBackendDone}
                onVisualizationComplete={handleVisualizationComplete}
                onVisualizationError={handleVisualizationError}
                onStemSelectionChange={handleStemSelectionChange}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Upload zone ───────────────────────────────────────────────────────────────

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

// ── Styles ────────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  shell: {
    display: 'flex', flexDirection: 'column',
    width: '100vw', height: '100vh',
    background: '#0b0e14', color: '#fff',
    fontFamily: 'Inter, sans-serif',
    boxSizing: 'border-box', overflow: 'hidden',
    padding: 20, gap: 10,
  },
  row:       { display: 'flex', gap: 10, alignItems: 'center', flexShrink: 0 },
  demucsText: { fontSize: 13, color: 'rgba(255,255,255,0.45)', flexShrink: 0 },
  drop: {
    flex: 1, border: '1px dashed rgba(255,255,255,0.15)', borderRadius: 8,
    padding: '10px 14px', cursor: 'pointer', userSelect: 'none', minWidth: 0,
  },
  dropLabel: {
    fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.3)',
    textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 2,
  },
  dropHint: { fontSize: 12, color: 'rgba(255,255,255,0.25)' },
  fileName: { fontSize: 12, color: 'rgba(255,255,255,0.7)' },
  btn: {
    background: 'rgba(255,255,255,0.14)', color: '#fff', border: 'none',
    borderRadius: 6, padding: '7px 20px', fontSize: 13, cursor: 'pointer',
    fontFamily: 'inherit', flexShrink: 0,
  },
  btnSm:    { padding: '5px 14px', fontSize: 12 },
  btnGhost: { background: 'rgba(255,255,255,0.06)' },
  statusText: { fontSize: 13, color: 'rgba(255,255,255,0.4)', flexShrink: 0 },
  placeholder: {
    flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: 'rgba(255,255,255,0.2)', fontSize: 13,
    border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8,
  },
};
