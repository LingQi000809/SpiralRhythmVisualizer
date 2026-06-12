// ============================================================
// EXPERIMENT-ONLY: basic-pitch polyphonic pitch detection.
//
// Import analyzeAudioUrl from THIS module (not visAudioHelpers)
// when you need the pitchDetector parameter.  visAudioHelpers
// exports the prod-synced pitchy-only version.
// ============================================================

import type { FrameFeatures } from './visAudioHelpers';
import {
  analyzeAudioUrl as analyzeAudioUrlPitchy,
  normalizeFeatureArr,
} from './visAudioHelpers';
import Meyda from 'meyda';

export type PitchDetectorType = 'pitchy' | 'basic-pitch';

// ── Model loader (lazy singleton) ─────────────────────────────────────────────

const BASIC_PITCH_MODEL_URL = '/model/basic-pitch/model.json';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _basicPitchInstance: any = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getBasicPitch(): Promise<any> {
  if (!_basicPitchInstance) {
    const { BasicPitch } = await import('@spotify/basic-pitch');
    _basicPitchInstance = new BasicPitch(BASIC_PITCH_MODEL_URL);
  }
  return _basicPitchInstance;
}

// ── Resampler ─────────────────────────────────────────────────────────────────

async function resampleTo22050(buf: AudioBuffer): Promise<Float32Array> {
  const TARGET_SR = 22050;
  if (buf.sampleRate === TARGET_SR) return buf.getChannelData(0).slice();
  const off = new OfflineAudioContext(1, Math.ceil(buf.duration * TARGET_SR), TARGET_SR);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start(0);
  return (await off.startRendering()).getChannelData(0).slice();
}

// ── Basic-pitch analysis ──────────────────────────────────────────────────────

async function analyzeAudioUrlBasicPitch(
  url: string,
  onDone: (features: FrameFeatures[], duration: number) => void,
  isCancelled: () => boolean,
  minRms: number,
): Promise<void> {
  try {
    const buf = await (await fetch(url, { credentials: 'include' })).arrayBuffer();
    if (isCancelled()) return;
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const full = await audioCtx.decodeAudioData(buf);
    void audioCtx.close();
    if (isCancelled()) return;

    const sr        = full.sampleRate;
    const frameSize = 2048;
    const hopSize   = Math.max(512, Math.floor(sr / 20));

    Meyda.sampleRate = sr;
    Meyda.bufferSize = frameSize;
    const ch = full.getChannelData(0);

    // Meyda pass — rms + centroid per frame (used to map basic-pitch notes to energy)
    const rawRms: number[] = [];
    const rawC:   number[] = [];

    for (let i = 0; i < ch.length - frameSize; i += hopSize) {
      const frame = ch.slice(i, i + frameSize);
      const f = Meyda.extract(['spectralCentroid', 'rms'], frame);
      if (!f) continue;
      rawRms.push(f.rms || 0);
      rawC.push(f.spectralCentroid || 0);
      if (rawRms.length % 50 === 0) await new Promise(r => setTimeout(r, 0));
      if (isCancelled()) return;
    }

    const nRms = normalizeFeatureArr(rawRms);
    const nC   = normalizeFeatureArr(rawC);

    // Resample to 22050 Hz for basic-pitch
    const mono22k = await resampleTo22050(full);
    if (isCancelled()) return;

    const bp = await getBasicPitch();
    if (isCancelled()) return;

    const allFrames: number[][] = [];
    const allOnsets: number[][] = [];
    await bp.evaluateModel(
      mono22k,
      (f: number[][], o: number[][], _c: number[][]) => { allFrames.push(...f); allOnsets.push(...o); },
      (_p: number) => {},
    );
    if (isCancelled()) return;

    const { outputToNotesPoly, noteFramesToTime } = await import('@spotify/basic-pitch');
    const notes = noteFramesToTime(outputToNotesPoly(allFrames, allOnsets));

    const { extractMelodyNotes_salience, midiNoteLabel } = await import('./melodicAnalysis');
    const { melody, harmony } = extractMelodyNotes_salience(notes);

    const hopSec = hopSize / sr;

    const toFeature = (note: typeof notes[0], isMelody: boolean): FrameFeatures | null => {
      const fi = Math.min(Math.round(note.startTimeSeconds / hopSec), rawRms.length - 1);
      if ((rawRms[fi] ?? 0) < minRms) return null;
      return {
        time:       note.startTimeSeconds,
        duration:   note.durationSeconds,
        pitch:      note.pitchMidi,
        pitchConf:  note.amplitude,
        rms:        nRms[fi] ?? 0,
        centroid:   nC[fi]  ?? 0,
        isMelody,
        pitchLabel: isMelody ? midiNoteLabel(note.pitchMidi) : undefined,
      };
    };

    const feats: FrameFeatures[] = [
      ...melody.map(n => toFeature(n, true)),
      ...harmony.map(n => toFeature(n, false)),
    ].filter((f): f is FrameFeatures => f !== null);

    if (!isCancelled()) onDone(feats, full.duration);
  } catch (e) {
    console.error('[analyzeAudioUrl:basic-pitch]', e);
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Drop-in replacement for visAudioHelpers.analyzeAudioUrl that also supports
 * basic-pitch polyphonic detection via the pitchDetector parameter.
 *
 * When pitchDetector === 'pitchy' (default), delegates to the prod-synced
 * visAudioHelpers.analyzeAudioUrl with no overhead.
 */
export async function analyzeAudioUrl(
  url: string,
  onDone: (features: FrameFeatures[], duration: number) => void,
  isCancelled: () => boolean,
  minRms = 0.01,
  pitchDetector: PitchDetectorType = 'pitchy',
): Promise<void> {
  if (pitchDetector === 'basic-pitch') {
    return analyzeAudioUrlBasicPitch(url, onDone, isCancelled, minRms);
  }
  return analyzeAudioUrlPitchy(url, onDone, isCancelled, minRms);
}
