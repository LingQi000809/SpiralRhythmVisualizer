// ============================================================
// SYNCED WITH: Hearmi-Frontend/app/utils/api.ts
// Differences vs production:
//   - API_BASE_URL points to local backend (localhost:8000)
//   - Only the types and functions needed for StemVisualizationView are included
// Keep all exported types byte-for-byte identical to production.
// ============================================================

const API_BASE_URL = 'http://localhost:8000';

// ── Types (identical to production) ──────────────────────────────────────────

export interface VizFrame {
  pos: [number, number, number]; // 3D position (UMAP in production; placeholder locally)
  rms: number;                   // Energy level, normalized 0-1
  centroid: number;              // Spectral centroid, normalized 0-1
  time: number;                  // Seconds
  stem: 'drums' | 'bass' | 'vocals' | 'other' | 'silence';
}

export interface ChordInfo {
  chord: string;
  start: number;
  end: number;
  confidence: number;
  dissonance: number;
  tension?: number;
}

export interface KeyInfo {
  key_name: string;
  key_root: number;
  key_mode: string;
}

export type SongSectionLabel =
  | 'intro' | 'verse' | 'pre-chorus' | 'chorus'
  | 'bridge' | 'outro' | 'instrumental';

export interface SongSection {
  section: SongSectionLabel;
  start: number;
  end: number;
}

export interface FlamingoChord {
  chord: string;
  start: number;
  end: number;
}

export interface BassRootPoint {
  time: number;
  midi: number;
  note: string;
  pc: number;
}

export interface VocalMelodyPoint {
  time: number;
  midi: number;
  note: string;
  freq: number;
}

export interface VisualizationData {
  frames: VizFrame[];
  duration: number;
  chords: ChordInfo[];
  key: KeyInfo;
  songStructure: SongSection[];
  flamingoChords: FlamingoChord[];
  vocalMelodyContour: VocalMelodyPoint[];
  bassRootContour: BassRootPoint[];
  audioDataUri: string;
  stemAudioUris: {
    drums?: string;
    bass?: string;
    vocals?: string;
    other?: string;
  };
}

export type VisualizationCacheData = Pick<
  VisualizationData,
  | 'frames'
  | 'duration'
  | 'chords'
  | 'key'
  | 'songStructure'
  | 'flamingoChords'
  | 'bassRootContour'
  | 'vocalMelodyContour'
> & {
  stemAudioUris?: VisualizationData['stemAudioUris'];
  // Appwrite storage IDs for per-stem audio (production only; unused locally)
  stemAudioFileIds?: {
    drums?: string;
    bass?: string;
    vocals?: string;
    other?: string;
  };
};

export interface ProcessVisualizationResponse {
  data: VisualizationData | null;
  success: boolean;
  error?: string;
}

// ── Helpers (identical to production) ────────────────────────────────────────

export const toVisualizationCacheData = (data: VisualizationData): VisualizationCacheData => ({
  frames:               data.frames,
  duration:             data.duration,
  chords:               data.chords,
  key:                  data.key,
  songStructure:        data.songStructure,
  flamingoChords:       data.flamingoChords,
  bassRootContour:      data.bassRootContour,
  vocalMelodyContour:   data.vocalMelodyContour,
  stemAudioUris:        data.stemAudioUris,
});

// ── API call (identical to production) ───────────────────────────────────────

// ── Similarity detection (streaming SSE) ─────────────────────────────────────

export interface BackendSimilarityMatch {
  input_start:  number;
  input_end:    number;
  output_start: number;
  output_end:   number;
  similarity:   number;
  // Visualization data from backend
  in_note_times:    number[];
  in_note_pitches:  number[];
  out_note_times:   number[];
  out_note_pitches: number[];
  in_match_slice:   [number, number];
  out_match_slice:  [number, number];
  dtw_path:         [number, number][];
}

/**
 * POST both audio files to /similarity and stream progress events back.
 * Resolves with the list of matches when the backend signals "done".
 */
export async function streamSimilarity(
  inputFile:  Blob,
  outputFile: Blob,
  onProgress: (msg: string) => void,
): Promise<BackendSimilarityMatch[]> {
  const form = new FormData();
  form.append('input_audio',  inputFile,  'input.wav');
  form.append('output_audio', outputFile, 'output.wav');

  const response = await fetch(`${API_BASE_URL}/similarity`, { method: 'POST', body: form });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const reader  = response.body!.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const event = JSON.parse(line.slice(6)) as { type: string; msg?: string; matches?: BackendSimilarityMatch[] };
      if (event.type === 'progress' && event.msg) onProgress(event.msg);
      if (event.type === 'done')   return event.matches ?? [];
      if (event.type === 'error')  throw new Error(event.msg ?? 'Unknown error');
    }
  }
  return [];
}

export async function processVisualization(
  audioFile: File
): Promise<ProcessVisualizationResponse> {
  try {
    const formData = new FormData();
    formData.append('audio', audioFile);

    const response = await fetch(`${API_BASE_URL}/process`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error((err as any).error || `HTTP ${response.status}`);
    }

    const data: VisualizationData = await response.json();
    return { data, success: true };
  } catch (error) {
    console.error('[processVisualization]', error);
    return { data: null, success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}
