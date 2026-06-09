// Melodic analysis post-processing for basic-pitch output.
// Separates the salient melody voice from background harmony notes.

import type { NoteEventTime } from '@spotify/basic-pitch';

// ─── Note naming ──────────────────────────────────────────────────────────────

const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'] as const;

export function midiNoteLabel(midi: number): string {
  return NOTE_NAMES[midi % 12];  // pitch class only: "A", "C#", etc.
}

// ─── Melody extraction ────────────────────────────────────────────────────────

// Notes below this absolute amplitude are too quiet to be melody candidates.
// Keeps faint overtones and model artifacts out of the melody voice.
const MIN_AMP = 0.08;

function overlaps(note: NoteEventTime, other: NoteEventTime): boolean {
  const noteEnd  = note.startTimeSeconds  + note.durationSeconds;
  const otherEnd = other.startTimeSeconds + other.durationSeconds;
  return other.startTimeSeconds < noteEnd && otherEnd > note.startTimeSeconds;
}

/**
 * Splits basic-pitch note events into melody (always the outermost audible voice)
 * and harmony (everything else).
 *
 * Rule: a note is melody if no concurrent note is both higher-pitched AND above
 * MIN_AMP. This guarantees the highest audible pitch at any moment is always
 * the melody, matching what you see as "outermost" on the orbital ring.
 * Notes below MIN_AMP are unconditionally harmony (too faint to be melody).
 */
export function extractMelodyNotes_salience(notes: NoteEventTime[]): {
  melody:  NoteEventTime[];
  harmony: NoteEventTime[];
} {
  const melody:  NoteEventTime[] = [];
  const harmony: NoteEventTime[] = [];

  for (const note of notes) {
    if (note.amplitude < MIN_AMP) { harmony.push(note); continue; }

    const dominated = notes.some(other =>
      other !== note &&
      other.amplitude >= MIN_AMP &&
      other.pitchMidi > note.pitchMidi &&
      overlaps(note, other),
    );

    (dominated ? harmony : melody).push(note);
  }

  return { melody, harmony };
}
