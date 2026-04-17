import { Midi } from "@tonejs/midi";
import { MidiFeatures, MidiNote, InputData } from "../types";

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

const CHORD_TEMPLATES: Record<string, number[]> = {
  maj: [0, 4, 7],
  min: [0, 3, 7],
  dim: [0, 3, 6],
  aug: [0, 4, 8],
  maj7: [0, 4, 7, 11],
  min7: [0, 3, 7, 10],
  dom7: [0, 4, 7, 10],
  sus4: [0, 5, 7],
};

type ChordResult = {
  label: string;
  finalScore: number;
  root: number;
  type: keyof typeof CHORD_TEMPLATES | "maj";
};

// ==========================
// Pitch Scoring 
// ==========================
// MEMORY_SEC: How far back should we remember notes? 
// 1.0 to 1.5 seconds is usually the sweet spot for slow arpeggios.
const MEMORY_SEC = 2;

function detectChord(
  notes: MidiNote[],
  winStart: number,
  winEnd: number
): ChordResult {
  const profile = new Array(12).fill(0);
  let totalWeight = 0;

  const lookbackStart = winStart - MEMORY_SEC;

  for (const n of notes) {
    
    const overlapStart = Math.max(lookbackStart, n.startTime);
    const overlapEnd = Math.min(winEnd, n.startTime + n.duration);

    const weight = overlapEnd - overlapStart;
    if (weight <= 0) continue;

    const recency = 1 - (winEnd - n.startTime) / MEMORY_SEC;
    const effectiveWeight = weight * Math.max(0.2, recency);

    profile[n.pitch % 12] += effectiveWeight;
    totalWeight += effectiveWeight;
  }

  let best: ChordResult = {
    label: "N",
    finalScore: 0,
    root: 0,
    type: "maj",
  };

  for (let root = 0; root < 12; root++) {
    for (const [type, template] of Object.entries(CHORD_TEMPLATES)) {
      let chordWeight = 0;
      let matchCount = 0;

      for (const interval of template) {
        const pc = (root + interval) % 12;
        if (profile[pc] > 0) {
          chordWeight += profile[pc];
          matchCount++;
        }
      }

      const coverage = matchCount / template.length;
      const purity = chordWeight / (totalWeight || 1);
      const complexityPenalty = template.length === 4 ? 0.9 : 1.0;

      const score = chordWeight * coverage * purity * complexityPenalty;

      if (score > best.finalScore) {
        best = {
          label: NOTE_NAMES[root] + type,
          finalScore: score,
          root,
          type: type as keyof typeof CHORD_TEMPLATES,
        };
      }
    }
  }

  return best.finalScore > 0.3
    ? best
    : { label: "N", finalScore: 0, root: 0, type: "maj" };
}

export function analyzeMidi(input: InputData) {
    const notes = input.midiNotes;
    if (!notes?.length) return [];

    const windowSize = 0.4;
    const step = 0.2;

    const results: MidiFeatures[] = [];

    for (let t = input.inputStartTime; t < input.inputEndTime; t += step) {
        const winStart = t;
        const winEnd = t + windowSize;

        const activeNotes = notes.filter(
            (n) =>
            n.startTime < winEnd &&
            n.startTime + n.duration > winStart
        );

        if (!activeNotes.length) continue;

        const bestChord = detectChord(activeNotes, winStart, winEnd);

        const chordPcs =
            bestChord.label === "N"
            ? []
            : CHORD_TEMPLATES[bestChord.type].map(
                (i) => (bestChord.root + i) % 12
                );

        results.push({
            startTime: winStart,
            duration: windowSize,
            chord: bestChord.label,
            notes: activeNotes.map((n) => ({
                ...n,
                isChordTone: chordPcs.includes(n.pitch % 12),
            })),
        });
    }

    const merged: MidiFeatures[] = [];

    for (const seg of results) {
    const last = merged[merged.length - 1];

    if (last && last.chord === seg.chord) {
        last.duration += seg.duration;

        const combined = [...last.notes, ...seg.notes];
        const seen = new Set<string>();

        last.notes = combined.filter((n) => {
        const id = `${n.pitch}-${n.startTime}`;
        if (seen.has(id)) return false;
        seen.add(id);
        return true;
        });
    } else {
        merged.push({ ...seg });
    }
    }

    return merged.filter((s) => s.duration > 1);
}
