import { Midi } from "@tonejs/midi";

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const CHORD_TEMPLATES = {
    maj: [0, 4, 7],
    min: [0, 3, 7],
    dim: [0, 3, 6],
    aug: [0, 4, 8],
    maj7: [0, 4, 7, 11],
    min7: [0, 3, 7, 10],
    dom7: [0, 4, 7, 10],
    sus4: [0, 5, 7]
};

// ==========================
// Step 1: Parse MIDI
// ==========================
function parseMidiNotes(midiBuffer) {
    const midi = new Midi(midiBuffer);
    let notes = [];
    midi.tracks.forEach(track => {
        track.notes.forEach(n => {
            notes.push({
                midi: n.midi,
                startTime: n.time,
                duration: n.duration,
                velocity: 127
            });
        });
    });
    notes = notes.sort((a, b) => a.noteOn_sec - b.noteOn_sec);
    return notes;
}

// ==========================
// Step 2: Pitch Scoring 
// ==========================
// NEW CONSTANT: How far back should we remember notes? 
// 1.0 to 1.5 seconds is usually the sweet spot for slow arpeggios.
const MEMORY_SEC = 1.5;

function detectChord(allNotes, winStart, winEnd) {
    const profile = new Array(12).fill(0);
    let totalWeight = 0;

    // 1. ANALYZE HISTORY: Look at notes that started before now but are "recent"
    const lookbackStart = winStart - MEMORY_SEC;

    allNotes.forEach(n => {
        // We care about notes that overlap with our "Memory Window"
        const overlapStart = Math.max(lookbackStart, n.noteOn_sec);
        const overlapEnd = Math.min(winEnd, n.noteOff_sec);

        let weight = overlapEnd - overlapStart;

        if (weight > 0) {
            // DECAY FACTOR: Notes that happened longer ago are worth less
            // This prevents the chord from getting "stuck" on old data
            const recency = 1 - (winEnd - n.noteOn_sec) / MEMORY_SEC;
            const effectiveWeight = weight * Math.max(0.2, recency);

            profile[n.midi % 12] += effectiveWeight;
            totalWeight += effectiveWeight;
        }
    });

    let best = { label: "N", finalScore: -1, root: 0, type: "maj" };

    for (let root = 0; root < 12; root++) {
        for (const [type, template] of Object.entries(CHORD_TEMPLATES)) {
            let chordWeight = 0;
            let matchCount = 0;

            template.forEach(interval => {
                const pc = (root + interval) % 12;
                if (profile[pc] > 0) {
                    chordWeight += profile[pc];
                    matchCount++;
                }
            });

            const coverage = matchCount / template.length;
            const purity = chordWeight / (totalWeight || 1);

            // Complexity bonus: slightly penalize 7ths unless evidence is very strong
            // to prevent arpeggiated triads from being seen as 7ths.
            const complexityPenalty = template.length === 4 ? 0.9 : 1.0;

            const finalScore = chordWeight * coverage * purity * complexityPenalty;

            if (finalScore > best.finalScore) {
                best = { label: NOTE_NAMES[root] + type, finalScore, root, type };
            }
        }
    }

    // Threshold check
    return best.finalScore > 0.3 ? best : { label: "N", finalScore: 0 };
}

// ==========================
// Step 3: Pipeline
// ==========================
export function analyzeMidiChords(notes) {
    if (notes.length === 0) return [];

    const endTime = notes[-1].startTime + notes[-1].duration;
    const results = [];

    // Increased window size for stability
    const windowSize = 0.4;
    const step = 0.2;

    for (let t = 0; t < endTime; t += step) {
        const winStart = t;
        const winEnd = t + windowSize;

        const activeNotes = notes.filter(n => n.noteOn_sec < winEnd && n.noteOff_sec > winStart);
        if (activeNotes.length === 0) continue;

        const bestChord = detectChord(activeNotes, winStart, winEnd);
        console.log(bestChord);

        // Map chord tones
        const pcs = bestChord.label === "N" ? [] :
            CHORD_TEMPLATES[bestChord.type].map(i => (bestChord.root + i) % 12);

        results.push({
            noteOn_sec: winStart,
            noteOff_sec: winEnd,
            chord: bestChord.label,
            notes: activeNotes.map(n => ({
                ...n,
                isChordTone: pcs.includes(n.midi % 12)
            }))
        });
    }

    // Merge and Deduplicate 
    const merged = [];
    for (const seg of results) {
        const last = merged[merged.length - 1];
        if (last && last.chord === seg.chord) {
            last.noteOff_sec = seg.noteOff_sec;

            // Deduplicate notes by creating a unique set based on midi + start time
            const combinedNotes = [...last.notes, ...seg.notes];
            const seen = new Set();
            last.notes = combinedNotes.filter(n => {
                const id = `${n.midi}-${n.noteOn_sec}`;
                return seen.has(id) ? false : seen.add(id);
            });
        } else {
            merged.push(seg);
        }
    }

    return merged.filter(s => (s.noteOff_sec - s.noteOn_sec) > 0.1);
}