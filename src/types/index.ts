export interface FrameFeatures {
  time: number;        // seconds
  duration: number;    // seconds (per note segment)
  pitch: number;       // MIDI
  pitchConf: number;   // confidence about pitch
  rms: number;         // 0-1 normalized
  centroid: number;    // 0-1 normalized
}

export interface MidiFeatures {
  startTime: number;
  duration: number;
  chord: string;

  notes: (MidiNote & {
    isChordTone: boolean;
  })[];
}

export interface MidiNote {
  pitch: number; // MIDI note number (0-127)
  startTime: number; // in sec
  duration: number; // in sec
  velocity: number; // 0-127
}

export interface InputData {
  inputStartTime: number // seconds; start time of the input in the concatenated audio stream
  inputEndTime: number // seconds; end time of the input in the concatenated audio stream
  audioUrl: string // audio URL 
  midiNotes?: MidiNote[] // if the input is MIDI, also include its midi notes
}
