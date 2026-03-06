import React, { useState, useRef } from "react";
import { renderMidiToAudio } from "./utils/midiToAudio";
import { analyzeMidiChords } from "./utils/midiChordAnalysis";
import { MidiVisualizer } from "./components/MidiVisualizer";

export default function App() {
  const [segments, setSegments] = useState([]);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const audioRef = useRef(null);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const buffer = await file.arrayBuffer();
    
    // 1. Analyze Chords for Visualization
    const chordData = await analyzeMidiChords(buffer);
    console.log(chordData);
    setSegments(chordData);

    // 2. Render to PCM Audio for precision timing
    const url = await renderMidiToAudio(buffer);
    setAudioUrl(url);
    setLoading(false);
  };

  return (
    <div style={{ width: "98vw", height: "100vh", padding: "2rem", background: "#0b0e14", minHeight: "100vh", color: "white", fontFamily: "sans-serif" }}>
      <h2>MIDI Visualizer</h2>
      <div style={{ margin: "1rem 0" }}>
        <input type="file" accept=".mid,.midi" onChange={handleUpload} />
      </div>

      {loading && (
        <div style={{ color: "#8ecae6", fontStyle: "italic" }}>
          Generating high-precision audio buffer...
        </div>
      )}
      
      {audioUrl && (
        <div style={{ marginTop: "2rem", height: "60vh" }}>
          <div style={{ marginBottom: "1rem", background: "#1a1d23", padding: "1rem", borderRadius: "8px" }}>
             <audio ref={audioRef} src={audioUrl} controls style={{ width: "100%" }} />
          </div>
          
          <MidiVisualizer 
            segments={segments} 
            audioRef={audioRef} 
          />
        </div>
      )}
    </div>
  );
}