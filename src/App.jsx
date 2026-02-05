import React, { useRef, useState } from "react";
import AudioUploader from "./components/AudioUploader";
import SpiralRhythmVisualizer from "./components/SpiralRhythmVisualizer";

export default function App() {
  const audioRef = useRef(null);

  const [rhythmData, setRhythmData] = useState(null);

  const handleAudioData = (data) => {
    setRhythmData(data);
  };

  return (
    <div
      style={{
        width: "100vw",
        minHeight: "100vh",
        padding: "2rem",
        color: "white",
        background: "#0b0e14",
        boxSizing: "border-box",
        overflow: "hidden",
      }}
    >

      {/* Upload + audio player */}
      <AudioUploader
        audioRef={audioRef}
        onAudioData={handleAudioData}
      />

      {/* Visualization */}
      {rhythmData && (
        <div
          style={{
            width: "100%",
            height: "60vh",
            marginTop: "2rem",
          }}
        >
          <SpiralRhythmVisualizer
            audioRef={audioRef}
            rhythmData={rhythmData}
          />
        </div>
      )}
    </div>
  );
}
