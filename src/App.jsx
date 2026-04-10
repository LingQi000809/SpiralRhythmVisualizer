import { useState, useRef, useEffect } from "react";
import VoiceRecorder from "./components/VoiceRecorder";
import AudioUploader from "./components/AudioUploader";
import {VisualizationWaitingView} from "./components/VisualizationWaitingView";

export default function App() {
  const audioRef = useRef(null);

  const [audioURL, setAudioURL] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  async function handleRecordingComplete(blob) {
    setIsRecording(false);
    const url = URL.createObjectURL(blob);
    setAudioURL(url);
  }

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
        onFileSelect={(file, url) => {
          // setAudioFile(file);
          setAudioURL(url);
        }}
      />

      <VoiceRecorder
        onRecordingStart={() => setIsRecording(true)}
        onRecordingComplete={handleRecordingComplete}
      />

      {/* Audio Player */}
      {!isRecording && audioURL && (
        <audio
          ref={audioRef}
          src={audioURL}
          controls
          style={{ width: "100%", marginBottom: "1rem" }}
        />
      )}

      {/* Visualization */}
      {!isRecording && (
        <div style={{ width: "100%", height: "70vh" }}>
          <VisualizationWaitingView
            audioUrl={audioURL}
            audioRef={audioRef}
            isVisible={!isRecording && audioURL}
          />
        </div>
      )}
    </div>
  );
}
