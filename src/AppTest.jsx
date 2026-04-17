import { useState, useRef, useEffect } from "react";
import VoiceRecorder from "./components/VoiceRecorder";
import AudioUploader from "./components/AudioUploader";
import {VisualizationWaitingView} from "./components/VisualizationWaitingView";

export default function AppTest() {
  const audioRef = useRef(null);

  const [audioURL, setAudioURL] = useState(null);
  const [inputs, setInputs] = useState([]);

  useEffect(() => {
    async function loadInputData() {
      const [res] = await Promise.all([
        fetch("/data/input_data.json"),
      ]);

      const data = await res.json();
      setInputs(data);
      setAudioURL("/data/concatenated.wav");
    }

    loadInputData();
  }, []);

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

    {/* Audio Player */}
    <audio
        ref={audioRef}
        src={audioURL}
        controls
        style={{ width: "100%", marginBottom: "1rem" }}
    />

    {/* Visualization */}
    <div style={{ width: "100%", height: "70vh" }}>
        <VisualizationWaitingView
          concatenatedAudioUrl={audioURL}
          audioRef={audioRef}
          inputs={inputs}
          isVisible={!!audioURL}
        />
    </div>
    </div>
  );
}
